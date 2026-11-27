"""Grade the nominated source-only splits and their direction ablations."""
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image,ImageDraw
from benchmark import expanded_fixed_handoffs as fixed, expanded_scores as grades
from benchmark.visible_boundary_capture import BASE,RUN,CASES,read,save,configure
from benchmark.visible_boundary_screen import entries,inputs
from benchmark.visible_boundary_proposal import apply
from benchmark.expanded_baseline import replay_source_fields
from engine import surface_model,inset_returns
from engine.pillar_seams import native_mask,orientation


def signed_row(mask, normal, rotation, transform, world_normals):
    local=rotation.T@normal
    axis=int(np.argmax(abs(local)))
    direction=np.eye(3)[axis]*np.sign(local[axis])
    aligned=transform[:3,:3]@direction
    cardinal=np.zeros(3)
    k=int(np.argmax(abs(aligned)))
    cardinal[k]=np.sign(aligned[k])
    return int(np.sum(np.all(world_normals[mask]==cardinal,axis=1)))


def audit(image,parts,camera,event,labels,face_ids,transform):
    lookup=np.array([[0,0,0]]+[f['normal_world'] for f in labels['faces']])
    world_normals=lookup[face_ids]
    surface=np.array([0]+[f['surface_id'] for f in labels['faces']])[face_ids]
    split=apply(image,parts,camera,event,'both' if event.get('family','corner')=='corner' else 'boundary')[event['index']:event['index']+2]
    mask=native_mask(parts[event['index']],image.size)
    masks=[native_mask(p,image.size) for p in split]
    assert np.array_equal(mask,masks[0]|masks[1]) and not (masks[0]&masks[1]).any()
    counts=[np.bincount(surface[m],minlength=int(surface.max())+1) for m in masks]
    _,rot=orientation(camera,image.size)
    old=parts[event['index']]['normal_camera']
    def count(normals):
        return sum(signed_row(m,n,rot,transform,world_normals) for m,n in zip(masks,normals))
    dominants=[int(np.argmax(c[1:])+1) for c in counts]
    faces=[next(f for f in labels['faces'] if f['surface_id']==k) for k in dominants]
    normals=[np.array(f['normal_world']) for f in faces]
    heights=[float(np.median(np.array(f['corners_world'])@n)) for f,n in zip(faces,normals)]
    same_sign=bool(np.array_equal(normals[0],normals[1]))
    return dict(majority_normals=[n.tolist() for n in normals],majority_plane_levels=heights,
                different_parallel_planes=same_sign and abs(heights[1]-heights[0])>.01,
                area=int(mask.sum()),background=int(np.sum(mask & (face_ids==0))),
                signed_correct=dict(baseline=count([old,old]),boundary=count([old,old]),
                    direction=count([event['new_normal']]*2),both=count([p['normal_camera'] for p in split])),
                frontmost_surface_majority_before=int(sum(counts)[1:].max()),
                frontmost_surface_majority_after=sum(int(c[1:].max()) for c in counts),
                children_dominant_surfaces=[int(np.argmax(c[1:])+1) for c in counts])


def preview(image,parts,camera,event,path):
    before=image.copy()
    split=apply(image,parts,camera,event,'both' if event.get('family','corner')=='corner' else 'boundary')[event['index']:event['index']+2]
    colors=((48,205,240),(245,171,62))
    overlay=np.asarray(before).copy()
    for part,col in zip(split,colors):
        m=native_mask(part,image.size)
        overlay[m]=overlay[m]*.68+np.array(col)*.32
    a,b,c,d=event['box']
    rect=(max(0,a-20),max(0,b-20),min(image.width,c+20),min(image.height,d+20))
    panels=[before.crop(rect),Image.fromarray(overlay).crop(rect)]
    for panel in panels:
        panel.thumbnail((390,390))
    output=Image.new('RGB',(panels[0].width*2,panels[0].height+24),'white')
    draw=ImageDraw.Draw(output)
    for i,panel in enumerate(panels):
        output.paste(panel,(i*panel.width,24))
    draw.text((2,4),'Source crop',fill='black')
    draw.text((panels[0].width+2,4),'Split',fill='black')
    output.save(path)


def evaluate(task):
    entry,family=task
    folder=RUN/entry['dataset']/entry['id']
    event=read(folder/('proposal.json' if family=='corner' else 'parallel-proposal.json'))
    if family=='parallel':
        folder=folder/'parallel'
        folder.mkdir(exist_ok=True)
    im,parts,cam=inputs(entry)
    results={}
    reference=None
    if entry['dataset']==fixed.FIXED_DATASET:
        reference=fixed.fixed_reference(entry)
        labels,ids,transform=reference
        truth={tuple(b['xyz_world']) for b in labels['blocks']}
        with ZipFile(fixed._release()[2]) as archive:
            pack=archive.read(entry['id']+'/reconstruction.zip')
        base=next(r for r in read(CASES) if r['dataset']==entry['dataset'] and r['id']==entry['id'])
        local=fixed.decoded_pack(BytesIO(pack))
        basepoints=set(fixed.typed.aligned_sites(list(local),transform,base['translation']))
        results['baseline']=base
    else:
        baseline=BASE/entry['dataset']/entry['id']
        ref=grades.load_reference(entry)
        truth=set(ref.truth)
        labels=read(Path(entry['frame'])/'labels.json')
        with np.load(Path(entry['frame'])/'labels.npz') as z:
            ids=z['face_id']
        transform=grades.registration(np.asarray(cam['P']),ref.pose)
        def grade(path):
            manifest=read(path/'build-manifest.json')
            blocks=read(path/'placements.json')['blocks']
            assignment={(b['x'],b['y'],b['z']):b['material_source'] for b in blocks}
            return grades.evaluate_pack(entry,path/'reconstruction.zip',manifest['placement'],assignment=assignment,reference=ref)
        results['baseline'],basepoints=grade(baseline)
        basepoints=set(basepoints)
    preview(im,parts,cam,event,folder/'preview.jpg')
    stage=audit(im,parts,cam,event,labels,ids,transform)
    delta={}
    for arm in (('boundary','direction','both') if family=='corner' else ('boundary',)):
        if reference is not None:
            altered=fixed.inference_inputs(entry,boundary=lambda image,parts:apply(image,parts,cam,event,arm))
            result=fixed.replay_placement(altered)
            row,points=fixed.score_result(entry,result,reference=reference)
            (folder/(arm+'.zip')).write_bytes(result.pack)
        else:
            output=folder/arm
            original=inset_returns.detect
            injected=False
            def modify(image,live,*args,**kwargs):
                nonlocal injected
                if not injected:
                    assert [p['id'] for p in parts]==[p['id'] for p in live]
                    assert parts[event['index']]['support']==live[event['index']]['support']
                    live[:]=apply(image,live,cam,event,arm)
                    injected=True
                return original(image,live,*args,**kwargs)
            with replay_source_fields(Path(entry['frame'])/'rgb.png',baseline),patch.object(inset_returns,'detect',modify):
                surface_model.build(Path(entry['frame'])/'rgb.png','26.3',output,through='placement',atlas_path=fixed.suite.ATLAS)
            assert injected
            row,points=grade(output)
        points=set(points)
        delta[arm]=dict(gained=len((points-basepoints)&truth),lost=len((basepoints-points)&truth),
                        extras_added=len(points-basepoints-truth),extras_removed=len(basepoints-points-truth))
        results[arm]=row
        save(folder/(arm+'-score.json'),row)
    save(folder/'result.json',dict(proposal=event,stage=stage,arms=results,deltas=delta))
    return entry['id']


if __name__=='__main__':
    configure()
    import sys
    family=sys.argv[1] if len(sys.argv)>1 else 'corner'
    filename='proposal.json' if family=='corner' else 'parallel-proposal.json'
    todo=[(e,family) for e in entries() if read(RUN/e['dataset']/e['id']/filename) and (len(sys.argv)<3 or e['id'] in sys.argv[2:])]
    with ProcessPoolExecutor(2 if family=="parallel" else 3,initializer=configure) as pool:
        for key in pool.map(evaluate,todo):
            print(key,flush=True)
