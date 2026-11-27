"""Locate new lattice proposals in the decoded pack and audit visibly supported anchor size and direction."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import argparse
import numpy as np
from PIL import ImageDraw,Image
from benchmark import full_block_lattice_run as run
from benchmark.expanded_baseline import SOURCE,frames,read,save
from benchmark.camera_metrics import registration
from benchmark.full_block_lattice import line_camera
from benchmark.mask_blocks import decoded_pack
from benchmark.typed_structures import aligned_sites
from engine.geometry import project,transform


def observe(entry):
    dataset,key=entry['dataset'],entry['id']
    out=run.RUN/dataset/key
    evidence=read(out/'evidence.json.gz')
    row=read(out/'result.json')
    root=run.BASE/dataset/key
    anchors=evidence.get('anchors',[])
    if not anchors:
        save(out/'attribution.json',dict(id=key,dataset=dataset,cells=[],anchors=[],rescued=False))
        return
    # Source proposal fields are frozen before opening any corresponding reference data.
    delta=decoded_pack(out/'proposal.zip').keys()-decoded_pack(root/'reconstruction.zip').keys()
    hand=read(root/'handoffs/placement-input.json.gz')
    P=(hand['camera'] or {}).get('P')
    eye=np.array(evidence['eye'])
    if P is None:
        segments=read(root/'handoffs/mlsd.json.gz')['segments']
        linear=line_camera(segments,(1440,900))
        P=np.column_stack((linear,-linear@eye))
    P=np.array(P)
    shared=set(map(tuple,evidence['shared']))
    projected={}
    for a in anchors:
        ax=a['axis']
        center=np.array(a['xyz']).mean(0)+eye
        center[ax]-=np.sign(eye[ax]-center[ax])*.02
        cell=tuple(np.floor(center+1e-7).astype(int).tolist())
        projected[cell]=a
    pose=read(Path(entry['frame'])/'camera.json')
    labels=read(Path(entry['frame'])/'labels.json')
    with np.load(Path(entry['frame'])/'labels.npz') as native:
        raster=native['face_id']
    truth={tuple(b['xyz_world']):b.get('geometry_kind','full_cube') for b in labels['blocks']}
    T=registration(P,pose)
    shift=row['proposal']['translation']
    cells=sorted(delta)
    coords=aligned_sites(cells,T,shift)
    roles=[]
    for local,world in zip(cells,coords):
        group='multiple_surfaces' if local in shared else 'verified_sprite' if local in projected else 'neighbor_only'
        roles.append(dict(local=local,world=world,group=group,kind=truth.get(world,'extra')))
    hits=[]
    for cell,a in projected.items():
        quad=np.array(a['quad'])
        t=(np.arange(7)+.5)/7
        xy=project(transform(quad),np.stack(np.meshgrid(t,t),axis=-1))
        x,y=np.moveaxis(np.floor(xy).astype(int),-1,0)
        owner,count=Counter(raster[y.clip(0,899),x.clip(0,1439)].ravel()).most_common(1)[0]
        if not owner:
            continue
        f=labels['faces'][owner-1]
        block=labels['blocks'][f['block']-1]
        if block.get('geometry_kind','full_cube')!='full_cube' or count<.8*49:
            continue
        q=np.array(f['corners_px'])
        corners_error=min(np.linalg.norm(quad-np.roll(order,i,axis=0),axis=1).mean()
                          for order in (q,q[::-1]) for i in range(4))
        size=np.linalg.norm(q-np.roll(q,1,axis=0),axis=1).mean()
        world=aligned_sites([cell],T,shift)[0]
        hits.append(dict(scale_ratio=float(np.linalg.norm(quad-np.roll(quad,1,axis=0),axis=1).mean()/size),
                    corner_error_blocks=float(corners_error/size),
                    axis_correct=int(np.argmax(abs(T[:3,a['axis']]))==np.argmax(np.abs(f['normal_world']))),
                    intended_owner=tuple(block['xyz_world'])==world,owner=block['xyz_world'],world=world))
    extent=(np.ptp(np.array(coords),axis=0)+1).tolist() if coords else []
    truth_extent=(np.ptp(np.array(list(truth)),axis=0)+1).tolist()
    save(out/'attribution.json',dict(id=key,dataset=dataset,cells=roles,anchors=hits,accepted=row['changed'],
                                     rescued=row['rescued'],camera_before=row['baseline']['camera_usable'],
                                     added_extent=extent,visible_truth_extent=truth_extent))
    return key


def summarize():
    rows=[read(p) for p in run.RUN.glob('*/*/attribution.json')]
    summary={}
    for dataset in run.COHORTS:
        named={}
        for name,group in [('source',[r for r in rows if r['dataset']==dataset and r.get('accepted')]),
                           ('rescue_only',[r for r in rows if r['dataset']==dataset and r['rescued']])]:
            cells=[cell for r in group for cell in r['cells']]
            cats=defaultdict(Counter)
            for c in cells:
                cats[c['group']][c['kind']]+=1
            hits=[h for r in group for h in r['anchors']]
            named[name]=dict(cases=len(group),added=len(cells),groups=dict(cats),dominant_cube_anchors=len(hits),
                direction_correct=sum(h['axis_correct'] for h in hits),correct_intended_owner=sum(h['intended_owner'] for h in hits),
                median_span_ratio=float(np.median([h['scale_ratio'] for h in hits])) if hits else None,
                median_corner_error_blocks=float(np.median([h['corner_error_blocks'] for h in hits])) if hits else None)
        summary[dataset]=named
    save(run.REVIEW/'attribution.json',summary)
    save(run.REVIEW/'coordinates.json.gz',rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    entries=list(frames(SOURCE,run.COHORTS))
    with ProcessPoolExecutor(a.workers) as pool:
        for _ in pool.map(observe,[e for e in entries if not (run.RUN/e['dataset']/e['id']/'attribution.json').exists()]):
            pass
    summarize()


if __name__=='__main__':
    main()
