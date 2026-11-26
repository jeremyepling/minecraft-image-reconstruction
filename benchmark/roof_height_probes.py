"""Score the selected roof height independently of final placement on the 18 canopy photographs."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark import framed_depth_eval as framed
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked, digest, read, save
from engine.surface_perimeter import _native_alpha
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE
from training.visible_labels import visible_labels


def labels(frame, camera):
    path=framed.transfer.rgb_path(frame)
    pose=read(checked(path.with_suffix('.json'),frame['files']['camera.json']))
    raw=read(checked(path.with_name(frame['source_id']+'-blocks.json'),frame['files']['world-readback.json']))
    states={tuple(b[k] for k in 'xyz'):b for b in raw}
    truth,ids,_=visible_labels(pose,states,(0,0,0))
    course=framed.roof_specs()[frame['id']]['roof_y']
    roof=[f for f in truth['faces'] if f['xyz_world'][1]==course]
    local=int(course-registration(np.asarray(camera['P']),pose)[1,3])
    visible=np.isin(ids,[f['id'] for f in roof])
    horizontal=np.isin(ids,[f['id'] for f in roof if f['normal_world'][1]])
    return local,visible,horizontal


def prior_measure(cells, camera, target, roof):
    # This simple guess can be made before opening any reference. It need not correspond to a roof.
    courses=Counter(y for x,y,z in cells if y>0)
    modal=max(courses,key=lambda h:(courses[h],h)) if courses else None
    # The following diagnostic uses the reference roof pixel mask solely to score the frozen export.
    P=np.array(camera['P'],dtype=float,copy=True)
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    P[:2]*=SAMPLE_STRIDE
    coords=sorted(cells)
    owner,axes=_frontmost_cells(coords,P,eye,roof.shape,face_axes=True)
    y,x=np.nonzero(roof & (owner>=0))
    occupied_height=np.asarray(coords,dtype=int)[owner[y,x],1] if len(y) else np.array([],dtype=int)
    votes=Counter(map(int,occupied_height))
    return dict(largest_elevated_layer=modal,largest_layer_exact=modal==target,
                tallest_layer=max(courses,default=None),
                horizontal_reference_pixels=int(roof.sum()),pixels_covered=len(x),
                pixels_with_expected_height=votes[target],
                pixels_with_expected_height_and_horizontal_face=int(((occupied_height==target)&(axes[y,x]==1)).sum()),
                covering_pixel_height_histogram=dict(sorted(votes.items())))


def decision_score(decision,parts,size,roof,target,camera):
    if not decision:
        return dict(part=None,block_y=None,exact=False,associated=False)
    part=next((p for p in parts if p['id']==decision['part']),None)
    if part is None:
        raise ValueError(f"Unknown predicted surface {decision['part']}")
    course=decision.get('block_y')
    if course is None:
        plane=decision['plane_y']
        P=np.asarray(camera['P'])
        eye=np.linalg.solve(P[:,:3],-P[:,3])
        course=plane-int(eye[1]>plane)
    pixels=_native_alpha(part['support'],size)
    count=overlap=0
    if pixels:
        alpha,(x0,y0,x1,y1)=pixels
        mask=np.asarray(alpha).astype(bool)
        count=int(mask.sum())
        overlap=int((roof[y0:y1,x0:x1]&mask).sum())
    return dict(part=part['id'],block_y=int(course),exact=course==target,
                associated=overlap>=16 and overlap>=.5*count,
                overlap_pixels=overlap,part_pixels=count)


def report(runs,out):
    indexes={}
    fingerprints={}
    for name,folder in runs:
        summary=read(folder/'summary.json')
        indexes[name]={(r['split'],r['id']):r for r in read(checked(folder/'cases.json',summary['cases_sha256']))}
        fingerprints[name]=summary['cases_sha256']
    data=[]
    for split,frame,prior in framed.cases(('old','fresh','extra')):
        key=frame['id']
        if key not in framed.roof_specs():
            continue
        image,parts,_,camera=framed.photo(split,frame,prior)
        course,roof,horizontal=labels(frame,camera)
        row=dict(id=key,split=split,world_readback_sha256=frame['files']['world-readback.json'],
                 rgb_sha256=frame['files']['rgb.png'],target_block_y=course,methods={})
        for name,folder in runs:
            measured=indexes[name][split,key]
            assert measured['rgb_sha256']==row['rgb_sha256']
            pack=checked(folder/split/key/'reconstruction.zip',measured['pack_sha256'])
            if name=='control':
                row['prior']=prior_measure(set(decoded_pack(pack)),camera,course,horizontal)
            status=measured['status']
            primary=decision_score(status.get('primary_height'),parts,image.size,roof,course,camera)
            all_accepted=[decision_score(d,parts,image.size,roof,course,camera)
                          for d in status.get('height_decisions',[]) if d.get('accepted')]
            row['methods'][name]=dict(primary=primary,accepted=all_accepted,
                                      anchored=measured['roof']['spacing']['anchored_roof_cells'],
                                      visible=measured['roof']['spacing']['visible_roof_cells'],
                                      f1=measured['f1'])
        data.append(row)
    assert len(data)==18 and len({r['id'] for r in data})==18
    aggregates={}
    for label,rr in [('fixed',[r for r in data if r['split']!='extra']),('alternate',[r for r in data if r['split']=='extra'])]:
        agg=dict(frames=len(rr),target_heights=dict(sorted(Counter(r['target_block_y'] for r in rr).items())),
                 largest_layer_exact=sum(r['prior']['largest_layer_exact'] for r in rr),methods={})
        for name,_ in runs:
            m=[r['methods'][name] for r in rr]
            top=[v['primary'] for v in m]
            matched=[hit for v in m for hit in v['accepted'] if hit['associated']]
            agg['methods'][name]=dict(primary_correct=sum(h['exact'] for h in top),
                primary_correct_and_on_roof=sum(h['exact'] and h['associated'] for h in top),
                primary_wrong=sum(h['block_y'] is not None and not h['exact'] for h in top),
                primary_missing=sum(h['block_y'] is None for h in top),
                matched_parts=len(matched),correct_matched_parts=sum(h['exact'] for h in matched),
                anchored=sum(v['anchored'] for v in m))
        aggregates[label]=agg
    out.parent.mkdir(exist_ok=True,parents=True)
    save(out,dict(run_hashes=fingerprints,roof_probes_sha256=digest(framed.openings.PROBES),groups=aggregates,rows=data))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',nargs=2,metavar=('NAME','DIRECTORY'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report([(n,Path(p)) for n,p in args.run],args.out)
