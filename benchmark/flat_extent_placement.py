"""Recount and place the flat-color extent proposals on each affected source image."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from benchmark import facade_inset_round as suite, pavement_opening_eval as replay
from benchmark.edge_vote_outline_eval import baseline_pack
from benchmark.shopfront_outline_gaps import REVIEW
from benchmark.texture_camera_eval import read, save
from engine import flat_surface_extent
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs
from engine.pillar_seams import orientation
from engine.projected_appearance import Scene, STEP


def supplement(image,parts,camera,before,candidate,regions):
    # Preserve the existing coordinate frame. A new block must visibly occupy
    # the newly supported patch even with every original block still present.
    if candidate<=before or camera is None:
        return before
    merged=before|candidate
    scene=Scene(image,parts,camera,merged)
    eligible=np.zeros((3,*scene.shape),bool)
    _,rotation=orientation(camera,image.size)
    normals={p['id']:p.get('normal_camera') for p in parts}
    for region in regions:
        normal=normals[region['surface']]
        if normal is None:
            continue
        axis=int(np.argmax(np.abs(rotation.T@normal)))
        for y,a,b in region['runs']:
            if y%STEP==0:
                eligible[axis,y//STEP,(a+STEP-1)//STEP:(b+STEP-1)//STEP]=True
    eligible=eligible.reshape(3,-1)
    accepted=set()
    for i,idx in scene.owned.items():
        cell=scene.cells[i]
        if cell in before:
            continue
        supported=np.count_nonzero(eligible[scene.axes[idx],idx])
        if supported>=16 and supported>=.6*len(idx):
            accepted.add(cell)
    return before|accepted


def grade(task):
    job, events, veto, patch_veto = task
    split,frame,_,_=job
    image,parts,counts,camera=suite.ordinary_inputs(*job)
    proposals=flat_surface_extent.proposals(image,parts,exterior_veto=veto,patch_veto=patch_veto)
    assert [(p['surface'],p['box']) for p in proposals]==[(e['surface'],e['box']) for e in events],frame['id']
    edited=[]
    for part in parts:
        group=[p for p in proposals if p['surface']==part['id']]
        if not group:
            edited.append(part)
            continue
        mask=native_mask(part,image.size).copy()
        for p in group:
            for y,a,b in p['runs']:
                mask[y,a:b]=True
        updated={k:v for k,v in part.items() if k not in ('outline','source_crop','context_crop','sampling_chart')}
        edited.append(dict(updated,support_area=int(mask.sum()),support=dict(runs=row_runs(mask),size=list(image.size))))
    assert 'colonnade-01' not in frame['id']
    pack,expected=baseline_pack(job)
    before,old,old_dims=replay.build(image,parts,counts,camera)
    assert before==expected,frame['id']
    candidate,after,new_dims=replay.build(image,edited,counts,camera)
    from engine.pavement_opening import detect
    from engine.pavement_obstructions import window_obstructions
    for blocks,pieces in [(before,old),(candidate,after)]:
        rejected=set(detect(image,pieces,camera))
        if rejected:
            blocks-=window_obstructions(image.size,camera,[p for p in pieces if p['id'] in rejected],blocks)
    assert before==expected
    record=replay.grade_changed(job,before,candidate,old,after,{},set())
    local=supplement(image,edited,camera,before,candidate,proposals)
    local_record=replay.grade_changed(job,before,local,old,after,{},set()) if local!=candidate else record
    return dict(id=frame['id'],split=split,baseline_pack_sha256=pack,baseline_reproduced=True,
                local={k:local_record[k] for k in ('strict','facade','delta')},
                dimensions={k:[old_dims.get(k),new_dims.get(k)] for k in old_dims.keys()|new_dims.keys()
                            if old_dims.get(k)!=new_dims.get(k)},**record)


def main():
    from argparse import ArgumentParser
    p=ArgumentParser()
    from pathlib import Path
    p.add_argument('--out',type=Path,default=REVIEW)
    p.add_argument('--exterior-veto',action='store_true')
    p.add_argument('--patch-veto',action='store_true')
    args=p.parse_args()
    rows=read(args.out/'cases.json')
    changes={r['id']:[e for e in r['events'] if 'both_axes' not in e] for r in rows}
    tasks=[(j,changes[j[1]['id']],args.exterior_veto,args.patch_veto) for j in suite.jobs([*suite.framed.SIZES,'shelter']) if changes.get(j[1]['id'])]
    with ProcessPoolExecutor(5) as pool:
        results=list(pool.map(grade,tasks))
    save(args.out/'flat-placement.json',results)


if __name__=='__main__':
    main()
