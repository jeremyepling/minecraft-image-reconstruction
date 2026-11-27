"""Frozen pilot, source proposal inventory followed by native answer-side grading."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.expanded_baseline import ROOT, WORKTREES, read, save, digest
from benchmark.harbor_carrier_audit import native_face_planes
from benchmark.return_face_proposals import propose
from engine.plane_consensus import row_runs

RUN=ROOT/'runs/returns'
CONTROL=WORKTREES/'surface-depth-round-controls/benchmark/reviews/surface-depth-round-controls-01/controls.json'
PC=WORKTREES.parent


def source(row):
    if row['dataset']=='fixed-104':
        e=next(e for e in fixed.fixed_frames() if e['id']==row['id'])
        inputs=fixed.inference_inputs(e)
        return inputs.image,inputs.parts,inputs.camera,None
    folder=PC/row['replay']['root']/'handoffs'
    parts=read(folder/'placement-input.json.gz')['parts']
    with np.load(folder/'planarity.npz') as cache:
        planarity=(cache['relaxed'],cache['relaxed_normals'])
    receipt_path=row['destination']['placement_camera_receipt']
    receipt=read(PC/receipt_path['path']) if receipt_path else read(folder/'placement-input.json.gz')
    camera=receipt['camera']
    assert camera['P']==row['destination']['P']
    return Image.open(PC/row['rgb']['path']).convert('RGB'),parts,camera,planarity


def inventory(row):
    image,parts,camera,planarity=source(row)
    key=row['id']
    status=(row['destination']['historical_placement'] if row['dataset']=='fixed-104' else
            read(PC/row['destination']['root']/'build-manifest.json')['placement'])
    assert camera['P']==status['P']
    audit,groups=propose(image,parts,np.asarray(camera['P']),planarity)
    levels=[p['level'] for p in status['plane_offsets'] if p['id']==audit['parent']]
    if not levels:
        raise ValueError(f'{key}: incumbent carrier missing')
    initial=levels[0]
    options=[o for o in audit['options'] if o['target_level']!=initial]
    # Source ranking across alternate boundary contacts. Keep at most one child per build.
    def rank(o):
        return (o['fraction']*min(o['extent_px'],200)*o['strength'],o['support'],-o['group'])
    chosen=max((o for o in options if o['accepted']),key=rank,default=None)
    forced=max((o for o in options if o['support']>=3),key=rank,default=None)
    out=RUN/key
    save(out/'source.json',dict(id=key,initial=initial,**audit,chosen=chosen,forced=forced))
    masks={int(o['group']) for o in options}
    save(out/'masks.json', {g:row_runs(groups==g) for g in masks})
    # This section grades the already frozen candidates against native frontmost pixel labels.
    if row['dataset']=='fixed-104':
        e=next(e for e in fixed.fixed_frames() if e['id']==key)
        labels,ids,T=fixed.fixed_reference(e)
    else:
        labels=read(PC/row['reference']['labels.json']['path'])
        with np.load(PC/row['reference']['labels.npz']['path']) as archive:
            ids=archive['face_id']
        T=np.asarray(row['baseline']['registration']['transform'])
    planes=native_face_planes(labels,np.asarray(T))
    lookup=np.array([(-1,0)]+[planes[k] for k in range(1,len(planes)+1)])
    axes,levels=lookup[ids].transpose(2,0,1)
    dense=np.repeat(np.repeat(groups,4,axis=0),4,axis=1)
    axes,levels=axes[:dense.shape[0],:dense.shape[1]],levels[:dense.shape[0],:dense.shape[1]]
    records={}
    for group in range(1,int(groups.max())+1):
        local=levels[(dense==group)&(axes==audit['axis'])]
        native,n=np.unique(local,return_counts=True)
        offered=sorted({initial,*[o['target_level'] for o in audit['options'] if o['group']==group]})
        records[group]=dict(histogram={int(k):int(v) for k,v in zip(native,n)},offered=offered)
    save(out/'grade.json',records)
    print(key,audit['parent'],len(audit['options']),chosen and (chosen['group'],chosen['target_level']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=['dev','transfer','all'],default='all')
    p.add_argument('--ids',nargs='+')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    rows=read(CONTROL)['pilot']['frames']
    if args.ids: rows=[r for r in rows if r['id'] in args.ids]
    elif args.group=='dev': rows=rows[:4]
    elif args.group=='transfer': rows=rows[4:]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(inventory,rows))

if __name__=='__main__':
    main()
