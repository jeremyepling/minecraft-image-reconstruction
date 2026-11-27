"""Evaluator-only measurements of independent sheet pitch, phase, IDs and integer offsets."""
from argparse import ArgumentParser
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np
from engine.geometry import project, UNIT


def sheet_score(chart, front, faces, blocks):
    H=np.asarray(chart['H'])
    cells=chart['cells']
    all_deltas=[Counter() for _ in range(4)]
    axes=[np.array(d) for d in ((1,0,0),(-1,0,0),(0,0,1),(0,0,-1))]
    v_axis=np.array([0,1,0])
    near_corners=Counter()
    hit=same_id=same_face=0
    for c in cells:
        uv=np.array([c['u'],c['v']])
        quad=project(H,UNIT+uv)
        tests=np.array(list(product(np.linspace(.05,.95,9),repeat=2)))
        pts=np.floor(project(H,tests+uv)).astype(int)
        if np.any(pts<0) or np.any(pts>=front.shape[::-1]):
            raise ValueError('Off-image chart cell')
        labels=front[pts[:,1],pts[:,0]]
        f=faces.get(int(labels[40]))
        if f is None:
            continue
        hit+=1
        same_face+=np.mean(labels==f['id'])>=.85
        same_id+=blocks[f['block']]['block_id'].removeprefix('minecraft:')==c['block_id'].removeprefix('minecraft:')
        truth=np.array(f['corners_px'])
        edge=np.linalg.norm(truth-np.roll(truth,1,axis=0),axis=1).mean()
        error=min(np.sqrt(np.mean(np.sum((quad-np.roll(q,k,axis=0))**2,axis=1)))/edge
                  for q in (truth,truth[::-1]) for k in range(4))
        for tol in (.1,.25):
            near_corners[str(tol)]+=error<=tol
        normal=np.array(f['normal_world'])
        for axis, votes in zip(axes,all_deltas):
            if abs(normal@axis)>0 or abs(normal@v_axis)>0:
                continue
            offset=np.array(f['xyz_world']) - c['u']*axis - c['v']*v_axis
            votes[tuple(offset)]+=1
    consistent=max((max(v.values(),default=0) for v in all_deltas),default=0)
    return dict(cells=len(cells),center_on_structure=hit,exact_center_material=int(same_id),
                same_visible_face_85pct=int(same_face),consistent_integer_offsets=int(consistent),
                corners_within_10pct_edge=int(near_corners['0.1']),corners_within_25pct_edge=int(near_corners['0.25']))


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('run',type=Path)
    args=p.parse_args()
    cases=[]
    sums={}
    for case in json.loads(args.manifest.read_bytes())['cases']:
        frame=Path(case['frame'])
        labels=json.loads((frame/'labels.json').read_bytes())
        blocks={b['id']:b for b in labels['blocks']}
        faces={f['id']:f for f in labels['faces']}
        with np.load(frame/'labels.npz') as data:
            front=data['face_id']
        for path in sorted(args.run.glob('*/'+case['id']+'/*charts.json')):
            variant=path.parents[1].name
            charts=json.loads(path.read_bytes())['charts']
            results=[dict(id=c['id'],**sheet_score(c,front,faces,blocks)) for c in charts]
            cases.append(dict(id=case['id'],variant=variant,mode=path.stem,charts=results))
            group=sums.setdefault(variant,{}).setdefault(path.stem,Counter())
            group['nonempty_frames']+=int(bool(charts))
            group['charts']+=len(charts)
            for result in results:
                group.update({k:v for k,v in result.items() if k!='id'})
    path=args.run/'chart-metrics.json'
    path.write_text(json.dumps(dict(summary=sums,cases=cases),indent=2))
    print(json.dumps(sums,indent=2))


if __name__=='__main__':
    main()
