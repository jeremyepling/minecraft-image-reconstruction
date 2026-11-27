"""Score counts on clearly matched rectangular visible walls and validate accepted corner pairs."""
import argparse
from pathlib import Path

import numpy as np

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, read, save
from benchmark.relative_oracle_charts import ENDS, UNIT, owner_edge_samples, unit_projection


def wall_targets(case, baseline):
    frame = Path(case['frame'])
    faces = {x['id']:x for x in read(frame/'labels.json')['faces']}
    with np.load(frame/'labels.npz') as labels:
        owners = labels['face_id']
    targets = {}
    for t in baseline['targets']:
        if t['normal_axis'] in (2,-2):
            continue
        axis=abs(t['normal_axis'])-1
        axes=[i for i in range(3) if i!=axis]
        selected=[faces[i] for i in t['face_ids']]
        pos=np.array([f['xyz_world'] for f in selected],dtype=int)
        dx,dy,dz=np.ptp(pos,axis=0)+1
        w,h=int(max(dx,dz)),int(dy)
        edges={}
        for face in selected:
            pixels=unit_projection(face,axes)
            bits=owner_edge_samples(pixels,face['id'],owners)
            xyz=np.array(face['xyz_world'])
            xyz[axis]+=int(t['normal_axis']>0)
            corners=np.tile(xyz,(4,1))
            corners[:,axes]+=UNIT.astype(int)
            for (a,b),visible in zip(ENDS,bits):
                if visible:
                    key=tuple(sorted([tuple(corners[a]),tuple(corners[b])]))
                    edges[key]=edges.get(key,0)|visible
        targets[t['id']]=dict(width=w,height=h,rectangle=w*h==len(selected),edges=edges)
    return targets


def measure(arm, case, baseline, targets):
    name=case['id']
    root=ROOT/'runs/sky-crop-downstream'/arm
    parts={r['id']:r for r in read(root/'sky'/name/'source-handoff/counts.json')['surfaces']}
    predicted={r['id']:r for r in baseline['proposals']}
    counts=[]
    for p in predicted.values():
        t=targets.get(p['target'])
        if t and t['rectangle'] and p['iou']>=.5:
            estimate=parts.get(p['id'],{}).get('counts') or [None,None]
            for axis, (value,true) in enumerate(zip(estimate,(t['width'],t['height']))):
                counts.append(dict(id=p['id'],axis=axis,truth=true,value=value,
                                   status='missing' if value is None else
                                   ('correct' if round(value)==true else 'wrong')))
    seams=read(root/'shared-corner-rows/sky'/name/'seams.json')['seams']
    joins=[]
    for seam in seams:
        if seam['decision']!='accepted':
            continue
        a,b=[targets.get(predicted[seam[k]]['target']) for k in ('donor_id','receiver_id')]
        verified=bool(a and b and a is not b and any(bits & b['edges'].get(edge,0)
                                                      for edge,bits in a['edges'].items()))
        joins.append(dict(donor=seam['donor_id'],receiver=seam['receiver_id'],confirmed=verified))
    return dict(id=name,counts=counts,shared_row_transfers=joins)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('arms',nargs='+')
    cli.add_argument('--reference',type=Path,default=WORKTREES/'sky-crop-reference/runs/sky-crop-reference-01/report.json')
    cli.add_argument('--output',type=Path,default=ROOT/'runs/sky-crop-downstream/counts-and-joins.json.gz')
    args=cli.parse_args()
    rows={arm:[] for arm in args.arms}
    cases={c['id']:c for c in read(SOURCE/'floating-build-grid-01/fixtures.json')['cases']}
    for entry in read(args.reference)['cases']:
        case=cases[entry['id']]
        baseline=entry['arms']['baseline']
        targets=wall_targets(case,baseline)
        for arm in args.arms:
            rows[arm].append(measure(arm,case,baseline,targets))
    save(args.output,rows)
    for arm,frames in rows.items():
        counts=[c['status'] for f in frames for c in f['counts']]
        joins=[s for f in frames for s in f['shared_row_transfers']]
        print(arm,[counts.count(k) for k in ('correct','wrong','missing')],sum(j['confirmed'] for j in joins),len(joins))
