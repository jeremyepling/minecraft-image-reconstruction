"""Generate local plane and grid alternatives, then measure a truth-assisted selection with the native finalizer."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import framed_depth_eval as framed
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import STABLE,checked,read,save,digest
from engine import wall_section_grid as grid

ATLAS=STABLE/'runs/26.3/atlas'
CONTROL=STABLE.parent/(STABLE.name+'-worktrees')/'facade-placement-round/runs/production'




def extract(job):
    split,f,source,row,out=job
    image,parts,counts,camera=framed.photo(split,f,source)
    pack=checked(CONTROL/split/f['id']/'reconstruction.zip',row['pack_sha256'])
    prior=set(decoded_pack(pack))
    ctx,raw=grid.candidates(image,parts,camera,prior,ATLAS)
    groups=grid.disjoint(raw)
    # Open reference geometry and the scored global translation only now.
    ref_split='old' if split in ('new','extra') else split
    builds=framed.NEW_INPUTS if split=='new' else framed.transfer.BUILDS
    truth,occupied,T,_=framed.references(ref_split,f,framed.NATIVE_INPUTS.get(split),framed.TRAINING_DATA,
                                         framed.DEPTH,dict(native=framed.native.BUILDS,old=builds,fresh=builds))
    def world(c):
        return tuple(np.floor((np.asarray(c)+.5)@T[:3,:3].T+T[:3,3]).astype(int)+row['translation'])
    covered={}
    picked=[]
    for group in groups:
        for opt in group:
            for c in opt['cells']:
                if c not in covered:
                    covered[c]=world(c) in truth
            opt['positive']=sum(covered[c] for c in opt['cells'])
        # Maximize new true cells with zero false placements, independently for disjoint sections.
        valid=[opt for opt in group if opt['positive']==len(opt['cells'])]
        if valid:
            picked.append(max(valid,key=lambda opt:opt['positive']))
    seeded={c for opt in picked for c in opt['cells']}
    after=grid.materialize(ctx,prior,picked) if picked else prior
    added=after-prior
    assert all(covered[c] for c in added)
    m=row['relative_geometry']
    evidence=dict(id=f['id'],split=split,family=row.get('family'),camera=ctx is not None,control_correct=m['correct'],
                  target=m['target'],placed=m['placed'],extras=m['extras'],hidden=m['hidden_extras'],
                  candidates=len(covered),positive=sum(covered.values()),groups=len(groups),choices=sum(map(len,groups)),
                  pure_chooser_without_joint_visibility=len(seeded),actual_oracle_unique_correct=len(added),
                  estimated_face_axes={str(axis):sum(opt['axis']==axis for opt in picked) for axis in (0,2)},
                  rgb_sha256=row['rgb_sha256'],input_hashes=row['input_hashes'],base_pack_sha256=row['pack_sha256'])
    dest=out/split
    dest.mkdir(exist_ok=True,parents=True)
    save(dest/(f['id']+'.json'),dict(summary=evidence,sections=groups))
    return evidence


def run(args):
    source=read(CONTROL/'summary.json')
    rows=read(checked(CONTROL/'cases.json',source['cases_sha256']))
    index={(r['split'],r['id']):r for r in rows}
    jobs=[(s,f,h,index[s,f['id']],args.out) for s,f,h in framed.cases(args.splits) if not args.ids or f['id'] in args.ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        result=list(pool.map(extract,jobs))
    groups={'all':result,'fixed':[r for r in result if r['split'] in framed.EVALUATION]}
    groups.update({s:[r for r in result if r['split']==s] for s in args.splits})
    stats={}
    for name,group in groups.items():
        if not group:
            continue
        totals={k:sum(r[k] for r in group) for k in ('target','placed','control_correct','extras','candidates','positive',
                          'groups','choices','pure_chooser_without_joint_visibility','actual_oracle_unique_correct')}
        gain=totals['actual_oracle_unique_correct']
        totals['f1']=2*(totals['control_correct']+gain)/(totals['target']+totals['placed']+gain)
        totals['before_f1']=2*totals['control_correct']/(totals['target']+totals['placed'])
        stats[name]=totals
    save(args.out/'oracle.json',dict(control='028819a',control_cases_sha256=source['cases_sha256'],groups=stats,frames=result))
    print({k:(v['actual_oracle_unique_correct'],round(v['f1'],6)) for k,v in stats.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',nargs='+',default=list(framed.SIZES))
    p.add_argument('--ids',nargs='+')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=5)
    run(p.parse_args())
