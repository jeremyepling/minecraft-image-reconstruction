"""Compare two predeclared patch/texture confidence settings on development photographs."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse

from benchmark import framed_depth_eval as ev
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from engine import roof_height_atlas as method


def one(job):
    split,f,base,runs,mode,out=job
    image,parts,counts,camera=ev.photo(split,f,base)
    key=f['id']
    pack=runs/split/key/'reconstruction.zip'
    prior=set(decoded_pack(pack))
    after,status=method.recover(image,parts,counts,camera,prior,mode=mode,atlas_path=ev.STABLE/'runs/26.3/atlas')
    truth,occupied,T,_=ev.references(split,f,None,ev.TRAINING_DATA,ev.DEPTH,{})
    m=placement_scores(registered(after,T),registered(prior,T),dict.fromkeys(truth),occupied)['relative_geometry']
    row=dict(id=key,family=f['family'],mode=mode,relative_geometry=m,status=status)
    ev.save(out/(key+'.json'),row)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=Path('../roof-span-round/runs/production'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=('strict','loose'),required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    jobs=[(*row,a.runs,a.mode,a.out) for row in ev.cases(('train',))]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(one,jobs))
    ev.save(a.out/'all.json',rows)
    print({k:sum(r['relative_geometry'][k] for r in rows)
           for k in ('gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')})
