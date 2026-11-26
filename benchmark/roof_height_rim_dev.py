"""Develop geometry-only horizontal plane selection on the 72 training photos."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark import framed_depth_eval as ev
from benchmark.mask_blocks import decoded_pack,registered
from benchmark.relative_geometry import best_translation
from engine import roof_height_rim as model


def example(job):
    split,f,prior,base,out,exchange,expected=job
    image,parts,counts,camera=ev.photo(split,f,prior)
    before=set(decoded_pack(ev.checked(base/split/f['id']/'reconstruction.zip',expected)))
    if exchange=='band':
        from engine import roof_height_rim_band as candidate
    elif exchange:
        from engine import roof_height_rim_exchange as candidate
    else:
        candidate=model
    after,status=candidate.recover(image,parts,counts,camera,before)
    truth,occupied,transform,_=ev.references(split,f,None,ev.TRAINING_DATA,ev.DEPTH,{})
    shift,_=best_translation(registered(before,transform),truth)
    def moved(cells):
        return {tuple(a+b for a,b in zip(c,shift)) for c in registered(cells,transform)}
    new=moved(after-before)
    row=dict(id=f['id'],family=f['family'],**status,correct=len(new&truth),extra=len(new-truth),
             hidden=len((new-truth)&occupied),lost=len(moved(before-after)&truth),removed_extra=len(moved(before-after)-truth))
    ev.save(out/(f['id']+'.json'),row)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('../roof-span-round/runs/conservative'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--exchange',action='store_true')
    p.add_argument('--band',action='store_true')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    compare=ev.read(a.base/'summary.json')
    previous=ev.read(ev.checked(a.base/'cases.json',compare['cases_sha256']))
    index={r['id']:r['pack_sha256'] for r in previous if r['split']=='train'}
    jobs=[(*j,a.base,a.out,'band' if a.band else a.exchange,index[j[1]['id']]) for j in ev.cases(['train'])]
    assert len(jobs)==len(index)==72
    with ProcessPoolExecutor(a.workers) as pool:
        rows=list(pool.map(example,jobs))
    ev.save(a.out/'cases.json',rows)
    print({k:sum(r[k] for r in rows) for k in ('correct','extra','hidden')})
