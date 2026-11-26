"""Choose roof strip texture thresholds on the 72 development photos."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
from zipfile import ZipFile

from benchmark import framed_depth_eval as ev
from benchmark.mask_blocks import registered
from benchmark.relative_geometry import best_translation
from engine import roof_span_texture as method


def example(job):
    split, frame, base, out=job
    image,parts,counts,camera=ev.photo(split,frame,base)
    with ZipFile(ev.REVIEW/'framed-depth-01/scores.zip') as z:
        cells=set(map(tuple,json.loads(z.read(f"{split}/{frame['id']}/cells.json"))['local']))
    ctx,rows=method.observations(image,parts,camera,cells,ev.STABLE/'runs/26.3/atlas')
    modes=('loose','strict','guarded','conservative')
    candidates={mode:method.proposed(ctx,rows,cells,mode)-cells for mode in modes}
    truth,occupied,transform,_=ev.references(split,frame,None,ev.TRAINING_DATA,ev.DEPTH,{})
    shift,_=best_translation(registered(cells,transform),truth)
    def score(values):
        world={tuple(a+b for a,b in zip(c,shift)) for c in registered(values,transform)}
        return dict(correct=len(world & truth),extras=len(world-truth),hidden=len((world-truth)&occupied))
    record=dict(id=frame['id'],family=frame['family'],candidates=len(rows),
                measured_cell_pool=score({c for c,axis in rows}),before_visibility=score(method.continuous(rows,cells,'guarded')),
                **{k:score(v) for k,v in candidates.items()})
    ev.save(out/(frame['id']+'.json'),record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--limit',type=int,default=72)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    jobs=[(*j,a.out) for j in ev.cases(['train'])][:a.limit]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        result=list(pool.map(example,jobs))
    ev.save(a.out/'results.json',result)
    for k in ('loose','strict','guarded','conservative'):
        print(k,{m:sum(r[k][m] for r in result) for m in ('correct','extras','hidden')})
