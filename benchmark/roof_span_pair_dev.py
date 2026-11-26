"""Choose a paired occultation threshold only on development buildings."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
from zipfile import ZipFile

from benchmark import framed_depth_eval as ev
from benchmark.mask_blocks import registered
from benchmark.relative_geometry import best_translation
from engine import roof_span_occlusion as pair, roof_span_texture as strips


def example(job):
    split,frame,source,out=job
    image,parts,counts,camera=ev.photo(split,frame,source)
    with ZipFile(ev.REVIEW/'framed-depth-01/scores.zip') as archive:
        cells=set(map(tuple,json.loads(archive.read(f"{split}/{frame['id']}/cells.json"))['local']))
    ctx,rows=strips.observations(image,parts,camera,cells,ev.STABLE/'runs/26.3/atlas')
    screened=strips.screened(ctx,rows,cells,'conservative')
    base=strips.proposed(ctx,rows,cells,'conservative')
    outputs={name:pair.supported(ctx,cells,screened,rows,quality,margin)
             for name,quality,margin in [('strict',.78,.18),('clear',.65,.12)]}
    truth,occupied,transform,_=ev.references(split,frame,None,ev.TRAINING_DATA,ev.DEPTH,{})
    delta,_=best_translation(registered(base,transform),truth)
    def compare(s):
        return {tuple(v+d for v,d in zip(c,delta)) for c in registered(s,transform)}
    metrics={}
    for name,result in outputs.items():
        gain,drop=compare(result-base),compare(base-result)
        metrics[name]=dict(gained=len(gain&truth),lost=len(drop&truth),extras_added=len(gain-truth),
                           extras_removed=len(drop-truth),hidden_added=len(gain&(occupied-truth)))
    screened_only=screened-(base-cells)
    world=compare(screened_only)
    rec=dict(id=frame['id'],family=frame['family'],pre_guard_correct=len(world&truth),
             pre_guard_extras=len(world-truth),**metrics)
    ev.save(out/(frame['id']+'.json'),rec)
    return rec


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        records=list(pool.map(example,[(*j,args.out) for j in ev.cases(['train'])]))
    ev.save(args.out/'results.json',records)
    for k in ('strict','clear'):
        print(k,{m:sum(r[k][m] for r in records) for m in records[0][k]})
