"""After freezing, locate where the texture-span pipeline loses measured canopy roofs."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import numpy as np

from benchmark import framed_depth_eval as ev
from benchmark.mask_blocks import registered
from benchmark.relative_canopy_spacing import score, shift
from engine import roof_span_texture as method
from engine.visible_placement import _frontmost_cells


def inspect(job):
    split,frame,base=job
    image,parts,counts,camera=ev.photo(split,frame,base)
    with ZipFile(ev.REVIEW/'framed-depth-01/scores.zip') as pack:
        cells=set(map(tuple,json.loads(pack.read(f"{split}/{frame['id']}/cells.json"))['local']))
    ctx, measured=method.observations(image,parts,camera,cells,ev.STABLE/'runs/26.3/atlas')
    nominated=method.strip_lines(cells)
    subsets=dict(prior=cells,nominated=set.union(set(),*nominated.values()),
                 rays={c for c,axis in method.visible_candidates(ctx,nominated)} if ctx else set(),
                 texture={c for c,axis in measured},
                 conservative=method.proposed(ctx,measured,cells,'conservative')-cells)
    original=method.continuous
    protect=method._keep_prior_visible
    def trace_runs(*args):
        result=original(*args)
        stage='joint' if 'runs' in subsets else 'runs'
        subsets[stage]=result
        return result
    def trace_guard(*args):
        subsets['silhouette']=args[-1]
        return protect(*args)
    with patch.object(method,'continuous',trace_runs),patch.object(method,'_keep_prior_visible',trace_guard):
        subsets['guarded']=method.proposed(ctx,measured,cells,'guarded')-cells
    for stage in ('joint','silhouette'):
        subsets.setdefault(stage,set())
    # Scoring-only registration and labels. The full source cannot nominate a position.
    actual='old' if split=='extra' else split
    truth,_,t,_=ev.references(actual,frame,ev.NATIVE_INPUTS[split],ev.TRAINING_DATA,ev.DEPTH,
                              dict(native=ev.native.BUILDS,old=ev.transfer.BUILDS,fresh=ev.transfer.BUILDS))
    spec=ev.roof_specs()[frame['id']]
    old_world=registered(cells,t)
    anchored=score(old_world,truth,spec)
    roof={p for p in truth if p[1]==spec['roof_y']}
    delta=anchored['offset']
    counts={k:len(shift(registered(s,t),delta)&roof) for k,s in subsets.items()} if delta is not None else {}
    blocked=[]
    removed=subsets['silhouette']-subsets['guarded']
    if removed and delta is not None:
        original=sorted(cells)
        old=_frontmost_cells(original,ctx.native,ctx.eye,ctx.shape)
        survivors=set(subsets['silhouette'])
        occulted={}
        while survivors:
            scene=sorted(cells|survivors)
            new,_,_=ctx.joint(survivors)
            seen={scene[n] for n in np.unique(new) if n>=0}
            rejected=set()
            for i in np.unique(old):
                if i<0 or original[i] in seen:
                    continue
                for n in np.unique(new[old==i]):
                    if n>=0 and scene[n] in survivors:
                        occulted.setdefault(scene[n],set()).add(original[i])
                        rejected.add(scene[n])
            if not rejected:
                break
            survivors-=rejected
        assert survivors==subsets['guarded']
        for c in sorted(removed):
            if not (shift(registered({c},t),delta)&roof):
                continue
            hidden=occulted.get(c,set())
            verified=shift(registered(hidden,t),delta)&truth
            blocked.append(dict(cell=list(c),prior_cells_obscured=sorted(hidden),prior_correct=len(verified)))
    return dict(id=frame['id'],split=split,visible_roof=len(roof),post_offset=delta,
                matching_post_courses=anchored['matching_post_courses'],confirmed_roof=counts,
                suppressed_true_roof=blocked)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=3)
    args=cli.parse_args()
    jobs=[r for r in ev.cases(['old','fresh','extra']) if r[1]['id'] in ev.roof_specs()]
    with ProcessPoolExecutor(args.workers) as pool:
        result=list(pool.map(inspect,jobs))
    ev.save(args.out,result)
