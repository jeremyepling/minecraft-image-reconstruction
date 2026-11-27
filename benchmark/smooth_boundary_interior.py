"""Grade native atlas-confirmed outline interiors after the shipped eave and seam refinements."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import facade_inset_round as suite
from benchmark.shared_face_error_budget import current_stages, LATEST
from benchmark.texture_camera_eval import read, save
from benchmark.wall_texture_evaluate import metrics
from engine import atlas_count, eave_pinholes, jagged_texture_join, smooth_atlas_outline as outline
from engine.pillar_seams import native_mask
from engine.smooth_extent_placement import proposals

OUT=suite.ROOT/'benchmark/reviews/smooth-boundary-atlas-01'


@lru_cache(maxsize=1)
def get_atlas():
    return atlas_count.load_atlas(suite.ATLAS,'26.3')


def score(task):
    job, expected = task
    split, frame, _, _ = job
    image,parts,_,camera = current_stages(job,expected)
    parts,_ = eave_pinholes.refine(image,parts)
    if not any(expected[k] for k in ('returns','texture','outlines')):
        parts,_,_ = jagged_texture_join.refine(image,parts,'26.3',camera=camera)
    started=perf_counter()
    raw=proposals(image,parts)
    found=outline.checked_regions(image,parts,raw,get_atlas(),radius=0) if raw else []
    by_id={p['id']:p for p in parts}
    masks={k:native_mask(by_id[k],image.size) for k in {r['surface'] for r in found}}
    tested={}
    for radius in (0,1,2):
        regions=[]
        for r in found:
            runs=outline.supported_runs(masks[r['surface']],r['runs'],r['box'],radius)
            if runs:
                regions.append(dict(r,runs=runs,pixels=sum(b-a for _,a,b in runs)))
        tested[str(radius)]=regions
    elapsed=perf_counter()-started
    record=dict(id=frame['id'], source_sha256=frame['files']['rgb.png'],split=split,
                cohort='fixed' if split in suite.framed.EVALUATION else 'transfer',
                raw=len(raw),accepted_before_trim=len(found),seconds=round(elapsed,4),radii={})
    chosen=outline.apply_regions(image,parts,tested['2'])
    assert all(a['support']==b['support'] for a,b in zip(parts,outline.measurement_parts(chosen)))
    if not found:
        return record
    truth=suite.visible_map(split,frame)
    record['label_sha256']=sha256(truth.astype('<i4').tobytes()).hexdigest()
    baseline=metrics(parts,truth,image.size)
    owners={r['id']:r['dominant'] for r in baseline['parts']}
    for radius,regions in tested.items():
        summary=Counter()
        events=[]
        for r in regions:
            samples=np.concatenate([truth[y,a:b] for y,a,b in r['runs']])
            counts=Counter(pixels=len(samples),host=int((samples==owners[r['surface']]).sum()),background=int((samples==0).sum()))
            counts['neighbor']=len(samples)-counts['host']-counts['background']
            if frame['id']=='legacy-light_gray_concrete-holdout-registry-plaster-high-right':
                counts['registry_right']=int((samples==10).sum())
            if frame['id']=='legacy-light_gray_concrete-dev-shopfront-plaster-low-left':
                counts['shopfront_center']=sum(max(0,min(b,850)-max(a,790)) for y,a,b in r['runs'] if 490<=y<545)
            summary.update(counts)
            events.append(dict(surface=r['surface'],box=r['box'],**counts))
        after=metrics(outline.apply_regions(image,parts,regions),truth,image.size)
        record['radii'][radius]=dict(summary,events=events,gained=sorted(set(after['matched'])-set(baseline['matched'])),
                 lost=sorted(set(baseline['matched'])-set(after['matched'])),
                 paired=after['paired_pixels']-baseline['paired_pixels'],unmatched=after['unmatched_predictions']-baseline['unmatched_predictions'])
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    exp={r['id']:r for r in read(LATEST)['rows']}
    jobs=[(j,exp[j[1]['id']]) for j in suite.jobs([*suite.framed.SIZES,'shelter'])]
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(score,jobs))
    results={}
    for group in ('fixed','transfer','shopfront','registry','portico','courtyard'):
        subset=[r for r in rows if r['cohort']==group or (r['cohort']=='fixed' and group in r['id'])]
        values={}
        for radius in ('0','1','2'):
            total=Counter(views=len(subset))
            for row in subset:
                r=row['radii'].get(radius,{})
                for key in ('pixels','host','neighbor','background','paired','unmatched','registry_right','shopfront_center'):
                    total[key]+=r.get(key,0)
                total['gained']+=len(r.get('gained',[]))
                total['lost']+=len(r.get('lost',[]))
                total['affected']+=bool(r.get('events',[]))
            values[radius]=dict(total)
        results[group]=values
    save(OUT/'interior.json',dict(rows=rows,results=results,
         seconds=dict(zip(('median','p95','max'),np.quantile([r['seconds'] for r in rows],[.5,.95,1]).tolist()))))


if __name__=='__main__':
    main()
