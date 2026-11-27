"""Verify the production texture-seam rule, its fallback, and the four ordinary exports."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from benchmark import facade_inset_round as suite
from benchmark.edge_vote_outline_eval import baseline_pack
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import digest, read, save
from engine import inset_mask_chooser, inset_returns, jagged_texture_join as rule
from engine.pillar_seams import native_mask

OUT=suite.ROOT/'benchmark/reviews/shopfront-seam-continuity-01'


def verify(task):
    job,expected=task
    s,f,_,_=job
    key=f['id']
    image,base,_,camera=suite.ordinary_inputs(*job)
    parts,returns,phase,outlines,_=inset_returns.detect(image,base,'26.3',camera=camera)
    pre_calibrated=inset_returns.possible(image.size,base) or inset_mask_chooser.possible(image.size,base)
    clock=perf_counter()
    with patch('engine.visible_placement.seam_camera',return_value=camera) as estimate:
        if returns or phase or outlines:
            joins=[]
        else:
            changed,joins,used=rule.refine(image,parts,'26.3',camera=camera if pre_calibrated else None)
            if pre_calibrated:
                assert used is camera and estimate.call_count==0
            assert estimate.call_count<=1
            for event in joins:
                union=next(p for p in changed if p['id']==event['output'])
                targets=[p for p in parts if p['id'] in event['inputs']]
                assert np.array_equal(native_mask(union,image.size),
                                      np.logical_or.reduce([native_mask(p,image.size) for p in targets]))
    elapsed=perf_counter()-clock
    protected_insets=bool(returns or phase or outlines)
    assert [e['inputs'] for e in joins]==([] if protected_insets else expected),key
    if joins:
        for invalid in [dict(P=None),dict(P=np.zeros((3,4)).tolist())]:
            with patch('engine.visible_placement.seam_camera',side_effect=AssertionError('second camera fit')):
                untouched,events,same_camera=rule.refine(image,parts,'26.3',camera=invalid)
            assert untouched is parts and events==[] and same_camera is invalid
    return dict(id=key,joins=joins,protected_insets=protected_insets,pre_calibrated=pre_calibrated,early_camera=estimate.call_count,
                elapsed_seconds=round(elapsed,6))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli',action='store_true')
    args=parser.parse_args()
    manifest=read(OUT/'phase-only.json')
    index={r['id']:[p['inputs'] for p in r['pairs'] if p['accepted']] for r in manifest['rows']}
    jobs=list(suite.jobs([*suite.framed.SIZES,'shelter']))
    if args.cli:
        scored={r['id']:r for r in read(OUT/'phase-placement.json')['rows']}
        directories={'train-arcbridge1-cool-high-right':'arcbridge',
                     'validation-cloister2-earth-low-left':'cloister',
                     'legacy-light_gray_concrete-holdout-portico-masonry-high-right':'portico',
                     'rim-views-26.3--positive--heldout-facade-oblique-left':'facade'}
        assert directories.keys()==scored.keys()
        jobs={j[1]['id']:j for j in jobs}
        checks=[]
        for key,name in directories.items():
            row=scored[key]
            dest=suite.ROOT/'runs'/('repeat-cli-'+name)
            build=read(dest/'build-manifest.json')
            assert build['image_sha256']==jobs[key][1]['files']['rgb.png']
            _,frozen=baseline_pack(jobs[key])
            zipfile=dest/'reconstruction.zip'
            decoded=decoded_pack(zipfile)
            assert set(decoded)==frozen and set(decoded.values())=={'minecraft:smooth_stone'}
            counts={r['id']:r['dimensions']['blocks'] for r in read(dest/'counts.json')['surfaces']}
            settings=build['surface_finder']['boundary_settings']
            protected=any(settings['visible_insets'].values())
            join=settings.get('jagged_texture_join',{}).get('events',[])
            if protected:
                assert not join and all(k not in counts for k in row['joined_dimensions'])
            else:
                assert all(counts[k]==v for k,v in row['joined_dimensions'].items())
                assert [e['inputs'] for e in join]==index[key]
            checks.append(dict(id=key,image_sha256=build['image_sha256'],export_sha256=digest(zipfile),
                               blocks=len(decoded),dimensions={} if protected else row['joined_dimensions'],
                               protected_insets=protected,events=join))
        save(OUT/'ordinary.json',dict(checks=checks))
        return
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(verify,((j,index[j[1]['id']]) for j in jobs)))
    timing=np.array([r['elapsed_seconds'] for r in rows])
    data=dict(frames=len(rows),joins=sum(len(r['joins']) for r in rows),
              frames_requiring_earlier_camera=sum(r['early_camera'] for r in rows),
              seconds=dict(median=float(np.median(timing)),p95=float(np.quantile(timing,.95)),max=float(timing.max())),
              settings='four workers; excludes image/field loading and camera fitting; checks expected exact unions',
              cases=rows)
    save(OUT/'production.json',data)
    print({k:v for k,v in data.items() if k!='cases'})


if __name__=='__main__':
    main()
