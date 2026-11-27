"""Bypass only the texture-guided counter retry; let the downstream stages consume that result."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from benchmark import reference_surface_feedback as replay, typed_structures as typed
from engine import surface_model

OUT = Path(__file__).resolve().parent/'reviews/material-feedback-surfaces-01'


def without_retry(image, parts, atlas):
    with TemporaryDirectory() as stage:
        return surface_model._count_regions(image,parts,'26.3','masked-repeat',atlas,Path(stage))


def case(job):
    entries = {}
    with TemporaryDirectory() as temp, patch.object(replay.viewer,'recount',without_retry):
        for mode in ('baseline','chart'):
            folder=Path(temp)
            key,_,_=replay.evaluate((mode,job,folder))
            row=json.loads((folder/(key+'--'+mode+'.json')).read_bytes())
            entries[mode]={k:row[k] for k in ('geometry','scores','natural_span_exact','pack_sha256')}
    return dict(id=job[1]['id'],split=job[0],without_retry=entries)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    controls={r['id']:r for r in typed.read(replay.ROOT/'cases.json')}
    jobs=[j for j in replay.all_jobs().values() if j[1]['id'] in controls]
    assert len(jobs)==20
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(case,jobs))
    totals={}
    for group,members in [('all',rows),('fixed',[r for r in rows if r['split'] in typed.SUITE.framed.EVALUATION])]:
        totals[group]={}
        for mode in ('baseline','chart'):
            result={}
            for arm in ('with_retry','without_retry'):
                subset=[controls[r['id']]['modes'][mode] if arm=='with_retry' else r[arm][mode] for r in members]
                result[arm] = dict(geometry=typed.pooled_scores([e['geometry'] for e in subset]),
                                   typed=typed.pooled_scores([e['scores'] for e in subset]),
                                   exact_on_true_surfaces=sum(e['natural_span_exact'] for e in subset) if mode=='chart' else None)
            result['packs_changed']=sum(r['without_retry'][mode]['pack_sha256']!=controls[r['id']]['modes'][mode]['pack_sha256'] for r in members)
            totals[group][mode]=result
    typed.compact(OUT/'retry-bypass.json',dict(summary=totals,cases=rows))


if __name__=='__main__':
    main()
