"""Regrade all frozen surface combinations and run the ordinary Block Counter on each."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
from uuid import uuid4

from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.normal_round import groups
from benchmark.split_count_check import count_frame
from benchmark.split_round import score_regions, pooled
from benchmark.surface_reference import ARCHIVE
from runtime.mode import image_only, mark_image_only


REVIEWS = ROOT/'benchmark/reviews'
PRIMARY = 'recovery-01'
JUNCTION = 'junction-underside-01'
MOBILE = {'mobile-wall':'mobile-walls-01','combined':'mobile-combined-01'}
METHODS = ('current','wall-and-underside','junction-cuts','mobile-wall','junction-then-underside','combined')


def verified(path, sha):
    if digest(path)!=sha:
        raise ValueError(f'Changed archived input {path}')
    return path


def sources(fixtures):
    manifests, frames = {},{}
    def load(directory):
        path = REVIEWS/directory/'manifest.json'
        manifest = read(path)
        manifests[path.relative_to(REVIEWS).as_posix()] = digest(path)
        if manifest.get('fixtures',manifest.get('fixture'))!=fixtures:
            raise ValueError(f'Fixture versions differ in {directory}')
        return manifest, read(verified(path.parent/'cases.json',manifest['artifacts']['cases.json']))
    for directory in (PRIMARY,JUNCTION):
        manifest, rows = load(directory)
        if manifest['reference_sha256']!=digest(ARCHIVE):
            raise ValueError(f'Reference geometry differs in {directory}')
        for row in rows:
            key = row['key']
            if directory==PRIMARY:
                frames[key]=dict(key=key,sha256=row['sha256'],layout=row['layout'],cohort=row['cohort'],
                                 resources={},reported={m:row['scores'][m]['matched'] for m in row['scores']})
            item=frames[key]
            if row['sha256']!=item['sha256']:
                raise ValueError(f'Wrong screenshot for {key}')
            if directory==JUNCTION:
                item['reported']['junction-then-underside']=row['methods']['junction-then-underside']['totals']['matched']
            relative=row['predictions']
            item['resources'][directory]=(directory+'/'+relative,manifest['artifacts'][relative])
    for method,directory in MOBILE.items():
        for cohort in ('development','regression'):
            _, rows = load(directory+'/'+cohort)
            for row in rows:
                item=frames[row['key']]
                if row['sha256']!=item['sha256'] or row['cohort']!=item['cohort']:
                    raise ValueError(f"Wrong MobileSAM archive row for {row['key']}")
                item['resources'][method]=(directory+'/'+row['predictions'],row['prediction_sha256'])
                item['reported'][method]=row['split']['matched']
    return frames,manifests


def grade(job):
    case,frame=job
    verified(case.path('image'),frame['sha256'])
    predictions={}
    for source,(relative,sha) in frame['resources'].items():
        with gzip.open(verified(REVIEWS/relative,sha),'rt',encoding='utf8') as f:
            archive=json.load(f)
        if source in (PRIMARY,JUNCTION):
            predictions.update(archive)
        else:
            predictions[source]=archive
    predictions={m:predictions[m] for m in METHODS}
    scores,scopes=score_regions(case,predictions,frame['key'])
    if {m:row['matched'] for m,row in scores.items()}!=frame['reported']:
        raise ValueError(f"Independent grades differ on {frame['key']}")
    frame.update(scores=scores,scopes=scopes)
    with Image.open(case.path('image')) as im:
        counts,_ = count_frame(case,im.convert('RGB'),frame,predictions,METHODS)
    frame['counts']={m:{k:counts[m]['totals'][k] for k in ('correct','incorrect','missing','unmatched_returned')}
                     for m in METHODS}
    frame['sizes_by_surface']={m:c['references'] for m,c in counts.items()}
    return frame


@image_only()
def run(out, workers):
    cohorts,suite,fixtures=groups()
    cases={f'{c.floor}/{c.id}':c for group in cohorts.values() for c in group}
    frames,manifests=sources(fixtures)
    if len(cases)!=56 or frames.keys()!=cases.keys():
        raise ValueError('The frozen experiment requires the same 56 screenshots')
    out.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        rows=list(pool.map(grade,[(cases[key],row) for key,row in frames.items()]))
    scores={}
    for group in ('all','development','previously_reviewed_regression',*sorted({r['layout'] for r in rows})):
        members=[r for r in rows if group in ('all',r['cohort'],r['layout'])]
        scores[group]={}
        for method in METHODS:
            result=pooled(members,method)
            counts=Counter()
            for row in members:
                counts.update(row['counts'][method])
            scores[group][method]=dict(**result,counts=dict(counts))
    save(out/'cases.json',rows)
    save(out/'scores.json',scores)
    save(out/'manifest.json',dict(run_id=out.name,code=revision(),version=suite['version'],fixtures=fixtures,
        sources=manifests,reference_sha256=digest(ARCHIVE),methods=METHODS,counter='masked-repeat',
        scope='Recorded surface matches and visible dimensions. In-game placement was not tested.',
        frames=[{k:r[k] for k in ('key','sha256','cohort','layout')} for r in rows],
        settings='Frozen settings from each source manifest. No inference or tuning during this replay.',
        artifacts={f:digest(out/f) for f in ('cases.json','scores.json')},
        reproduce=r'.\Invoke-SurfaceRound.ps1 -Round recovery-summary'))
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'recovery-summary-{uuid4().hex[:12]}')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    run(args.out,args.workers)
