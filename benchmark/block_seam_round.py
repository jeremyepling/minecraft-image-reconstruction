"""Compare a frozen seam decision with ordinary masked repetition on identical surfaces."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
import json
from pathlib import Path

from PIL import Image

from benchmark import grid_placement_check
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.normal_round import groups
from benchmark.verify_default_strip import SIGNATURES, region_signature
from benchmark.wall_join_test import checked, read_gzip
from engine.masked_count import count
from engine.strip_neighbor import clean_surfaces
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from runtime.mode import image_only, mark_image_only


POOL = ROOT/'benchmark/reviews/grid-chooser-01/pool'
MODELS = dict(boundaries='boundary_block_seam', learned='learned_block_seam', template='template_block_seam')
PROMOTION = ('Require more aligned visible faces on regression and fresh-family views, with no increase '
             'in wrong sizes, complete footprints failing alignment, or extra exported coordinates, '
             'and no decrease in correct coordinates. Compare each frozen candidate directly to the default.')


def decide(job):
    capture, row, source_sha, signature, frozen, model = job
    predict = import_module('benchmark.'+MODELS[model]).predict
    with Image.open(checked(capture.path('image'),row['sha256'])) as src:
        image = src.convert('RGB')
    upstream = read_gzip(checked(BASE/row['predictions'],source_sha))['gap']
    parts,_ = clean_surfaces(image,upstream)
    if region_signature(parts)!=signature:
        raise ValueError('Outline mismatch on '+row['key'])
    results = []
    for part, prior in zip(parts,frozen['surfaces'],strict=True):
        sample = sample_surface(image,part)
        quad = sample['quad'] if sample else None
        ordinary = count(sample)
        if (part['id']!=prior['id'] or quad!=prior['quad'] or
                any(ordinary[k]!=prior['baseline'][k] for k in ('counts','phase'))):
            raise ValueError(f'Control mismatch on {row["key"]} {part["id"]}')
        choice = predict(sample,ordinary)
        candidate = {k:list(choice[k]) for k in ('counts','phase')}
        candidate['blocks'] = dimensions(part,quad,image.size,candidate['counts'])['blocks']
        results.append(dict(id=part['id'],quad=quad,baseline=prior['baseline'],candidates=[candidate]))
    return dict(key=row['key'],sha256=row['sha256'],cohort=frozen['cohort'],layout=row['layout'],surfaces=results)


def dimensions_score(frames,targets):
    stats = Counter(eligible=0,correct=0,incorrect=0,missing=0,unmatched_returned=0,
                    gained_exact=0,lost_exact=0,new_exact_among_295=0)
    for frame in frames:
        target = targets[frame['key']]
        stats['eligible'] += target['eligible']
        for part,answer in zip(frame['surfaces'],target['surfaces'],strict=True):
            if part['id']!=answer['id']:
                raise ValueError('Scoring proposal identity changed')
            guess = part['candidates'][0]['blocks']
            prev = part['baseline']['blocks']
            expected = answer['expected']
            if expected is None:
                stats['unmatched_returned'] += int(None not in guess)
                continue
            stats['correct'] += int(guess==expected)
            stats['incorrect'] += int(None not in guess and guess!=expected)
            stats['gained_exact'] += int(guess==expected and prev!=expected)
            stats['lost_exact'] += int(prev==expected and guess!=expected)
            stats['new_exact_among_295'] += int(answer['empty_pool_chart'] and guess==expected)
    stats['missing'] = stats['eligible']-stats['correct']-stats['incorrect']
    return dict(stats)


def compact(path,data):
    path.write_text(json.dumps(data,separators=(',',':'),allow_nan=False),encoding='utf8')


@image_only()
def run(method,out,workers):
    source = read(checked(BASE/'manifest.json',BASE_SHA))
    original = read(POOL/'manifest.json')
    prior = read(checked(POOL/'candidates.json',original['files']['candidates.json']))
    targets = read(checked(POOL/'targets.json',original['files']['targets.json']))
    targets = {f['key']:f for f in targets}
    for frame in prior:
        target = targets[frame['key']]
        for row,answer in zip(frame['surfaces'],target['surfaces'],strict=True):
            if row['id']!=answer['id']:
                raise ValueError('Original proposal identity changed')
            answer['empty_pool_chart'] = row['quad'] is not None and not row['candidates']
    cohorts,_,fixtures = groups()
    captures = {c.floor+'/'+c.id:c for group in cohorts.values() for c in group}
    rows = read(checked(BASE/'cases.json',source['artifacts']['cases.json']))
    sigs = read(checked(SIGNATURES,original['signatures_sha256']))['frames']
    if source['frames']!=original['frames'] or fixtures!=original['fixtures'] or len(rows)!=56:
        raise ValueError('Expected identical fixed suite')
    jobs = []
    for r,p in zip(rows,prior,strict=True):
        if r['key']!=p['key'] or r['sha256']!=p['sha256'] or targets[r['key']]['sha256']!=p['sha256']:
            raise ValueError('Unexpected frame')
        jobs.append((captures[r['key']],r,source['artifacts'][r['predictions']],sigs[r['key']],p,method))
    out.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        frames = list(pool.map(decide,jobs))
    choices = [dict(key=f['key'],sha256=f['sha256'],surfaces=[dict(id=p['id'],index=0) for p in f['surfaces']])
               for f in frames]
    compact(out/'readings.json',frames)
    compact(out/'choices.json',choices)
    grid_placement_check.run(out/'readings.json',out/'choices.json',out/'placement',workers)
    table = {s:dimensions_score([f for f in frames if s in ('all',f['cohort'],f['layout'])],targets)
             for s in ['all','development','regression',*sorted({f['layout'] for f in frames})]}
    save(out/'dimensions.json',table)
    save(out/'manifest.json',dict(method=method,code=revision(),model_sha256=digest(ROOT/'benchmark'/(MODELS[method]+'.py')),
         promotion=PROMOTION,frame_manifest_sha256=digest(POOL/'manifest.json'),fixtures=fixtures,
         reference_sha256=original['reference_sha256'],approved_outlines_sha256=digest(SIGNATURES),
         artifacts={name:digest(out/name) for name in ('readings.json','choices.json','dimensions.json','placement/manifest.json')}))


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--method',choices=MODELS,required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=4)
    args = cli.parse_args()
    run(args.method,args.out.resolve(),args.workers)
