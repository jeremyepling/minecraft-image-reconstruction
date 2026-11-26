"""How often do the masked counter's own texture peaks admit the correct visible grid?"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path

from PIL import Image

from benchmark.camera_round import digest, read, revision, save
from benchmark.count_metrics import reference_planes
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.mask_metrics import measure
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.verify_default_strip import APPROVED, SIGNATURES, region_signature
from benchmark.wall_join_test import checked, read_gzip
from engine import masked_count
from engine.strip_neighbor import clean_surfaces
from engine.surface_dimensions import dimensions
from engine.surface_pitch import _phase
from engine.surface_sample import sample_surface
from runtime.mode import image_only, mark_image_only


def pitch_pairs(options, selected, aspect):
    if all(selected):
        yield selected, 'ordinary'
    for pair in product(*options):
        yield pair, 'two observed peaks'
    if aspect:
        for axis, peaks in enumerate(options):
            for peak in peaks:
                # Reuse the production evidence threshold for transferring one axis.
                if (peak['native_support'] <= .20 and
                        peak['agreement'] <= masked_count.SETTINGS['aspect_transfer_agreement']):
                    continue
                inferred = peak['count'] / aspect if axis == 0 else peak['count'] * aspect
                transfer = dict(count=inferred, source='measured axis and plane aspect')
                yield ([peak, transfer] if axis == 0 else [transfer, peak]), 'one peak and aspect'


def proposals(image, part):
    sample = sample_surface(image, part)
    reading = masked_count.count(sample)
    quad = None if sample is None else sample['quad']
    baseline = dict(counts=reading['counts'], phase=reading['phase'],
                    blocks=dimensions(part, quad, image.size, reading['counts'])['blocks'])
    result = dict(id=part['id'], quad=quad, baseline=baseline, baseline_index=None, candidates=[])
    if sample is None:
        return result
    evidence = reading['evidence']
    aspect = evidence['aspect']
    pairs = pitch_pairs(evidence['alternatives'], evidence['axes'], aspect)
    # Computing per-axis phase needs the same row sampling and validity mask as count().
    @lru_cache(maxsize=2)
    def scanlines(axis):
        _, colors, valid = masked_count._options(sample, axis)
        return colors, valid

    @lru_cache(maxsize=None)
    def phase(axis, n):
        if n == baseline['counts'][axis]:
            return baseline['phase'][axis]
        colors, valid = scanlines(axis)
        return None if colors is None else _phase(colors, valid, n)

    @lru_cache(maxsize=None)
    def blocks(axis, n):
        counts = [None, None]
        counts[axis] = n
        return dimensions(part, quad, image.size, counts)['blocks'][axis]

    seen = set()
    height, width = sample['mask'].shape
    fill = float(sample['mask'].mean())
    for axes, origin in pairs:
        counts = tuple(p['count'] for p in axes)
        if counts in seen:
            continue
        seen.add(counts)
        size = [blocks(i, n) for i, n in enumerate(counts)]
        if None in size:
            continue
        features = dict(aspect_known=int(bool(aspect)), fill=fill, chart_width=width, chart_height=height,
                        aspect_error=abs(math.log(counts[0]/counts[1]/aspect)) if aspect else 0.)
        for key, p in zip(('u', 'v'), axes):
            features['agreement_'+key] = p.get('agreement', 0.)
            features['native_'+key] = p.get('native_support', 0.)
            features['rows_'+key] = p.get('rows', 0)
            features['transfer_'+key] = int(p['source'] == 'measured axis and plane aspect')
        if origin == 'ordinary':
            result['baseline_index'] = len(result['candidates'])
        result['candidates'].append(dict(counts=counts, blocks=size, phase=[phase(i,n) for i,n in enumerate(counts)],
                                         origin=origin, features=features))
    return result


def frame_job(job):
    capture, row, prediction_sha, signature, approved = job
    with Image.open(checked(capture.path('image'), row['sha256'])) as src:
        image = src.convert('RGB')
    upstream = read_gzip(checked(BASE/row['predictions'], prediction_sha))['gap']
    parts, _ = clean_surfaces(image, upstream)
    if region_signature(parts) != signature:
        raise ValueError('Default outlines changed on '+row['key'])
    info = dict(key=row['key'], sha256=row['sha256'], layout=row['layout'],
                cohort='development' if row['cohort']=='development' else 'regression')
    predicted = dict(**info, surfaces=[proposals(image, part) for part in parts])
    _, ref = recorded(capture)
    sizes = reference_planes(ref, read(capture.path('labels'))['faces'])
    _, pairs = measure(parts, ref, use_mask=True)
    by_index = dict(pairs)
    if sorted(by_index.values()) != approved['scopes']['neighbor']['matched']:
        raise ValueError('Default pairings changed on '+row['key'])
    targets = dict(**info, eligible=sum(p['eligible'] for p in ref['patches'].values()), surfaces=[
        dict(id=p['id'], reference_id=by_index.get(i), expected=sizes[by_index[i]]['size'] if i in by_index else None)
        for i,p in enumerate(parts)])
    stats = score_frame(predicted, targets)
    for key, value in approved['sizes']['neighbor'].items():
        if stats[key] != value:
            raise ValueError(f'Baseline {key} changed on {row["key"]}')
    return predicted, targets, stats


def score_frame(predicted, targets):
    stats = Counter(frames=1, eligible=targets['eligible'], correct=0, incorrect=0, unmatched_returned=0,
                    matched=0, oracle_exact=0, no_candidates=0, wrong_candidates_only=0,
                    two_peak_or_default_exact=0, candidate_grids=0)
    for p,t in zip(predicted['surfaces'], targets['surfaces'], strict=True):
        if p['id'] != t['id']:
            raise ValueError('Proposal/target mismatch')
        candidates = p['candidates']
        stats['candidate_grids'] += len(candidates)
        truth = t['expected']
        complete = None not in p['baseline']['blocks']
        if truth is None:
            stats['unmatched_returned'] += int(complete)
            continue
        stats['matched'] += 1
        if complete:
            stats['correct' if p['baseline']['blocks']==truth else 'incorrect'] += 1
        hits = [c for c in candidates if c['blocks']==truth]
        stats['oracle_exact'] += int(bool(hits))
        stats['two_peak_or_default_exact'] += int(any(c['origin']!='one peak and aspect' for c in hits))
        stats['no_candidates'] += int(not candidates)
        stats['wrong_candidates_only'] += int(bool(candidates) and not hits)
    stats['missing'] = stats['eligible']-stats['correct']-stats['incorrect']
    stats['no_matched_outline'] = stats['eligible']-stats['matched']
    return dict(stats)


@image_only()
def run(out, workers):
    source = read(checked(BASE/'manifest.json', BASE_SHA))
    gold = read(APPROVED/'manifest.json')
    checked(ARCHIVE, gold['reference_sha256'])
    cohorts, suite, fixtures = groups()
    if fixtures != gold['fixtures'] or suite['version'] != gold['version'] or gold['frames'] != source['frames']:
        raise ValueError('Different fixture release or frame roster')
    rows = read(checked(BASE/'cases.json', source['artifacts']['cases.json']))
    approved = {r['key']:r for r in read(checked(APPROVED/'cases.json', gold['artifacts']['cases.json']))}
    sigs = read(SIGNATURES)['frames']
    captures = {c.floor+'/'+c.id:c for group in cohorts.values() for c in group}
    if len(rows)!=56 or set(captures)!=set(approved) or set(approved)!=set(sigs):
        raise ValueError('Missing fixed suite frame')
    jobs = [(captures[r['key']],r,source['artifacts'][r['predictions']],sigs[r['key']],approved[r['key']]) for r in rows]
    out.mkdir(parents=True,exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        results = list(pool.map(frame_job,jobs))
    for index,name in enumerate(('candidates','targets')):
        (out/(name+'.json')).write_text(json.dumps([r[index] for r in results], separators=(',',':'), allow_nan=False),
                                        encoding='utf8')
    scores = {}
    for scope in ['all','development','regression',*sorted({r['layout'] for r in rows})]:
        tally = Counter()
        for pred,_,stats in results:
            if scope in ('all',pred['cohort'],pred['layout']):
                tally.update(stats)
        scores[scope] = dict(tally)
    save(out/'scores.json',scores)
    save(out/'manifest.json',dict(code=revision(), version=suite['version'], frames=gold['frames'],fixtures=fixtures,
         splits=gold['splits'],protocol=gold['protocol'],reference_sha256=digest(ARCHIVE),
         signatures_sha256=digest(SIGNATURES),source_manifest_sha256=BASE_SHA,
         approved_manifest_sha256=digest(APPROVED/'manifest.json'),
         candidates='Ordinary reading, all pairs of returned axis peaks, and strong single peaks transferred by measured plane aspect.',
         phase='One estimate per axis and pitch from the production joint finder; unresolved phase is allowed for dimension scores.',
         files={p.name:digest(p) for p in sorted(out.iterdir())}))
    print(json.dumps(scores['all']))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=4)
    args = cli.parse_args()
    run(args.out.resolve(),args.workers)
