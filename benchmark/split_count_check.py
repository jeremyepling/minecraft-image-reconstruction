"""Replay the ordinary masked texture counter on the development-selected surface finders."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from time import perf_counter

from PIL import Image

from benchmark.camera_round import digest, read, ROOT, save
from benchmark.count_metrics import count_totals, reference_planes
from benchmark.mask_metrics import measure
from benchmark.normal_round import groups
from benchmark.perimeter_round import downstream_counts
from benchmark.surface_reference import ARCHIVE, recorded
from runtime.mode import image_only


@image_only()
def run(source, out):
    selected = read(source/'development-selection.json')
    methods = ('current', selected['wall'], selected['strip'])
    source_manifest = read(source/'manifest.json')
    cohorts, _, fixtures = groups()
    if fixtures != source_manifest['fixtures'] or digest(ARCHIVE) != source_manifest['reference_sha256']:
        raise ValueError('The visible geometry or fixed screenshots differ from the surface round.')
    def verified(relpath):
        path = source/relpath
        if digest(path) != source_manifest['artifacts'][relpath]:
            raise ValueError(f'Saved artifact changed: {path}')
        return path
    verified('development-selection.json')
    cases = read(verified('cases.json'))
    lookup = {f'{c.floor}/{c.id}': c for cohort in cohorts.values() for c in cohort}
    if len(cases) != len(lookup) or set(r['key'] for r in cases) != lookup.keys():
        raise ValueError('Surface round has a different set of pictures.')
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        key = case['key']
        capture = lookup[key]
        if digest(capture.path('image')) != case['sha256']:
            raise ValueError(f'Screenshot changed for {key}')
        with Image.open(capture.path('image')) as source_image:
            image = source_image.convert('RGB')
        with gzip.open(verified(case['predictions']), 'rt', encoding='utf8') as f:
            predictions = json.load(f)
        start = perf_counter()
        counts = {}
        # Predictions with identical full input records on this frame produce the same count.
        # Each distinct proposal is measured before recorded geometry is accessed.
        cache = {}
        for name in methods:
            readings = []
            for part in predictions[name]:
                serialized = json.dumps(part, sort_keys=True, separators=(',',':'))
                if serialized not in cache:
                    cache[serialized] = downstream_counts(image, [part])[0]
                readings.append(dict(cache[serialized]))
            counts[name] = readings
        elapsed = perf_counter() - start
        _, ref = recorded(capture)
        sizes = reference_planes(ref, read(capture.path('labels'))['faces'])
        eligible = {pid for pid, p in ref['patches'].items() if p['eligible']}
        methods_data = {}
        for name in methods:
            readings = counts[name]
            _, pairs = measure(predictions[name], ref, use_mask=True)
            if sorted(pid for _, pid in pairs) != sorted(case['scopes'][name]['matched']):
                raise ValueError(f'Surface matches changed for {key}/{name}.')
            detail = {}
            for i, pid in pairs:
                rec = readings[i]
                expected = sizes[pid]['size']
                rec['expected'] = expected
                guess = rec['measures']['masked-repeat']['dimensions']['blocks']
                detail[pid] = dict(proposal=rec['id'], expected=expected, guess=guess,
                                   status='missing' if None in guess else 'correct' if guess==expected else 'incorrect')
            for pid in eligible - detail.keys():
                detail[pid] = dict(proposal=None, expected=sizes[pid]['size'], guess=[None,None], status='missing')
            totals = count_totals([dict(eligible=len(eligible), surfaces=readings)], 'masked-repeat')
            if totals['correct']!=sum(r['status']=='correct' for r in detail.values()):
                raise ValueError('Recorded block sizes disagree with the suite counter.')
            current = methods_data.get('current', {}).get('references',detail)
            difference = dict(gained=sorted(p for p in eligible if detail[p]['status']=='correct' and current[p]['status']!='correct'),
                              lost=sorted(p for p in eligible if detail[p]['status']!='correct' and current[p]['status']=='correct'))
            methods_data[name] = dict(totals=totals, comparisons=difference, references=detail)
        rows.append(dict(key=key, sha256=case['sha256'], layout=case['layout'], cohort=case['cohort'],
                         counting_seconds_for_shared_proposals=elapsed, methods=methods_data))
    totals = {}
    for scope in ('all', 'development', 'previously_reviewed_regression', *sorted({r['layout'] for r in rows})):
        members = [r for r in rows if scope in ('all', r['cohort'], r['layout'])]
        totals[scope] = {}
        for name in methods:
            combined = Counter()
            for row in members:
                rec = row['methods'][name]
                combined.update({k:rec['totals'][k] for k in ('eligible','matched','correct','incorrect','missing','unmatched_returned')})
                combined.update({k:len(pids) for k,pids in rec['comparisons'].items()})
            totals[scope][name] = dict(combined)
    save(out/'cases.json',rows)
    save(out/'scores.json',totals)
    save(out/'manifest.json',dict(run_id=out.name, version=source_manifest['version'], counter='masked-repeat',
         stages=dict(finder=methods, dimensions='visible support extent', scorer='count_totals'),
         source_run=source.name, source_inputs={p:digest(source/p) for p in ('cases.json','development-selection.json','protocol.json')},
         fixtures=fixtures, reference_sha256=digest(ARCHIVE), frames=[dict(key=r['key'],sha256=r['sha256']) for r in rows],
         note='Wrong and missing use all eligible visible surfaces. Unmatched complete proposals are counted separately.',
         scores=totals['all'], artifacts={n:digest(out/n) for n in ('scores.json','cases.json')},
         source_files={p:digest(ROOT/p) for p in ('engine/masked_count.py','engine/surface_dimensions.py',
                     'engine/surface_sample.py','benchmark/count_metrics.py','benchmark/perimeter_round.py')},
         runner_sha256=digest(Path(__file__))))
    return totals['all']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path,help='Completed separation round directory')
    parser.add_argument('--out',required=True,type=Path,help='New private output directory')
    args=parser.parse_args()
    print(json.dumps(run(args.run,args.out),indent=2))
