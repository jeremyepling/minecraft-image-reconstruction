"""Compare a completed single-image Wall Recovery build with a saved evaluation case."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from benchmark.camera_round import ROOT, digest, read
from benchmark.mask_metrics import measure
from benchmark.normal_round import groups
from benchmark.surface_reference import recorded


def verify(build, key):
    archive = ROOT/'benchmark/reviews/mobile-combined-01'
    rows = [r for split in ('development', 'regression') for r in read(archive/split/'cases.json')]
    row = next(r for r in rows if r['key'] == key)
    manifest = read(build/'build-manifest.json')
    if manifest['image_sha256'] != row['sha256'] or manifest['stages']['surface_boundary'] != 'wall-recovery':
        raise ValueError('Image or selected surface method differs from the selected case')
    predicted = read(build/'surfaces.json')['planes']
    path = archive/row['predictions']
    if digest(path) != row['prediction_sha256']:
        raise ValueError('Archived combined prediction changed')
    with gzip.open(path, 'rt', encoding='utf8') as f:
        expected = json.load(f)
    if [p['id'] for p in predicted] != [p['id'] for p in expected]:
        raise ValueError('Predicted region IDs differ')
    comparable = [{k: actual.get(k) for k in frozen} for actual, frozen in zip(predicted, expected)]
    if comparable != expected:
        raise ValueError('At least one saved region property differs')
    case = next(c for cohort in groups()[0].values() for c in cohort if f'{c.floor}/{c.id}' == key)
    scores, pairs = measure(predicted, recorded(case)[1], use_mask=True)
    assignments = {predicted[i]['id']: pid for i, pid in pairs}
    if assignments != row['assignments'] or scores != row['split']:
        raise ValueError('Matched reference IDs or surface scores differ')
    counts = read(build/'counts.json')['surfaces']
    if [c['id'] for c in counts] != [p['id'] for p in predicted]:
        raise ValueError('The ordinary Block Counter did not receive every predicted region')
    return dict(fixture=key, screenshot_sha256=row['sha256'], original_png_basename=case.path('image').name,
                stage_order=manifest['stages']['boundary_sequence'], model_sha256=manifest['model_sha256'],
                visible_proposals=len(predicted), reference_surfaces=scores['eligible'], matched=len(assignments),
                assignments=assignments, mobile_additional_reference_ids=row['gained'],
                all_saved_region_properties_equal=True, canonical_regions_sha256=hashlib.sha256(
                    json.dumps(comparable, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                source_prediction_sha256=row['prediction_sha256'], counter=manifest['stages']['block_count'],
                block_count_records=len(counts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = json.dumps(verify(args.build, args.case), indent=2) + '\n'
    if args.out:
        args.out.write_text(result, encoding='utf8')
    print(result, end='')
