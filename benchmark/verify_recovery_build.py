"""Compare a completed single-image Wall Recovery build with a saved evaluation case."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from benchmark.camera_round import ROOT, digest, read
from benchmark.mask_metrics import measure
from benchmark.surface_reference import recorded
from benchmark.wall_join_test import capture_index
from engine.planarity_entry import METHOD as PLANE_RECOVERY
from engine.surface_model import CROP_SPLIT


def verify(build, key, source_run=None, data=None):
    manifest = read(build/'build-manifest.json')
    boundary=manifest['stages']['surface_boundary']
    methods={'wall-recovery-single-v2':'single-v2','wall-recovery-single-pro':'single-pro',
             PLANE_RECOVERY:'planarity',CROP_SPLIT:'split'}
    method=methods.get(boundary)
    sky_filtered = manifest['stages'].get('background_filter') == 'clear-noon-sky'
    if sky_filtered:
        method = 'sky'
    if method:
        review = {CROP_SPLIT:'crop-split-01',PLANE_RECOVERY:'afterpass-models-01'}.get(boundary,'single-wall-depth-01')
        if sky_filtered:
            review = 'sky-01'
        archive=source_run or ROOT/'benchmark/reviews'/review
        source=read(archive/'manifest.json')
        if digest(archive/'cases.json')!=source['artifacts']['cases.json']:
            raise ValueError('Archived single-wall scores changed')
        rows=read(archive/'cases.json')
    elif boundary=='wall-recovery':
        archive=source_run or ROOT/'benchmark/reviews/mobile-combined-01'
        rows=[r for split in ('development','regression') for r in read(archive/split/'cases.json')]
    else:
        raise ValueError(f'Unsupported surface recovery method {boundary}')
    row=next(r for r in rows if r['key']==key)
    if manifest['image_sha256'] != row['sha256']:
        raise ValueError('Image or selected surface method differs from the selected case')
    predicted = read(build/'surfaces.json')['planes']
    path = archive/row['predictions']
    expected_hash=source['artifacts'][row['predictions']] if method else row['prediction_sha256']
    if digest(path) != expected_hash:
        raise ValueError('Archived combined prediction changed')
    with gzip.open(path, 'rt', encoding='utf8') as f:
        payload=json.load(f)
    expected=payload[method] if method else payload
    if [p['id'] for p in predicted] != [p['id'] for p in expected]:
        raise ValueError('Predicted region IDs differ')
    comparable = [{k: actual.get(k) for k in frozen} for actual, frozen in zip(predicted, expected)]
    if comparable != expected:
        raise ValueError('At least one saved region property differs')
    data = data or ROOT/'benchmark/data'
    case = capture_index(data)[key]
    scores, pairs = measure(predicted, recorded(case,data/'visible-geometry-v1.zip')[1], use_mask=True)
    assignments = {predicted[i]['id']: pid for i, pid in pairs}
    expected_scores=row['scores'][method] if method else row['split']
    expected_ids=row['scopes'][method]['matched'] if method else list(row['assignments'].values())
    if set(assignments.values())!=set(expected_ids) or scores!=expected_scores or (not method and assignments!=row['assignments']):
        raise ValueError('Matched reference IDs or surface scores differ')
    counts = read(build/'counts.json')['surfaces']
    if [c['id'] for c in counts] != [p['id'] for p in predicted]:
        raise ValueError('The ordinary Block Counter did not receive every predicted region')
    result = dict(fixture=key, screenshot_sha256=row['sha256'], original_png_basename=case.path('image').name,
                stage_order=manifest['stages']['boundary_sequence'], model_sha256=manifest['model_sha256'],
                visible_proposals=len(predicted), reference_surfaces=scores['eligible'], matched=len(assignments),
                assignments=assignments,
                **({'split_additional_reference_ids':row['scopes'][method]['gained'],
                    'crop_splits':manifest['surface_finder']['boundary_settings']['crop_splits']} if boundary==CROP_SPLIT else
                   {'planarity_additional_reference_ids':row['scopes'][method]['gained'],
                    'planarity':manifest['surface_finder']['boundary_settings']['planarity']} if boundary==PLANE_RECOVERY else
                   {'single_wall_additional_reference_ids':row['scopes'][method]['gained']} if method else
                   {'mobile_additional_reference_ids':row['gained']}),
                all_saved_region_properties_equal=True, canonical_regions_sha256=hashlib.sha256(
                    json.dumps(comparable, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                source_prediction_sha256=expected_hash, counter=manifest['stages']['block_count'],
                block_count_records=len(counts))
    if boundary==CROP_SPLIT:
        if sky_filtered:
            result['background_filter'] = manifest['surface_finder']['boundary_settings']['background_filter']
        from PIL import Image
        from benchmark.perimeter_round import downstream_counts
        with Image.open(case.path('image')) as src:
            gold = downstream_counts(src.convert('RGB'), expected)
        for actual, frozen in zip(counts, gold, strict=True):
            reading = frozen['measures']['masked-repeat']
            for field in ('counts','phase','dimensions'):
                if actual[field] != reading[field]:
                    raise ValueError(f"Block Counter {field} differs for {actual['id']}")
        timings = manifest['surface_finder']['boundary_settings']['planarity']['inference']
        result.update(same_count_phase_and_dimensions_for_every_proposal=True,
                      candidate_build=manifest.get('candidate_verification'),
                      raw_model_source=timings['source'], model_revision=timings['model_revision'],
                      model_load_s=timings['load_s'], full_frame_inference_s=timings['inference_s'],
                      full_frame_segmentation_s=timings['segmentation_s'],
                      baseline_opening_samples_covered=row['scores']['current']['opening_filled'],
                      candidate_opening_samples_covered=scores['opening_filled'], opening_samples=scores['opening_pixels'],
                      paired_sizes=row['sizes'][method])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source-run',type=Path,help='Optional source review directory for archived predictions.')
    parser.add_argument('--data',type=Path,help='Optional root of the frozen fixture data for evaluation.')
    args = parser.parse_args()
    result = json.dumps(verify(args.build, args.case,args.source_run,args.data), indent=2) + '\n'
    if args.out:
        args.out.write_text(result, encoding='utf8')
    print(result, end='')
