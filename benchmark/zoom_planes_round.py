"""Test whether closer RGB views give the existing plane finder better outlines."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.afterpass_round import COHORTS, picture_table, validate_fields
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.ledge_join_round import score_dimensions, summary
from benchmark.mask_metrics import PROTOCOL, footprint
from benchmark.normal_round import groups
from benchmark.plane_acceptance_round import BASE, PLANES
from benchmark.planarity_afterpass import raw_diagnostic
from benchmark.split_round import save_and_score
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.wall_join_test import checked, read_gzip
from benchmark.zoom_plane_fields import TILING, registered_tiles
from engine import planarity_fusion as fusion
from engine.planarity_entry import LABELS, PRESET, RELEASE, selected_settings
from engine.planarity_source import SETTINGS as MODEL_SETTINGS
from runtime.mode import image_only, mark_image_only


NAMES = dict(current='513-surface baseline', zoom='Overlapping close views',
             repeat_full='Diagnostic, full frame once more', repeat_full_nine='Diagnostic, full frame reused nine times')
METHODS = tuple(NAMES)


def predict(image, parts, tiles):
    """Start each parent split from the frozen 513 masks. No split is fed back into the finder."""
    shape = (image.height//4, image.width//4)
    parents = [footprint(part, shape, image.size, True) for part in parts]
    owned = np.logical_or.reduce(parents, axis=0) if parents else np.zeros(shape, bool)
    replacements, added_parts, added_masks, events = {}, [], [], []
    for tile, labels, normals, centers in tiles:
        areas = np.bincount(labels.ravel())
        counts = np.zeros((len(parts), len(areas)), int)
        for index, (part, parent) in enumerate(zip(parts, parents)):
            hits = np.bincount(labels[parent], minlength=len(areas))
            counts[index] = hits
            pieces = fusion.split_parent(image, part, hits, areas, labels, normals, centers)
            if pieces:
                seed_ids = {p['planarity_label'] for p in pieces if p.get('planarity_label') is not None}
                support = sum(int(hits[label]) for label in seed_ids)
                if index not in replacements or support > replacements[index][0]:
                    replacements[index] = (support, tile, pieces)
        # In an overlap, already-accepted additions participate in the identical novelty and duplicate checks.
        if added_masks:
            extra = [np.bincount(labels[m], minlength=len(areas)) for m in added_masks]
            counts = np.vstack([counts, extra])
        for new in fusion.new_regions(image, owned, labels, normals, counts, areas):
            new['id'] = f"zoom-{tile}-{new['id']}"
            mask = footprint(new, shape, image.size, True)
            added_parts.append(new)
            added_masks.append(mask)
            owned |= mask
            events.append(dict(action='add', tile=tile, id=new['id']))
    result = []
    for index, part in enumerate(parts):
        if index in replacements:
            support, tile, pieces = replacements[index]
            result.extend(pieces)
            events.append(dict(action='split', parent=part['id'], tile=tile,
                               support_samples=support, children=[p['id'] for p in pieces]))
        else:
            result.append(part)
    return result + added_parts, events


def raw_scores(global_labels, fields, ref, missed):
    """Pair raw candidate masks with each baseline miss independently, after the predictions were saved."""
    evidence = {r['patch']:dict(patch=r['patch'], global_iou=r['best_iou'], crop_iou=0., unfiltered_crop_iou=0.,
                                crop_tile=None, crop_label=None) for r in raw_diagnostic(global_labels, ref, missed)}
    if evidence:
        for complete in (True, False):
            for tile, labels, _, _ in registered_tiles(fields, complete_only=complete):
                for hit in raw_diagnostic(labels, ref, missed):
                    dest = evidence[hit['patch']]
                    field = 'crop_iou' if complete else 'unfiltered_crop_iou'
                    if hit['best_iou'] > dest[field]:
                        dest[field] = hit['best_iou']
                        if complete:
                            dest.update(crop_tile=tile, crop_label=hit['label'])
    return list(evidence.values())


def raw_summary(rows):
    items = [item for row in rows for item in row['raw_baseline_misses']]
    full = sum(r['global_iou'] >= .5 for r in items)
    both = sum(r['global_iou'] >= .5 and r['crop_iou'] >= .5 for r in items)
    crops = sum(r['crop_iou'] >= .5 for r in items)
    return dict(misses=len(items), full_frame=full, complete_crops=crops, both=both, full_only=full-both,
                crop_only=crops-both, either=full+crops-both, neither=len(items)-full-crops+both,
                unfiltered_crops=sum(r['unfiltered_crop_iou'] >= .5 for r in items),
                newly_visible_and_recovered=sum(r['global_iou'] < .5 <= r['crop_iou'] and r['recovered'] for r in items))


def evaluate(job):
    row, capture, out, fields_dir, hashes = job
    sha = row['sha256']
    name = f'fields/{sha}.npz'
    with Image.open(checked(capture.path('image'), sha)) as source:
        photo = source.convert('RGB')
    parts = read_gzip(checked(BASE/row['predictions'], hashes['parts']))['planarity']
    with np.load(checked(fields_dir/name, hashes['crop']), allow_pickle=False) as archive:
        fields = dict(archive)
    if tuple(fields['image_size']) != photo.size:
        raise ValueError('Crop coordinates refer to another image size')
    start = perf_counter()
    zoom, events = predict(photo, parts, registered_tiles(fields))
    elapsed = perf_counter() - start
    # A full field has no artificial crop edge. The same fusion functions still exclude
    # genuine image-border labels for additions and allow supported splits at the image rim.
    with np.load(checked(PLANES/name, hashes['full']), allow_pickle=False) as archive:
        labels = archive[LABELS]
        full = (labels, archive[LABELS+'_normals'], archive[LABELS+'_centers'])
    timing = dict(current=0., zoom=elapsed)
    predictions = dict(current=parts, zoom=zoom)
    for method, repeats in (('repeat_full', 1), ('repeat_full_nine', len(fields['boxes']))):
        start = perf_counter()
        predictions[method], _ = predict(photo, parts, ((i, *full) for i in range(repeats)))
        timing[method] = perf_counter()-start
    identical = predictions['repeat_full'] == predictions['repeat_full_nine']
    result = save_and_score(capture, row, predictions, timing, out, 'planarity')
    score_dimensions(capture, row, predictions, result, baseline_method='planarity')
    _, ref = recorded(capture)
    matched = set(result['scopes']['current']['matched'])
    missed = {i for i, part in ref['patches'].items() if part['eligible']} - matched
    evidence = raw_scores(labels, fields, ref, missed)
    for item in evidence:
        item['recovered'] = item['patch'] in result['scopes']['zoom']['matched']
    result.update(events=events, raw_baseline_misses=evidence, repeated_full_predictions_identical=identical)
    return result


def make_fields(out, rows, captures, device):
    """Model worker sees only the source files, never the reference geometry."""
    paths = {r['sha256']:str(checked(captures[r['key']].path('image'), r['sha256'])) for r in rows}
    if len(paths) != 52:
        raise ValueError('Unexpected number of distinct screenshots')
    save(out/'inference-roster.json', [dict(sha256=k, path=v) for k,v in paths.items()])
    worktrees = Path(str(ROOT)+'-worktrees')
    if not worktrees.is_dir():
        worktrees = ROOT.parent
    assets = worktrees/'afterpass-planarity/runs/assets'
    exe = os.environ.get('MCIR_RECOVERY_PYTHON') or worktrees/'mask-scalelsd/runs/env/Scripts/python.exe'
    env = os.environ.copy()
    extra = os.environ.get('MCIR_PLANARITY_PYTHONPATH') or str(assets/'deps')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (extra, env.get('PYTHONPATH'))))
    subprocess.run([str(exe), '-B', '-m', 'benchmark.zoom_plane_fields', '--roster', str(out/'inference-roster.json'),
                    '--out', str(out/'crop-fields'), '--device', device,
                    '--upstream', os.environ.get('MCIR_PLANARITY_SOURCE') or str(assets/'PixelwisePlanarity'),
                    '--utils3d', os.environ.get('MCIR_PLANARITY_UTILS3D') or str(assets/'utils3d'),
                    '--checkpoint', os.environ.get('MCIR_PLANARITY_CHECKPOINT') or str(assets/'model.pt')],
                   cwd=ROOT, env=env, check=True)
    return out/'crop-fields'


@image_only()
def run(out, workers, fields=None, device='cuda'):
    base = read(BASE/'manifest.json')
    rows = read(checked(BASE/'cases.json', base['artifacts']['cases.json']))
    checked(ARCHIVE, base['reference_sha256'])
    cohorts, suite, fixtures = groups()
    captures = {f'{c.floor}/{c.id}':c for group in cohorts.values() for c in group}
    roster = {r['key']:r['sha256'] for r in rows}
    if len(rows)!=56 or set(captures)!=set(roster) or fixtures!=base['fixtures']:
        raise ValueError('Different frozen suite')
    checked(PLANES/'manifest.json', base['candidates']['planarity']['fields_manifest_sha256'])
    global_meta = validate_fields(PLANES, fixtures, roster)
    dev = [r for r in rows if r['cohort']=='development']
    if len(dev)!=20 or {r['layout'] for r in dev}!={'wing','shopfront'}:
        raise ValueError('Unexpected development split')
    out.mkdir(parents=True, exist_ok=False)
    freeze = dict(selected='fixed-overlapping-half-frames', tuning='fixed before scoring; zero settings selected with labels',
                  frames=[r['key'] for r in dev], tiling=TILING, fusion_preset=PRESET,
                  fusion_settings=fusion.SETTINGS, model_settings=selected_settings(MODEL_SETTINGS))
    save(out/'development-freeze.json', freeze)
    stamp = revision()
    supplied = fields is not None
    if fields is None:
        fields = make_fields(out, rows, captures, device)
    else:
        fields = Path(shutil.copytree(fields, out/'crop-fields'))
    crop_meta = read(fields/'manifest.json')
    if (any(crop_meta.get(k)!=v for k,v in RELEASE.items()) or crop_meta['settings']!=TILING or
            crop_meta['model_settings']!=selected_settings(MODEL_SETTINGS) or
            set(crop_meta['source_frames'])!=set(roster.values()) or
            set(crop_meta['artifacts'])!={f'fields/{s}.npz' for s in roster.values()}):
        raise ValueError('Incompatible crop cache')
    jobs = []
    for r in rows:
        name = f"fields/{r['sha256']}.npz"
        jobs.append((r, captures[r['key']], out, fields, dict(parts=base['artifacts'][r['predictions']],
                                                               crop=crop_meta['artifacts'][name],
                                                               full=global_meta['artifacts'][name])))
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        results = list(pool.map(evaluate, jobs))
    scopes = {'all':results}
    scopes.update({k:[r for r in results if k in (r['cohort'],r['layout'])]
                   for k in (*COHORTS,*sorted({r['layout'] for r in rows}))})
    scores = {k:summary(rr,METHODS) for k,rr in scopes.items()}
    raw = {k:raw_summary(rr) for k,rr in scopes.items()}
    save(out/'scores.json', scores)
    save(out/'cases.json', results)
    save(out/'raw-scores.json', raw)
    (out/'pictures.md').write_text(picture_table(results,METHODS,NAMES),encoding='utf8')
    source_times = {k:sum(v[k] for v in crop_meta['timing'].values()) for k in
                    ('global_geometry_s','crop_inference_s','crop_segmentation_s')}
    save(out/'manifest.json',dict(run_id=out.name,code=stamp,version=suite['version'],protocol=PROTOCOL,
        fixtures=fixtures,reference_sha256=base['reference_sha256'],frames=base['frames'],splits=base['splits'],
        capture=base['capture'], methods=NAMES, baseline='plane-recovery',selected=freeze['selected'],
        settings=freeze, stages=dict(outlines='Regions First + Wall Recovery + full-frame and crop Planarity',
                                     counter='masked-repeat',crop_fusion_preset=PRESET),
        diagnostics=dict(repeat_full='Reuse global regions against 513 masks with exactly the same acceptance function',
                         repeat_full_nine='Repeat the global evidence in each of the nine crop slots',
                         repeat_full_identical_frames=sum(r['repeated_full_predictions_identical'] for r in results)),
        model=RELEASE,checkpoint_sha256=RELEASE['checkpoint_sha256'], fields_root=os.path.relpath(fields,out).replace(os.sep,'/'),
        fields_manifest_sha256=digest(fields/'manifest.json'), full_fields_manifest_sha256=digest(PLANES/'manifest.json'),
        upstream_manifest_sha256=digest(BASE/'manifest.json'), unique_frames=52, crop_forwards=468, full_geometry_forwards=52,
        source_model_timing=source_times, source_model_load_s=crop_meta['load_s'], used_cached_crop_fields=supplied,
        new_inference_s=0. if supplied else source_times['crop_inference_s']+source_times['global_geometry_s'],
        scores='scores.json',raw_diagnostics='raw-scores.json',scope='Reference geometry opened by scorer only',
        reproduce='python -m benchmark.zoom_planes_round --fields <crop-fields> --out <new-directory>',
        artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}))
    print('Final surfaces / dimensions', [(m,scores['all'][m]['matched'],scores['all'][m]['sizes']['correct']) for m in NAMES])
    print('Raw evidence on baseline misses', raw['all'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fields',type=Path,help='Portable zoom cache, containing manifest.json and fields/.')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    opt = parser.parse_args()
    run(opt.out.resolve(),opt.workers,opt.fields.resolve() if opt.fields else None,opt.device)

