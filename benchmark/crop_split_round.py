"""Replay split-only crop Planarity against the 513-surface finder on the fixed exterior suite."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark.afterpass_round import COHORTS, picture_table
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.ledge_join_round import score_dimensions, summary
from benchmark.mask_metrics import PROTOCOL, footprint
from benchmark.normal_round import groups
from benchmark.plane_acceptance_round import BASE
from benchmark.split_round import save_and_score
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.wall_join_test import checked, read_gzip
from benchmark.zoom_plane_fields import TILING, registered_tiles
from engine import crop_splits, planarity_fusion
from engine.planarity_entry import PRESET, RELEASE, selected_settings
from engine.planarity_source import SETTINGS as MODEL_SETTINGS
from runtime.mode import image_only, mark_image_only


PREVIOUS = ROOT / 'benchmark/reviews/guard-zoom-01'
ZOOM = PREVIOUS / 'candidate-runs/zoom'
FIELDS = ZOOM / 'crop-fields'
NAMES = {'current': '513-surface baseline', 'split': 'Crop splits only'}


def support_mask(part):
    width, height = part['support']['size']
    mask = np.zeros((height, width), bool)
    for y, start, stop in part['support']['runs']:
        mask[y, start:stop] = True
    return mask


def audit_partition(parents, parts, events):
    """Verify exact parent ownership on each parent's own mask lattice, before resampling."""
    by_id = {p['id']: p for p in parts}
    before = {p['id']: p for p in parents}
    audit = []
    for event in events:
        original = support_mask(before[event['parent']])
        union = np.zeros_like(original)
        for child_id in event['children']:
            child = by_id[child_id]
            mask = support_mask(child)
            if child['parent_id'] != event['parent'] or mask.shape != original.shape or np.any(union & mask):
                raise ValueError(f"Misaligned or overlapping children of {event['parent']}")
            union |= mask
        if not np.array_equal(union, original):
            raise ValueError(f"Split changed the support of {event['parent']}")
        audit.append(dict(parent=event['parent'], children=event['children'], source_mask_size=list(original.shape[::-1]),
                          original_samples=int(original.sum()), child_union_samples=int(union.sum())))
    return audit


def union_at_scoring_resolution(parts, reference):
    mask = np.zeros(reference['id_map'].shape, bool)
    for part in parts:
        mask |= footprint(part, mask.shape, reference['image_size'], use_mask=True)
    return mask


def evaluate(job):
    row, capture, out, prior, hashes = job
    with Image.open(checked(capture.path('image'), row['sha256'])) as source:
        image = source.convert('RGB')
    parents = read_gzip(checked(BASE/row['predictions'], hashes['baseline']))['planarity']
    path = FIELDS/'fields'/(row['sha256']+'.npz')
    with np.load(checked(path, hashes['fields']), allow_pickle=False) as archive:
        data = dict(archive)
    if tuple(data['image_size']) != image.size:
        raise ValueError('Crop evidence comes from another image size')
    start = perf_counter()
    parts, events = crop_splits.predict(image, parents, registered_tiles(data))
    elapsed = perf_counter()-start
    proposals = dict(current=parents, split=parts)
    result = save_and_score(capture, row, proposals, dict(current=0., split=elapsed), out, 'planarity')
    score_dimensions(capture, row, proposals, result, baseline_method='planarity')

    # Archived closer-view outputs and reference geometry are audit inputs only.
    old = read_gzip(checked(ZOOM/prior['predictions'], hashes['zoom']))['zoom']
    added = {e['id'] for e in prior['events'] if e['action']=='add'}
    if parts != [p for p in old if p['id'] not in added] or events != [e for e in prior['events'] if e['action']=='split']:
        raise ValueError(f"Removing additions changed a split in {row['key']}")
    result['events'] = events
    result['parent_partitions'] = audit_partition(parents, parts, events)
    _, reference = recorded(capture)
    before = union_at_scoring_resolution(parents, reference)
    after = union_at_scoring_resolution(parts, reference)
    if not np.array_equal(before, after):
        raise ValueError(f"Union of the scored masks changed in {row['key']}")
    result['support_audit'] = dict(scoring_lattice=list(before.shape[::-1]), covered_samples=int(after.sum()),
                                   changed_samples=0, old_zoom_additions_omitted=len(added),
                                   matches_archived_split_records=True)
    return result


@image_only()
def run(out, workers):
    baseline = read(BASE/'manifest.json')
    rows = read(checked(BASE/'cases.json', baseline['artifacts']['cases.json']))
    checked(ARCHIVE, baseline['reference_sha256'])
    previous = read(PREVIOUS/'manifest.json')
    zoom = read(checked(ZOOM/'manifest.json', previous['artifacts']['candidate-runs/zoom/manifest.json']))
    zoom_rows = read(checked(ZOOM/'cases.json', zoom['artifacts']['cases.json']))
    old = {r['key']: r for r in zoom_rows}
    field_sha = zoom['fields_manifest_sha256']
    fields = read(checked(FIELDS/'manifest.json', field_sha))
    cohorts, suite, fixtures = groups()
    captures = {f'{c.floor}/{c.id}': c for group in cohorts.values() for c in group}
    roster = {r['key']:r['sha256'] for r in rows}
    if (len(rows)!=56 or len(set(roster.values()))!=52 or roster.keys()!=captures.keys() or
            {r['key']:r['sha256'] for r in zoom_rows}!=roster or fixtures!=baseline['fixtures'] or
            fixtures!=zoom['fixtures'] or zoom['reference_sha256']!=baseline['reference_sha256']):
        raise ValueError('Different frozen fixture set')
    if (any(fields.get(k)!=v for k,v in RELEASE.items()) or fields['model_settings']!=selected_settings(MODEL_SETTINGS)
            or fields['settings']!=TILING or set(fields['source_frames'])!=set(roster.values())):
        raise ValueError('Different crop model, tiling or image hashes')
    dev = [r for r in rows if r['cohort']=='development']
    if len(dev)!=20 or {r['layout'] for r in dev}!={'wing','shopfront'}:
        raise ValueError('Unexpected development split')
    out.mkdir(parents=True, exist_ok=False)
    code = revision()
    freeze = dict(selection='Previously proposed after reviewing guard-zoom-01; one rule fixed before this replay.',
                  standalone_additions=False, counter='masked-repeat', fields_manifest_sha256=field_sha,
                  backend_model=RELEASE, tiling=TILING, preset=PRESET, settings=planarity_fusion.SETTINGS,
                  frames=[r['key'] for r in dev])
    save(out/'development-freeze.json', freeze)
    jobs = []
    for row in rows:
        name = f"fields/{row['sha256']}.npz"
        prior = old[row['key']]
        jobs.append((row,captures[row['key']],out,prior,
                     dict(baseline=baseline['artifacts'][row['predictions']], fields=fields['artifacts'][name],
                          zoom=zoom['artifacts'][prior['predictions']])))
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        results = list(pool.map(evaluate, jobs))
    score = {'all':summary(results,NAMES)}
    for group in (*COHORTS, *sorted({r['layout'] for r in results})):
        score[group] = summary([r for r in results if group in (r['cohort'],r['layout'])],NAMES)
    save(out/'cases.json', results)
    save(out/'scores.json', score)
    (out/'pictures.md').write_text(picture_table(results,NAMES,NAMES), encoding='utf8')
    totals = Counter(action['action'] for row in results for action in row['events'])
    save(out/'manifest.json',dict(
        run_id=out.name,code=code,version=suite['version'],protocol=PROTOCOL,fixtures=fixtures,
        frames=baseline['frames'],splits=baseline['splits'],capture=baseline['capture'],
        reference_sha256=baseline['reference_sha256'],methods=NAMES,baseline='plane-recovery',
        stages={'source':'one source screenshot and its frozen model outputs', 'outlines':'Regions First + six-model Plane Recovery',
                'split':'one supported crop partition per existing surface; no new foreground', 'counter':'masked-repeat'},
        selection='development-freeze.json', settings=freeze,
        source_manifests={p.relative_to(ROOT).as_posix(): digest(p) for p in
                          (BASE/'manifest.json', PREVIOUS/'manifest.json', ZOOM/'manifest.json', FIELDS/'manifest.json')},
        checkpoint_sha256=RELEASE['checkpoint_sha256'],new_inference_s=0.,source_model_timing=zoom['source_model_timing'],
        source_model_load_s=zoom['source_model_load_s'], cached_split_seconds=score['all']['split']['cached_extra_seconds'],
        timing='Cached split seconds exclude field decoding, full-frame inference, scoring and block counting.',
        audit=dict(splits=totals['split'],frames_identical_to_previous_zoom_without_additions=len(results),
                   frames_with_identical_foreground=len(results),native_partition_equal_for_every_split=True),
        scores='scores.json',per_picture='pictures.md',reproduce=r'.\Invoke-SurfaceRound.ps1 -Round crop-split',
        artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print([(name, s['matched'], s['unpaired'], s['opening_filled'], s['sizes']['correct'])
           for name, s in score['all'].items()], flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'crop-split-{uuid4().hex[:12]}')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('At least one worker is required.')
    run(args.out.resolve(),args.workers)
