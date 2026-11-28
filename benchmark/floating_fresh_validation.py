"""Replay the frozen floating wall phases on independent building layouts."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

import numpy as np

from benchmark import expanded_crop_ceiling as crops, expanded_wall_source_resample as walls
from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save
from benchmark.full_block_lattice import atlas
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_assembly import axis_vector, oriented
from benchmark.relative_face_scoring import _yaw
from benchmark.sky_crop_guard import pooled
from engine import surface_fields
from PIL import Image


OUT = ROOT / 'runs/floating-fresh'
MODEL_ROOT = WORKTREES / 'normal-round/.deps'
ARCHIVED_FIELDS = WORKTREES / 'full-expanded-controls/runs/expanded-baseline-01'
REVIEW = ROOT / 'benchmark/reviews/floating-fresh-validation-01'
FROZEN = REVIEW / 'frames.json'
MODEL_HASHES = {kind: spec[2] for kind, spec in surface_fields.MODELS.items()}
ARMS = ('ordinary', 'source_resample', 'source_material')


def fixture_hash(path):
    canonical = json.dumps(read(path), sort_keys=True, separators=(',', ':')).encode()
    return sha256(canonical).hexdigest()


def cases(frozen_path=FROZEN):
    fixture = read(frozen_path)
    for dataset, expected in fixture['dataset_manifest_sha256'].items():
        manifest = SOURCE / dataset / 'manifest.json'
        if digest(manifest) != expected:
            raise ValueError(f'Frozen manifest differs: {manifest}')
        records = {entry['id']: entry for entry in read(manifest)['frames']}
        for frozen in (entry for entry in fixture['cases'] if entry['dataset'] == dataset):
            entry = records[frozen['id']]
            source = manifest.parent / 'data' / entry['split'] / 'frames' / entry['id']
            if any(entry[key] != frozen[key] for key in ('rgb_sha256', 'split', 'layout')):
                raise ValueError(f'Frozen frame does not match its dataset: {source}')
            for name, sha in frozen['files_sha256'].items():
                if digest(source / name) != sha:
                    raise ValueError(f'Frozen frame file differs: {source / name}')
            yield dict(entry, dataset=dataset, frame=str(source), dataset_manifest_sha256=expected,
                       family=entry.get('family', entry['layout'].rstrip('0123456789')),
                       variants={'sky': {'image': str(source / 'rgb.png'), 'source_sha256': entry['rgb_sha256']}})


def fields(case):
    relative_path = Path(case['dataset']) / case['id']
    cache = OUT / 'fields' / relative_path
    receipt = cache / 'receipt.json'
    if receipt.exists():
        provenance = read(receipt)
        if (provenance['source_sha256'] != case['rgb_sha256'] or provenance['model_sha256'] != MODEL_HASHES
                or any(digest(cache / f'{kind}.npz') != provenance['fields_sha256'][kind] for kind in MODEL_HASHES)):
            raise ValueError(f'Cached source fields differ: {cache}')
        return cache

    cache.mkdir(parents=True, exist_ok=True)
    previous = ARCHIVED_FIELDS / relative_path
    archive_receipt = previous / 'receipt.json'
    if archive_receipt.exists():
        archived = read(archive_receipt)
        if (archived['source_sha256'] != case['rgb_sha256']
                or any(archived['model_sha256'][kind] != sha for kind, sha in MODEL_HASHES.items())):
            raise ValueError(f'Archived source fields differ: {previous}')
        for kind in MODEL_HASHES:
            copyfile(previous / 'handoffs' / f'{kind}.npz', cache / f'{kind}.npz')
    else:
        image = Image.open(case['variants']['sky']['image']).convert('RGB')
        for kind, (env, name, _) in surface_fields.MODELS.items():
            os.environ[env] = str(MODEL_ROOT / name)
            np.savez_compressed(cache / f'{kind}.npz', value=surface_fields.infer(image, kind))

    save(receipt, dict(source_sha256=case['rgb_sha256'], model_sha256=MODEL_HASHES,
                       fields_sha256={kind: digest(cache / f'{kind}.npz') for kind in MODEL_HASHES},
                       derivation='verified expanded control cache' if archive_receipt.exists() else 'pinned image-only model'))
    return cache


def run(case):
    with patch.object(crops, 'OUT', OUT), patch.object(crops, 'fields', fields):
        crops.paired_case(case, arms=('source',), source_name='ordinary')
    for arm in ('source_resample', 'source_material'):
        receipt = OUT / arm / 'assembly/connected-atlas/sky' / case['id'] / 'score.json'
        if not receipt.is_file():
            walls.run(case, material=arm == 'source_material', output_root=OUT, source=OUT / 'ordinary')
    return case['id']


def ordinary_compass(before, after, ordinary_yaw):
    ordinary = {chart['id']: chart for chart in oriented(before['charts'])[0]}
    candidate = {chart['id']: chart for chart in oriented(after['charts'])[0]}
    transfers = {}
    for ident in sorted(ordinary.keys() & candidate.keys()):
        a, b = ordinary[ident], candidate[ident]
        if a['H'] != b['H'] or a['normal_camera'] != b['normal_camera']:
            continue
        before_axes = [axis_vector(code) for code in (*a['axes'], a['normal_axis'])]
        after_axes = [axis_vector(code) for code in (*b['axes'], b['normal_axis'])]
        compatible = [turn for turn in range(4) if all(np.array_equal(_yaw(source, turn), target)
                      for source, target in zip(after_axes, before_axes, strict=True))]
        if len(compatible) != 1:
            return None
        transfers[ident] = compatible[0]
    if len(set(transfers.values())) == 1:
        turn = next(iter(transfers.values()))
        return dict(ordinary_yaw_quarters=ordinary_yaw, candidate_to_ordinary_quarters=turn,
                    candidate_yaw_quarters=(ordinary_yaw + turn) % 4, shared_charts=transfers)
    return None


def score_case(case):
    name = case['id']
    source = Path(case['frame'])
    reference = {tuple(block['xyz_world']): block['block_id'] for block in read(source / 'labels.json')['blocks']}
    reference_hashes = {path: digest(source / path) for path in ('camera.json', 'labels.json', 'labels.npz')}
    arms, positions = {}, {}
    ordinary_chart = None
    for arm in ARMS:
        folder = OUT / arm / 'assembly/connected-atlas/sky' / name
        receipt = read(folder / 'score.json')
        pack = folder / 'reconstruction.zip'
        placements = read(folder / 'placements.json')
        score = receipt['score']['fixed'] or receipt['score']['relaxed_yaw']
        aligned = {tuple(cell): material for cell, material in placements['aligned']}
        hits = aligned.keys() & reference.keys()
        if (receipt['source_sha256'] != case['rgb_sha256'] or receipt['pack_sha256'] != digest(pack)
                or receipt['reference_sha256'] != reference_hashes
                or decoded_pack(pack) != {tuple(cell): material for cell, material in placements['local']}
                or score['geometry']['strict']['correct'] != len(hits)
                or score['scores']['strict']['correct'] != sum(aligned[p] == reference[p] for p in hits)):
            raise ValueError(f'Export or score disagrees with frozen inputs: {folder}')
        chart_path = 'atlas/connected' if arm == 'ordinary' else 'charts'
        chart = OUT / arm / chart_path / 'sky' / name / 'charts.json'
        if digest(chart) != receipt['chart_sha256']:
            raise ValueError(f'Chart differs from assembly handoff: {chart}')
        arms[arm] = {key: receipt[key] for key in ('chart_sha256', 'pack_sha256', 'charts_seen', 'kept_charts', 'components')}
        arms[arm].update(score=score, fixed_yaw=receipt['score']['fixed'] is not None)
        if arm == 'ordinary':
            ordinary_chart = chart
        if not arms[arm]['fixed_yaw'] and arm != 'ordinary' and arms['ordinary']['fixed_yaw']:
            compass = ordinary_compass(read(ordinary_chart), read(chart), arms['ordinary']['score']['yaw_quarters'])
            if compass is not None:
                compass['agrees_with_relaxed'] = compass['candidate_yaw_quarters'] == score['yaw_quarters']
            arms[arm]['ordinary_compass'] = compass
        positions[arm] = aligned.keys()
    before, after = positions['ordinary'], positions['source_material']
    truth = reference.keys()
    effects = dict(gained=len((after - before) & truth), lost=len((before - after) & truth),
                   extras_added=len((after - before) - truth), extras_removed=len((before - after) - truth))
    for metric in ('geometry', 'scores'):
        a, b = (arms[arm]['score'][metric]['strict']['f1'] for arm in ('ordinary', 'source_material'))
        effects[metric + '_improved'] = int(b > a)
        effects[metric + '_regressed'] = int(b < a)
        effects[metric + '_tied'] = int(b == a)
    fields_receipt = read(fields(case) / 'receipt.json')
    return dict(id=name, dataset=case['dataset'], split=case['split'], layout=case['layout'],
                source_sha256=case['rgb_sha256'], reference_sha256=reference_hashes,
                fields=fields_receipt, arms=arms, paired=effects)


def summarize(entries, frozen_path, output):
    rows = [score_case(case) for case in entries]
    categories = {'all': rows, 'fixed_yaw_all_arms': [r for r in rows if all(a['fixed_yaw'] for a in r['arms'].values())]}
    for key in ('split', 'dataset', 'layout'):
        for value in sorted({entry[key] for entry in rows}):
            categories[value] = [entry for entry in rows if entry[key] == value]
    cohorts = {}
    for key, group in categories.items():
        if not group:
            continue
        effects = Counter()
        for entry in group:
            effects.update(entry['paired'])
        metrics = {}
        for arm in ARMS:
            records = [entry['arms'][arm]['score'] for entry in group]
            materials = [record['scores'] for record in records]
            correct = sum(record['material']['correct'] for record in materials)
            observed = sum(record['material']['positions'] for record in materials)
            metrics[arm] = dict(pooled(records), assigned=sum(r['assigned'] for r in materials),
                                fallback=sum(r['fallback'] for r in materials), exact_ids=correct,
                                material_accuracy=correct / observed if observed else None)
        cohorts[key] = dict(views=len(group), arms=metrics, paired=effects)
        print(key, len(group), {arm: [round(100 * metrics[arm][field]['f1'], 2)
                                     for field in ('geometry_strict', 'geometry_facade', 'scores_strict', 'scores_facade')]
                               for arm in ARMS}, flush=True)
    save(output, dict(frozen_manifest_sha256=fixture_hash(frozen_path), atlas=atlas().provenance, cases=rows, cohorts=cohorts))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--supplemental', action='store_true', help='Use the second, disjoint six-view cohort')
    parser.add_argument('--id', nargs='+')
    args = parser.parse_args()
    os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    prefix = 'supplemental-' if args.supplemental else ''
    frozen = REVIEW / f'{prefix}frames.json'
    results = REVIEW / f'{prefix}results.json.gz'
    entries = list(cases(frozen))
    save(OUT / f'{prefix}manifest.json', dict(frozen_manifest_sha256=fixture_hash(frozen), cases=entries))
    if args.manifest_only:
        return
    if args.score_only:
        summarize(entries, frozen, results)
        return
    selected = [entry for entry in entries if not args.id or entry['id'] in args.id]
    if args.id and len(selected) != len(set(args.id)):
        raise ValueError('Unknown requested fresh frame')
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run, entry) for entry in selected]
        errors = []
        for future in as_completed(futures):
            try:
                print('COMPLETE', future.result(), flush=True)
            except Exception as error:
                print(error.__cause__ or repr(error), flush=True)
                errors.append(error)
        if errors:
            raise ExceptionGroup('Fresh image replay failures', errors)
    if not args.id:
        summarize(entries, frozen, results)


if __name__ == '__main__':
    main()
