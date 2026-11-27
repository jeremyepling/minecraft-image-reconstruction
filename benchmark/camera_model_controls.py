"""Freeze and independently grade the Java 26.3 camera-model controls."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZipFile

os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

from benchmark import expanded_fixed_handoffs as fixed
from benchmark import expanded_scores as scores
from benchmark import typed_structures as typed
from benchmark.expanded_baseline import COHORTS, MEASURED_REVISION, ROOT, SOURCE, WORKTREES, digest, frames, read, save
from benchmark.expanded_score_validation import best_translation_fast
from benchmark.mask_blocks import decoded_pack
from benchmark.strip_visibility_audit import horizontal_slivers


REVIEW = ROOT / 'benchmark/reviews/camera-model-controls-01'
RELEASES = ROOT / 'benchmark/reviews'
DEFAULT_BUILDS = WORKTREES / 'full-expanded-controls/runs/expanded-baseline-01'
REPEAT_BUILDS = WORKTREES / 'camera-horizon-floor/runs/camera-horizon-floor-01'
DATASETS = {scores.FIXED_DATASET: 104, **COHORTS}


def check(actual, expected, what):
    if actual != expected:
        raise ValueError(f'{what} differs: {actual!r} != {expected!r}')


def indexes(path, expected_hash):
    check(digest(path), expected_hash, str(path))
    rows = read(path)
    result = {row['id']: row for row in rows}
    check(len(result), len(rows), f'{path} unique frame IDs')
    return result


@lru_cache(maxsize=1)
def releases():
    ordinary = RELEASES / 'expanded-baseline-01'
    repeat = RELEASES / 'camera-horizon-floor-01'
    base_summary = read(ordinary / 'summary.json')
    repeat_summary = read(repeat / 'summary.json')
    check(repeat_summary['control_cases_sha256'], base_summary['cases_sha256'], 'paired case ablations')
    check(digest(ordinary / 'manifest.json'), base_summary['manifest_sha256'], 'ordinary manifest')
    ordinary_rows = indexes(ordinary / 'cases.json', base_summary['cases_sha256'])
    repeat_rows = indexes(repeat / 'cases.json.gz', repeat_summary['cases_sha256'])
    check(ordinary_rows.keys(), repeat_rows.keys(), 'paired frame keys')
    return read(ordinary / 'manifest.json'), base_summary, repeat_summary, ordinary_rows, repeat_rows


def frozen_frames():
    _, _, _, ordinary, repeat = releases()
    entries = [*fixed.fixed_frames(), *frames(SOURCE, COHORTS)]
    check(Counter(e['dataset'] for e in entries), DATASETS, 'cohort counts')
    check(len({e['id'] for e in entries}), len(entries), 'unique frame IDs')
    check({e['id'] for e in entries}, set(ordinary), 'published baseline frame set')
    for entry in entries:
        key = entry['id']
        base, repeated = ordinary[key], repeat[key]
        for field in ('dataset', 'split'):
            check(entry[field], base[field], f'{key} ordinary {field}')
            check(entry[field], repeated[field], f'{key} repeat {field}')
        check(entry['rgb_sha256'], base['source_sha256'], f'{key} ordinary RGB')
        check(entry['rgb_sha256'], repeated['source_sha256'], f'{key} repeat RGB')
        check(base['pack_sha256'], repeated['pack_sha256'], f'{key} linked baseline pack')
        check(base['camera_usable'], repeated['ordinary_supported'], f'{key} ordinary camera flag')
        attempt = repeated['arms'].get('source')
        rgb = entry.get('rgb') or Path(entry['frame']) / 'rgb.png'
        yield {**entry, 'rgb': rgb, 'ordinary_supported': base['camera_usable'],
               'repeat_supported': base['camera_usable'] or attempt['camera']['P'] is not None}


def data_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def inventory_hash(blocks):
    return data_hash([(*position, name) for position, name in sorted(blocks.items())])


def write_frames(output):
    rows = list(frozen_frames())
    fields = ('id', 'dataset', 'split', 'layout', 'family', 'palette', 'shape_sha256', 'rgb_sha256',
              'dataset_manifest_sha256', 'ordinary_supported', 'repeat_supported')
    save(output / 'frames.json', [{name: row.get(name) for name in fields} for row in rows])


def fixed_reference(entry):
    labels, ids, transform = fixed.fixed_reference(entry)
    truth = {tuple(b['xyz_world']): b['block_id'] for b in labels['blocks']}
    optional = horizontal_slivers(labels, ids)
    files = fixed._jobs()[entry['id']][1]['files']
    hashes = {name: value for name, value in files.items()
              if name in ('camera.json', 'labels.json', 'labels.npz', 'world-readback.json')}
    return scores.FrameReference(truth, optional, labels['camera'], {}, hashes), transform


def fixed_archive_score(entry, pack, base):
    reference, transform = fixed_reference(entry)
    local = decoded_pack(BytesIO(pack))
    registered = typed.aligned_sites(list(local), transform, (0, 0, 0))
    translation, maximum = best_translation_fast(registered, reference.truth)
    saved_translation = base['translation']
    aligned = typed.aligned_sites(list(local), transform, saved_translation)
    predictions = dict(zip(aligned, local.values(), strict=True))
    geometry = typed.geometry_score(predictions, reference.truth, reference.optional)
    material = typed.score(predictions, reference.truth, reference.optional, set())
    check(geometry, base['geometry'], f'{entry["id"]} fixed geometry')
    check(maximum, geometry['strict']['correct'], f'{entry["id"]} fixed optimum translation')
    check({k: material[k] for k in ('strict', 'facade', 'material')},
          {k: base['scores'][k] for k in ('strict', 'facade', 'material')}, f'{entry["id"]} fixed typed score')
    return local, reference, transform.tolist(), translation != tuple(saved_translation)


def verify(task):
    entry, ordinary_dir, repeat_dir = task
    _, _, _, ordinary, repeat = releases()
    key, dataset = entry['id'], entry['dataset']
    base, repeat_row = ordinary[key], repeat[key]
    check(digest(entry['rgb']), entry['rgb_sha256'], f'{key} image')
    fallback = base
    tie = False
    if dataset == scores.FIXED_DATASET:
        _, _, archive = fixed._release()
        with ZipFile(archive) as source:
            pack = source.read(f'{key}/reconstruction.zip')
        check(sha256(pack).hexdigest(), base['pack_sha256'], f'{key} fixed pack')
        local, reference, transform, tie = fixed_archive_score(entry, pack, base)
        base = dict(base, reference_sha256=reference.hashes)
    else:
        build = ordinary_dir / dataset / key
        reference = scores.load_reference(entry)
        measured = scores.score_task((entry, ordinary_dir))
        for field in ('geometry', 'scores', 'camera_usable', 'reference_sha256', 'pack_sha256', 'translation'):
            check(measured[field], base[field], f'{key} ordinary {field}')
        local = decoded_pack(build / 'reconstruction.zip')
        transform = measured['registration']['transform']
        if not base['camera_usable']:
            folder = repeat_dir / key
            cached = read(folder / 'repeat.json')
            accepted = repeat_row['arms']['source']
            check(cached['rgb_sha256'], entry['rgb_sha256'], f'{key} repeat RGB cache')
            check(cached['camera'], accepted['camera'], f'{key} repeat camera cache')
            if entry['repeat_supported']:
                fallback = scores.score_build(entry, folder / 'repeat')
                for field in ('geometry', 'scores', 'pack_sha256'):
                    check(fallback[field], accepted[field], f'{key} repeat {field}')
            else:
                for field in ('geometry', 'scores', 'pack_sha256'):
                    check(base[field], accepted[field], f'{key} unrescued repeat {field}')
                fallback = dict(base, **scores.camera_fields(accepted['camera']))
    repeated_local = (local if fallback['pack_sha256'] == base['pack_sha256'] else
                      decoded_pack(repeat_dir / key / 'repeat/reconstruction.zip'))
    evidence = dict(id=key, dataset=dataset, split=entry['split'], rgb_sha256=entry['rgb_sha256'],
                    source_reference_sha256=reference.hashes, truth_sha256=inventory_hash(reference.truth),
                    facade_exempt_sha256=data_hash(sorted(reference.optional)), fixed_translation_tie=bool(tie),
                    ordinary=dict(zip_sha256=base['pack_sha256'], decoded_sha256=inventory_hash(local),
                                  registration=transform),
                    repeat=dict(zip_sha256=fallback['pack_sha256'], decoded_sha256=inventory_hash(repeated_local)))
    return base, fallback, evidence


def summary(cases, entries):
    result = scores.summarize(cases, entries)
    for dataset, cohort in result.items():
        dataset_rows = [r for r in cases if r['dataset'] == dataset]
        for split, group in ((None, cohort), *cohort['splits'].items()):
            subset = [r for r in dataset_rows if split is None or r['split'] == split]
            group['camera']['supported_empty'] = sum(r['camera_usable'] and r['exported'] == 0 for r in subset)
    return result


def freeze(output, ordinary_dir, repeat_dir, workers):
    entries = list(frozen_frames())
    write_frames(output)
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(verify, ((e, ordinary_dir, repeat_dir) for e in entries)))
    ordinary, repeat, inventory = map(list, zip(*results, strict=True))
    release, base_summary, repeat_summary, _, _ = releases()
    save(output / 'inventory.json.gz', inventory)
    save(output / 'baseline-cases.json.gz', {'ordinary': ordinary, 'repeat': repeat})
    save(output / 'baseline-summary.json', {'ordinary': summary(ordinary, entries), 'repeat': summary(repeat, entries)})
    save(output / 'manifest.json', dict(schema=1, minecraft_version='26.3', production_revision=MEASURED_REVISION,
         dataset_sha256=release['dataset_sha256'], fixed=release['fixed'],
         atlas_sha256=repeat_summary['atlas_sha256'], ordinary_cases_sha256=base_summary['cases_sha256'],
         repeat_cases_sha256=repeat_summary['cases_sha256'],
         hashes={name: digest(output / name) for name in
                 ('frames.json', 'inventory.json.gz', 'baseline-cases.json.gz', 'baseline-summary.json')}))
    print(f'Verified {len(entries)} RGB/reference pairs and ordinary packs, '
          f'{sum(b["pack_sha256"] != a["pack_sha256"] for a, b in zip(ordinary, repeat, strict=True))} repeat packs.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('frames', 'freeze'))
    parser.add_argument('--output', type=Path, default=REVIEW)
    parser.add_argument('--ordinary-builds', type=Path, default=DEFAULT_BUILDS)
    parser.add_argument('--repeat-builds', type=Path, default=REPEAT_BUILDS)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'frames':
        write_frames(args.output)
    else:
        freeze(args.output, args.ordinary_builds, args.repeat_builds, args.workers)


if __name__ == '__main__':
    main()
