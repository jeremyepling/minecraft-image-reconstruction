"""Replay late Floating handoffs on saved, visible-only comparison exports."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from benchmark.floating_stage_report import hash_file, read
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_scoring import _yaw, evaluate_relative_voxels
from benchmark.expanded_scores import FrameReference
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.typed_structures import geometry_score, score
from engine.export import JAVA_26_3_FORMAT, write_pack


ROOT = Path(__file__).resolve().parents[1]
PASSES = {
    'heading-chooser': ('Local heading', 'Reference chooser',
                        'On unresolved views, choose the best of four upright headings on the existing pack.'),
    'exact-material': ('Material assignment', 'Exact at correctly placed coordinates',
                       'Replace IDs at already correct visible coordinates and keep every exported position.'),
    'exact-export': ('Pack writing', 'Exact visible coordinates and IDs',
                     'Supply the full visible reference to the real writer and decode the resulting Java pack.'),
}
FIELDS = ('correct', 'missed', 'extras')


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.part')
    temp.write_text(json.dumps(data, separators=(',', ':')) + '\n', encoding='utf8', newline='\n')
    temp.replace(path)


def dependency_hashes():
    names = ('benchmark/floating_perfect_export.py', 'benchmark/relative_face_scoring.py',
             'benchmark/expanded_score_validation.py', 'benchmark/typed_structures.py',
             'benchmark/strip_visibility_audit.py', 'benchmark/mask_blocks.py',
             'engine/export.py', 'engine/nbt.py', 'calibration/world.py')
    return {name: hash_file(ROOT / name) for name in names}


def alignment(local, heading, shift):
    return {p: tuple(x + delta for x, delta in zip(_yaw(p, heading), shift, strict=True)) for p in local}


def collect(predictions, reference):
    result = {}
    for family, measures in (('geometry', geometry_score(predictions, reference.truth, reference.optional)),
                             ('material', score(predictions, reference.truth, reference.optional, set(predictions)))):
        for subset in ('strict', 'facade'):
            result[f'{family}_{subset}'] = {k: measures[subset][k] for k in FIELDS}
    return result


def grade(local, reference, coordinates):
    if coordinates is not None:
        return collect({q: local[p] for p, q in coordinates.items()}, reference)
    # An unresolved nonempty export receives no coordinate or ID credit.
    result = collect({}, reference)
    for entry in result.values():
        entry['extras'] = len(local)
    return result


def export(path, blocks):
    temporary = path.with_suffix('.part')
    write_pack(temporary, {p: block.removeprefix('minecraft:') for p, block in blocks.items()}, JAVA_26_3_FORMAT)
    if decoded_pack(temporary) != blocks:
        raise ValueError(f'Changed coordinate or material after export: {temporary}')
    temporary.replace(path)
    return hash_file(path)


def run(job):
    release, out, frame, row, base, source_code = job
    marker = frame['id']
    files = frame['reference_sha256']
    directory = Path(frame['frame'])
    arm = row['arms'][base]
    if row['source_sha256'] != frame['rgb_sha256'] or row['reference_sha256'] != files:
        raise ValueError(f'Fixture does not match export receipt: {marker}')
    if not frame['framing']['accepted'] or frame['palette'] == 'orchid' or frame['dataset'] == 'decorative-builds-v1':
        raise ValueError(f'Unexpected fixture admission: {marker}')
    if hash_file(Path(frame['variants']['sky']['image'])) != frame['rgb_sha256']:
        raise ValueError(f'Changed source image: {marker}')
    for filename, expected in files.items():
        if hash_file(directory / filename) != expected:
            raise ValueError(f'Changed visible reference: {marker}/{filename}')
    with ZipFile(release / 'exports.zip') as archive:
        payload = archive.read(f'{marker}/{base}.zip')
    if digest(payload) != arm['pack_sha256']:
        raise ValueError(f'Changed archived export: {marker}')

    fingerprint = dict(stage_contract=1, source_sha256=frame['rgb_sha256'], reference_sha256=files,
                       base_pack_sha256=arm['pack_sha256'], base_chart_sha256=arm['chart_sha256'],
                       base_score=arm['score'], unresolved=arm['nonempty_unresolved'], code_sha256=source_code)
    key = digest(json.dumps(fingerprint, sort_keys=True, separators=(',', ':')).encode())
    folder = out / 'cache' / key
    receipt = folder / 'result.json'
    if receipt.is_file():
        saved = read(receipt)
        if saved['id'] == marker and saved['inputs'] == fingerprint and all(
                (folder / f'{k}.zip').is_file() and hash_file(folder / f'{k}.zip') == saved['results'][k]['pack_sha256']
                for k in ('exact-material', 'exact-export')):
            return saved, True

    labels = read(directory / 'labels.json')
    truth = {tuple(block['xyz_world']): block['block_id'] for block in labels['blocks']}
    with np.load(directory / 'labels.npz', allow_pickle=False) as channels:
        ids = channels['face_id']
        present = set(np.unique(ids)) - {0}
        face_owners = {tuple(face['xyz_world']) for face in labels['faces'] if face['id'] in present}
        if face_owners != truth.keys() or len(truth) != frame['visible_blocks']:
            raise ValueError(f'Visible block set is not exactly supported by native frontmost faces: {marker}')
        optional = horizontal_slivers(labels, ids)
    reference = FrameReference(truth, optional, read(directory / 'camera.json'), {'visible': set(truth)}, files)
    incumbent = decoded_pack(BytesIO(payload))
    existing_heading = arm['score']['yaw_quarters']
    existing_shift = arm['score']['translation']
    matched = None if arm['nonempty_unresolved'] else alignment(incumbent, existing_heading, existing_shift)
    baseline = grade(incumbent, reference, matched)
    if matched is not None:
        for family, key_group in (('geometry', 'geometry'), ('material', 'scores')):
            for kind in ('strict', 'facade'):
                if baseline[f'{family}_{kind}'] != {k: arm['score'][key_group][kind][k] for k in FIELDS}:
                    raise ValueError(f'Archived pack score did not reproduce: {marker}/{family}/{kind}')

    folder.mkdir(parents=True, exist_ok=True)
    heading_score, heading_aligned = baseline, (existing_heading, existing_shift)
    if arm['nonempty_unresolved']:
        all_yaws, _ = evaluate_relative_voxels(incumbent, reference, yaw_quarters=None, relaxed_yaw=True)
        ranked = all_yaws['relaxed_yaw']
        heading_aligned = ranked['yaw_quarters'], ranked['translation']
        heading_score = grade(incumbent, reference, alignment(incumbent, *heading_aligned))
    results = {'baseline': dict(scores=baseline, pack_sha256=arm['pack_sha256']),
               'heading-chooser': dict(scores=heading_score, pack_sha256=arm['pack_sha256'],
                                       selected_heading=heading_aligned[0], translation=heading_aligned[1],
                                       offered=[0, 1, 2, 3] if matched is None else [],
                                       intervention_applied=matched is None)}
    rebound = incumbent.copy()
    if matched is not None:
        rebound.update({p: truth[q] for p, q in matched.items() if q in truth})
    material_pack = folder / 'exact-material.zip'
    material_hash = export(material_pack, rebound)
    material_score = grade(rebound, reference, matched)
    if any(material_score[k] != baseline[k] for k in ('geometry_strict', 'geometry_facade')):
        raise ValueError(f'Material substitution changed the scored geometry: {marker}')
    results['exact-material'] = dict(scores=material_score, pack_sha256=material_hash,
                                     substituted=sum(rebound[p] != incumbent[p] for p in incumbent),
                                     heading=existing_heading if matched is not None else None,
                                     translation=existing_shift if matched is not None else None)

    # Use one local origin for the complete set of frontmost-face owners.
    origin = tuple(min(p[axis] for p in truth) for axis in range(3))
    visible = {tuple(x - base for x, base in zip(p, origin, strict=True)): material for p, material in truth.items()}
    exact_hash = export(folder / 'exact-export.zip', visible)
    exact = grade(visible, reference, alignment(visible, 0, origin))
    if any(value['missed'] or value['extras'] for value in exact.values()):
        raise ValueError(f'The exact visible build failed the writer check: {marker}')
    results['exact-export'] = dict(scores=exact, pack_sha256=exact_hash, visible_origin=origin)
    payload = dict(id=marker, split=frame['split'], cache_key=key, inputs=fingerprint, results=results)
    write_json(receipt, payload)
    return payload, False


def pooled(rows):
    outcome = {}
    for name in ('baseline', *PASSES):
        values = {}
        for measure in ('geometry_strict', 'geometry_facade', 'material_strict', 'material_facade'):
            counts = Counter()
            for row in rows:
                counts.update(row['results'][name]['scores'][measure])
            n = counts['correct'] * 2
            values[measure] = dict(counts, f1=100 * n / (sum(counts.values()) + counts['correct']))
        outcome[name] = values
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', default='current', help='Arm key in the saved conditions and export archive')
    parser.add_argument('--source-root', type=Path, help='Restored V1 release root with data/<split>/frames')
    parser.add_argument('--split', choices=['train', 'validation', 'test', 'regression'])
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    conditions = args.release / 'conditions.json.gz'
    rows = {record['id']: record for record in read(conditions)}
    manifest = args.release / 'fixtures.json.gz'
    fixtures = [frame for frame in read(manifest)['cases'] if args.split is None or frame['split'] == args.split]
    if args.source_root:
        for frame in fixtures:
            directory = args.source_root / 'data' / frame['split'] / 'frames' / frame['id']
            frame['frame'] = str(directory)
            frame['variants']['sky']['image'] = str(directory / 'rgb.png')
    if not fixtures:
        parser.error('The saved release has no matching fixtures.')
    source = dependency_hashes()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        result = list(pool.map(run, [(args.release, args.out, frame, rows[frame['id']], args.base, source) for frame in fixtures]))
    records = [value for value, _ in result]
    write_json(args.out / 'controls.json', dict(base=args.base, views=len(records), split=args.split,
               comparison_sha256=hash_file(conditions), fixtures_sha256=hash_file(manifest), stages=PASSES,
               pooled=pooled(records), cases=records))
    with ZipFile(args.out / 'perfect-packs.zip', 'w', compression=ZIP_DEFLATED) as archive:
        for row in records:
            for key in ('exact-material', 'exact-export'):
                source_pack = args.out / 'cache' / row['cache_key'] / f'{key}.zip'
                entry = ZipInfo(f"{row['id']}/{key}.zip")
                archive.writestr(entry, source_pack.read_bytes(), compress_type=ZIP_DEFLATED)
    with ZipFile(args.out / 'perfect-packs.zip') as archive:
        for row in records:
            for key in ('exact-material', 'exact-export'):
                if digest(archive.read(f"{row['id']}/{key}.zip")) != row['results'][key]['pack_sha256']:
                    raise ValueError(f'Archive checksum mismatch: {row["id"]}/{key}')
    print(f'{len(records)} verified views, {sum(reused for _, reused in result)} reused controls')


if __name__ == '__main__':
    main()
