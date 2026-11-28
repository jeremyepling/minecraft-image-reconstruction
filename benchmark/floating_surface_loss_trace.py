"""Attribute changed scored positions to the source charts that owned the decoded cells."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from benchmark.floating_stage_report import hash_file, read
from benchmark.floating_visual_review import strict_delta
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_scoring import _yaw
from benchmark.typed_structures import aligned_sites


class TraceUnavailable(Exception):
    """The saved evidence cannot establish registered ownership."""


def _read_arm(case, label, folder, package):
    local_grid = (folder / 'case.json').is_file()
    receipt = read(folder / ('case.json' if local_grid else 'score.json'))
    score = receipt['score']
    metadata = score if local_grid else receipt
    pack = folder / 'reconstruction.zip'
    expected = case[label.lower()]
    if (receipt['id'] != case['id'] or metadata['source_sha256'] != case['source_sha256']
            or metadata['reference_sha256'] != case['reference_sha256']
            or metadata['pack_sha256'] != expected['pack_sha256']
            or hash_file(pack) != expected['pack_sha256']):
        raise ValueError(f'{label} score or pack differs from the case receipt.')
    local = decoded_pack(pack)
    owners = {}
    for surface in package['surfaces']:
        for cell in surface.get('exports', {}).get(label, {}).get('cells', []):
            xyz = tuple(cell['xyz'])
            if xyz in owners:
                raise ValueError(f'{label} has multiple export owners for {xyz}.')
            owners[xyz] = surface['id'], cell
    if {xyz: cell['block'] for xyz, (_, cell) in owners.items()} != local:
        raise ValueError(f'{label} surface ownership differs from the decoded pack.')
    if local_grid:
        transform = score['registration']['transform']
        if transform is None and local:
            raise TraceUnavailable(f'{label} export has no saved registration.')
        coordinates = aligned_sites(list(local), np.asarray(transform), score['translation'])
        saved = read(folder / 'aligned-placements.json.gz')
        saved_world = {tuple(item[:3]): item[3] for item in saved}
    else:
        fixed = score['fixed']
        if fixed is None and local:
            raise TraceUnavailable(f'{label} export has no fixed scored registration.')
        if fixed and any(name in expected and expected[name] != fixed[name]
                         for name in ('yaw_quarters', 'translation')):
            raise ValueError(f'{label} registration differs from the visual case receipt.')
        coordinates = [
            tuple(v + delta for v, delta in zip(_yaw(xyz, fixed['yaw_quarters']), fixed['translation'], strict=True))
            for xyz in local
        ]
        saved = read(folder / 'placements.json')
        if local != {tuple(xyz): block for xyz, block in saved['local']}:
            raise ValueError(f'{label} saved local cells differ from the decoded pack.')
        saved_world = {tuple(xyz): block for xyz, block in saved['aligned']}
    world = dict(zip(coordinates, local.values(), strict=True))
    if world != saved_world:
        raise ValueError(f'{label} decoded cells differ from the saved registered placements.')
    return world, dict(zip(coordinates, (owners[xyz] for xyz in local), strict=True))


def _visible_truth(case, frame):
    for name, expected in {'rgb.png': case['source_sha256'], **case['reference_sha256']}.items():
        if hash_file(frame / name) != expected:
            raise ValueError(f'Case source or reference differs: {name}')
    labels = read(frame / 'labels.json')
    blocks = {block['id']: block['block_id'] for block in labels['blocks']}
    return {tuple(face['xyz_world']): blocks[face['block']] for face in labels['faces'] if face['visible_pixels']}


def trace_case(case, package, frame_dir, baseline_folder, candidate_folder):
    """Bind one case's surface owners to its archived or local-grid score receipts.

    Per-surface markers record the writer's projected cell center and the scored world
    position. A dropped owner's loss does not establish a compatible attachment.
    """
    result = dict(schema=1, id=case['id'], sourceSha256=case['source_sha256'],
                  baselinePackSha256=case['baseline']['pack_sha256'],
                  candidatePackSha256=case['candidate']['pack_sha256'],
                  status='unavailable', reason=None, paired=None, surfaces={}, droppedLosses=None)
    if not package or package.get('schema') != 2:
        return dict(result, reason='Paired surface ownership was not supplied.')
    if (package['id'] != result['id'] or package['sourceSha256'] != result['sourceSha256']
            or any(package['provenance']['arms'][label]['packSha256'] != result[f'{label.lower()}PackSha256']
                   for label in ('Baseline', 'Candidate'))):
        raise ValueError('Surface evidence differs from the case source or packs.')
    try:
        truth = _visible_truth(case, frame_dir)
        before, old_owners = _read_arm(case, 'Baseline', baseline_folder, package)
        after, new_owners = _read_arm(case, 'Candidate', candidate_folder, package)
    except TraceUnavailable as error:
        return dict(result, reason=str(error))
    except FileNotFoundError as error:
        return dict(result, reason=f'Saved trace input is missing: {Path(error.filename).name}')
    gained, lost, paired = strict_delta(truth, before, after)
    if paired != case['paired']:
        raise ValueError('Decoded paired changes differ from the reviewed case.')
    removed = before.keys() - after.keys() - truth.keys()
    added = after.keys() - before.keys() - truth.keys()
    dropped = set((case.get('component') or {}).get('dropped', []))
    surfaces = {s['id']: dict(lost=0, gained=0, extrasAdded=0, extrasRemoved=0,
                              dropped=False, droppedLosses=0, charts={}, markers=[]) for s in package['surfaces']}
    for field, positions, owners in (('lost', lost, old_owners), ('gained', gained, new_owners),
                                     ('extrasRemoved', removed, old_owners), ('extrasAdded', added, new_owners)):
        for xyz in sorted(positions):
            surface_id, cell = owners[xyz]
            surface = surfaces[surface_id]
            chart = cell['chart']
            surface[field] += 1
            counts = surface['charts'].setdefault(chart, dict(lost=0, gained=0, extrasAdded=0, extrasRemoved=0))
            counts[field] += 1
            if field == 'lost' and re.sub(r'~p\d+$', '', chart) in dropped:
                surface['dropped'] = True
                surface['droppedLosses'] += 1
            center = [round(sum(values) / 4, 1) for values in zip(*cell['quad'], strict=True)]
            surface['markers'].append(dict(kind=field, chart=chart, source=center, xyz=xyz))
    return dict(result, status='verified', paired=paired, surfaces=surfaces,
                droppedLosses=sum(s['droppedLosses'] for s in surfaces.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue', type=Path, help='Visual queue receipts.json')
    parser.add_argument('case', help='Exact saved case ID')
    parser.add_argument('package', type=Path, help='Paired surfaces.json')
    parser.add_argument('frame', type=Path, help='Saved frame containing source and reference files')
    parser.add_argument('baseline', type=Path, help='Baseline folder containing its pack and score')
    parser.add_argument('candidate', type=Path, help='Candidate folder containing its pack and score')
    args = parser.parse_args()
    rows = [row for row in read(args.queue)['cases'] if row['id'] == args.case]
    if len(rows) != 1:
        parser.error('Expected one matching saved case.')
    print(json.dumps(trace_case(rows[0], read(args.package), args.frame, args.baseline, args.candidate), indent=2))


if __name__ == '__main__':
    main()
