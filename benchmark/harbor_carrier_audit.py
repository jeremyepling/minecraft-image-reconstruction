"""Audit native visible-face ownership of the actual carriers and decoded exports."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, frames, read, save
from benchmark.mask_blocks import decoded_pack
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.typed_structures import aligned_sites

REVIEW = ROOT / 'benchmark/reviews/harbor-placement-evidence-01'
CASES = ROOT / 'benchmark/reviews/camera-model-controls-01/independent-matched-cases.json.gz'
NEW = WORKTREES / 'camera-geocalib/runs/camera-geocalib-01/matched-control/builds'
BASE = WORKTREES / 'full-expanded-controls/runs/expanded-baseline-01'
REPEAT = WORKTREES / 'camera-horizon-floor/runs/camera-horizon-floor-01'


def build_dir(entry, case):
    key, dataset = entry['id'], entry['dataset']
    for folder in (NEW / dataset / key, REPEAT / key / 'repeat', BASE / dataset / key):
        pack = folder / 'reconstruction.zip'
        if pack.exists() and digest(pack) == case['pack_sha256']:
            return folder
    raise ValueError(f'{key}: no matching released pack')


def native_mask(part, size):
    sw, sh = part['support']['size']
    mask = np.zeros((sh, sw), bool)
    for y, x0, x1 in part['support']['runs']:
        mask[y, x0:x1] = True
    w, h = size
    return mask if (w, h) == (sw, sh) else mask[np.ix_(
        ((np.arange(h) + .5) * sh / h).astype(int),
        ((np.arange(w) + .5) * sw / w).astype(int))]


def native_face_planes(labels, transform):
    """Return the integer face level, expressed in the predicted camera's local XYZ grid."""
    inverse = np.linalg.inv(transform)
    result = {}
    for face in labels['faces']:
        corners = np.array(face['corners_world']) @ inverse[:3, :3].T + inverse[:3, 3]
        axis, = np.flatnonzero(np.ptp(corners, axis=0) == 0)
        result[face['id']] = (int(axis), int(corners[0, axis]))
    return result


def audit(entry, case, overlay=False):
    folder = build_dir(entry, case)
    frame = Path(entry['frame'])
    labels = read(frame / 'labels.json')
    with np.load(frame / 'labels.npz') as archive:
        ids = archive['face_id']
    size = ids.shape[::-1]
    assert digest(frame / 'rgb.png') == case['source_sha256']
    assert digest(frame / 'labels.npz') == case['reference_sha256']['labels.npz']
    transform = np.asarray(case['registration']['transform'])
    truth = {tuple(b['xyz_world']): b['block_id'] for b in labels['blocks']}
    optional = horizontal_slivers(labels, ids)
    raw = decoded_pack(folder / 'reconstruction.zip')
    pred = dict(zip(aligned_sites(list(raw), transform, case['translation']), raw.values(), strict=True))
    hits = truth.keys() & pred.keys()
    assert len(hits) == case['geometry']['strict']['correct']
    status = read(folder / 'build-manifest.json')['placement']
    # The placer can consume pre-inset masks absent from the public surface artifact.
    placement_input = read(BASE / entry['dataset'] / entry['id'] / 'handoffs/placement-input.json.gz')
    parts = placement_input['parts']
    offsets = {r['id']: r for r in status['plane_offsets']}
    counts = {r['id']: r for r in placement_input['counts']['surfaces']}
    planes = native_face_planes(labels, transform)
    native_count = np.bincount(ids.ravel())
    assert all(native_count[f['id']] == f['visible_pixels'] for f in labels['faces'])
    xyz = {f['id']: tuple(f['xyz_world']) for f in labels['faces']}
    evidence = np.zeros(ids.shape, np.uint8)
    surface_rows = []
    for part in parts:
        key = part['id']
        mask = native_mask(part, size)
        hist = np.bincount(ids[mask], minlength=len(native_count))
        plane_hist = Counter()
        for face in np.flatnonzero(hist[1:]) + 1:
            plane_hist[planes[face]] += int(hist[face])
        decision = offsets.get(key)
        actual = (decision['axis'], decision['level']) if decision else None
        primary = plane_hist.most_common(1)[0][0] if plane_hist else None
        grade = np.array([0] + [3 if planes[i] == actual else 2 if actual and planes[i][0] == actual[0] else 1
                                for i in range(1, len(native_count))], dtype=np.uint8)
        evidence[mask] = np.maximum(evidence[mask], grade[ids[mask]])
        support = int(hist[1:].sum())
        surface_rows.append(dict(id=key, selected=actual, initial=decision['baseline'] if decision else None,
            primary=primary, observed_pixels=support, spill_pixels=int(hist[0]), primary_pixels=plane_hist[primary],
            correct_plane_pixels=plane_hist[actual], counts=counts.get(key, {}).get('counts'),
            top_planes=[(list(p), n) for p, n in plane_hist.most_common(5)]))
    # One pixel is a deliberately permissive upper bound on receiving evidence at each stage.
    per_block = {p: 0 for p in truth}
    for value in range(1, 4):
        for face in np.unique(ids[evidence == value]):
            if face:
                per_block[xyz[face]] = value
    misses = sorted(truth.keys() - pred.keys())
    required = [p for p in misses if p not in optional]
    top = sorted(surface_rows, key=lambda r: -r['observed_pixels'])
    pixel_tallies = [int(np.count_nonzero((ids != 0) & (evidence == k))) for k in range(4)]
    if overlay:
        image = Image.open(frame / 'rgb.png').convert('RGB')
        source = np.array(image)
        colors = np.array([(246, 72, 70), (208, 93, 229), (247, 173, 42), (41, 211, 164)])
        active = ids != 0
        source[active] = .30*source[active] + .70*colors[evidence[active]]
        sheet = Image.new('RGB', (size[0], size[1]+56), 'white')
        sheet.paste(Image.fromarray(source))
        draw = ImageDraw.Draw(sheet)
        draw.text((15, size[1]+5), entry['id']+' | source-image overlay / native frontmost pixels', fill='black')
        draw.text((15, size[1]+26), 'Red: no mask   Purple: wrong axis or no carrier   Orange: correct axis, wrong level   Green: correct axis and level', fill='black')
        REVIEW.mkdir(parents=True, exist_ok=True)
        sheet.save(REVIEW / (entry['id']+'.jpg'), quality=91)
    return dict(id=entry['id'], source_sha256=case['source_sha256'], pack_sha256=case['pack_sha256'],
        native_reference_sha256=case['reference_sha256'], translation=case['translation'],
        geometry=case['geometry'], scores=case['scores'], surfaces=len(parts),
        grounded_removed=status.get('ground_contact_removed'), strict_chart_cells=status.get('strict_blocks'),
        counts_both=sum(None not in r['counts'] for r in counts.values()),
        per_native_pixel=pixel_tallies,
        per_missed_block=[sum(per_block[p] == k for p in misses) for k in range(4)],
        per_required_missed_block=[sum(per_block[p] == k for p in required) for k in range(4)],
        dominant_single_plane_pixels=sum(r['primary_pixels'] for r in top),
        material_substitutions=Counter(f'{truth[p]} -> {pred[p]}' for p in hits).most_common(8),
        carriers=top)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ids', nargs='+')
    parser.add_argument('--overlay', action='store_true')
    args = parser.parse_args()
    cases = {r['id']: r for r in read(CASES)}
    for entry in frames(SOURCE, ['visible-builds-v3']):
        if entry['id'] in args.ids:
            result = audit(entry, cases[entry['id']], args.overlay)
            save(REVIEW / (entry['id']+'.json'), result)
            print(entry['id'], result['per_native_pixel'], result['per_required_missed_block'], result['counts_both'],
                  [(r['id'],r['selected'],r['primary'],r['observed_pixels'],r['primary_pixels']) for r in result['carriers'][:3]], flush=True)


if __name__ == '__main__':
    main()
