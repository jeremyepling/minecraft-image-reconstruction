"""Replay the four known-mask diagnostics and verify them against plank-audit.json."""
import argparse
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))
from benchmark import current_placement_viewer as viewer
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.pavement_opening_eval import priority
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, read
from benchmark.thin_face_placement_run import ref_inputs
from benchmark.visible_error_budget import projection
from engine import smooth_extent_placement, visible_placement
from engine.surface_perimeter import with_outline


def replay(job, packed, expected):
    key = job[1]['id']
    assert job[1]['files']['rgb.png'] == expected['rgb_sha256']
    image, parts, counts, camera = viewer.suite.ordinary_inputs(*job)
    truth, occupied, transform, _ = ref_inputs(job)
    _, _, _, _, pose, labels, ids, *_ = viewer.sheet.stages.read_view(viewer.suite.CONTROL, *job)
    prior = registered(packed, transform)
    optional = horizontal_slivers(labels, ids)
    base_scores = placement_scores(prior, prior, dict.fromkeys(truth), occupied)
    owners, _, _, coords = projection(camera, pose, sorted(packed), base_scores['translation'], ids.shape)
    blocks = {row['id']: row['block_id'] for row in labels['blocks']}
    materials = {face['id']: blocks[face['block']] for face in labels['faces']}
    materials[0] = 'background'
    extra_material = Counter()
    for i, xyz in enumerate(coords):
        if tuple(xyz) not in truth:
            votes = Counter(materials[f] for f in ids[owners == i].tolist())
            extra_material[votes.most_common(1)[0][0] if votes else 'unseen'] += 1
    assert dict(extra_material) == expected['extra_pixels_dominate']

    horizon = {'mask-6', 'mask-13'}
    strip = {'planarity-new-11-0'}
    omit = {'outer': horizon, 'near': strip, 'all3': horizon | strip, 'remove': {'underside-0'}}
    for trial, score in expected['tests'].items():
        excluded = omit.get(trial, set())
        surfaces = [p for p in parts if p['id'] not in excluded]
        dimensions = dict(counts, surfaces=[p for p in counts['surfaces'] if p['id'] not in excluded])
        delta = {'up': 1, 'down': -1}.get(trial, 0)
        estimate = visible_placement._estimate_plane_cells

        def shifted(image, parts, *args, **kwargs):
            reconcile = kwargs.get('reconcile')

            def change_height(selections):
                if reconcile is not None:
                    selections = reconcile(selections)
                return [(i, axis, height + (delta if parts[i]['id'] == 'underside-0' else 0))
                        for i, axis, height in selections]

            kwargs['reconcile'] = change_height
            return estimate(image, parts, *args, **kwargs)

        with patch.object(visible_placement, '_estimate_plane_cells', shifted):
            cells, _ = visible_placement.place(image, '26.3', surfaces, counts=dimensions,
                                               camera=camera, atlas_path=viewer.suite.ATLAS)
            if 'shopfront' in key:
                added, _ = smooth_extent_placement.recover(image, '26.3',
                    [with_outline(p) for p in surfaces], cells, camera, viewer.suite.ATLAS)
                cells = cells | added
        if trial == 'control':
            assert cells == set(packed), key
        revised = registered(cells, transform)
        stage = placement_scores(revised, prior, dict.fromkeys(truth), occupied)
        aligned = viewer.suite.framed.spacing.shift(revised, stage['translation'])
        result = {k: stage['relative_geometry'][k] for k in score['strict']}
        assert result == score['strict'] and priority(aligned, truth, optional) == score['facade'], (key, trial)
        print(key, trial, result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=['workshop', 'shopfront', 'galleries', 'portico'])
    args = parser.parse_args()
    audit = json.loads(Path(__file__).with_name('plank-audit.json').read_text())
    root = viewer.ROOT
    checked(root/'counted-ground-corner-01/cases.json', audit['release_cases_sha256'])
    _, history, packs = viewer.read_releases([viewer.RELEASE, root/'smooth-local-placement-01',
        root/'smooth-joint-placement-01', root/'window-wall-01', root/'counted-ground-corner-01'])
    for job in viewer.suite.jobs(viewer.suite.framed.EVALUATION):
        key = job[1]['id']
        if key not in audit['focus'] or (args.layout and '-'+args.layout+'-' not in key):
            continue
        expected = audit['focus'][key]
        if key in packs:
            assert history[key][-1]['pack_sha256'] == expected['pack_sha256']
            packed = decoded_pack(BytesIO(packs[key]))
        else:
            packed, digest, _, _ = viewer.current_pack(job)
            assert digest == expected['pack_sha256']
        replay(job, packed, expected)


if __name__ == '__main__':
    main()
