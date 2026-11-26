"""Score a visible placement export on one saved native capture and measured surface build."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.camera_round import digest, read, save
from benchmark.mask_blocks import _score, _write_read, decoded_pack, registered
from benchmark.supported_placement_round import projected_shape
from engine.visible_placement import place
from runtime.mode import image_only
from training.visible_labels import visible_labels


def by_position(rows):
    indexed = {tuple(row[k] for k in ('x', 'y', 'z')): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate block coordinate in capture records')
    return indexed


@image_only()
def replay(capture, case, build, out, baseline=None):
    if case != Path(case).name or case in ('', '.', '..'):
        raise ValueError('Expected a capture ID')
    rgb = capture/'captures'/f'{case}.png'
    manifest = read(build/'build-manifest.json')
    rgb_sha = digest(rgb)
    if manifest['image_sha256'] != rgb_sha:
        raise ValueError('Surface build belongs to a different screenshot')
    artifacts = manifest['artifacts']
    surfaces = build/artifacts['surfaces']
    source = read(surfaces)
    parts = source['planes']
    hashes = dict(rgb=rgb_sha, surfaces=digest(surfaces), build_manifest=digest(build/'build-manifest.json'))
    counts = None
    if 'counts' in artifacts:
        path = build/artifacts['counts']
        counts = read(path)
        if counts['version'] != manifest['version'] or [r['id'] for r in counts['surfaces']] != [r['id'] for r in parts]:
            raise ValueError('Count rows do not match the surface build')
        hashes['counts'] = digest(path)
    with Image.open(rgb) as frame:
        if list(frame.size) != source['image_size']:
            raise ValueError('Surface dimensions do not match the screenshot')
        cells, status = place(frame.convert('RGB'), manifest['version'], parts, counts=counts)
    out.mkdir(parents=True, exist_ok=False)
    exported = _write_read(out/'reconstruction.zip', cells)

    # Camera pose, staged coordinates, and server readbacks enter only after export.
    views_path, layout_path = capture/'views.json', capture/'layout.json'
    views = [v for v in read(views_path) if v['id'] == case]
    if len(views) != 1:
        raise ValueError('Expected exactly one matching view in capture recipe')
    layout = read(layout_path)
    pose_path = capture/'captures'/f'{case}.json'
    blocks_path = capture/'captures'/f'{case}-blocks.json'
    pose = read(pose_path)
    if (pose['id'] != case or pose['minecraft_version'] != manifest['version'] or
            layout['version'] != manifest['version'] or pose['image_size'] != source['image_size']):
        raise ValueError('Capture identity, version or image dimensions differ')
    expected = by_position([b for b in layout['blocks'] if b['page'] == views[0]['page']])
    actual = by_position(read(blocks_path))
    if actual.keys() != expected.keys() or any(actual[p]['block_id'] != expected[p]['block_id'] for p in actual):
        raise ValueError('Native readback differs from the staged geometry')
    states = {p:r for p,r in actual.items() if expected[p].get('role') != 'floor_probe'}
    labels, ids, _ = visible_labels(pose, states, (0,0,0))
    truth = {tuple(b['xyz_world']) for b in labels['blocks']}
    mask = ids[2::4,2::4] > 0
    hashes.update(pose=digest(pose_path), readback=digest(blocks_path), views=digest(views_path), layout=digest(layout_path))
    transform = None if status['P'] is None else registration(np.asarray(status['P']), pose)

    def align(coords):
        if coords and transform is None:
            raise ValueError('Cannot register nonempty output without a supported camera')
        return registered(coords, transform) if coords else set()

    world = align(exported)
    result = dict(case=case, inputs_sha256=hashes, camera=status['camera_status'],
                  pack_sha256=digest(out/'reconstruction.zip'), score=_score(world, truth),
                  source_view_shape=projected_shape(world, pose, mask))
    if baseline is not None:
        prior = align(set(decoded_pack(baseline)))
        added, removed = world-prior, prior-world
        result['baseline'] = dict(pack_sha256=digest(baseline), score=_score(prior, truth),
                                  source_view_shape=projected_shape(prior, pose, mask))
        result['changes'] = dict(correct_gained=sorted(added & truth), correct_lost=sorted(removed & truth),
                                 extras_added=sorted(added-truth), extras_removed=sorted(removed-truth))
    save(out/'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True, help='Fixture root with captures/, views.json and layout.json')
    parser.add_argument('--case', required=True)
    parser.add_argument('--build', type=Path, required=True, help='Surface build from the same screenshot')
    parser.add_argument('--out', type=Path, required=True, help='New directory for the replay pack and score')
    parser.add_argument('--baseline', type=Path, help='Earlier datapack for the same image and local camera grid')
    args = parser.parse_args()
    print(json.dumps(replay(**vars(args))['score']))
