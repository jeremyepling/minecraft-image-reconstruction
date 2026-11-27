"""Evaluate single-block deletions under the existing projected silhouette and axis loss."""
from io import BytesIO

import numpy as np

from benchmark import current_placement_viewer as viewer
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.thin_face_placement_run import ref_inputs
from benchmark.texture_camera_eval import read, save
from engine.projected_appearance import Scene


def favorable_deletions(scene):
    positive, clean = set(), set()
    for index, pixels in scene.owned.items():
        current = scene.loss(pixels, scene.owner[pixels], scene.axes[pixels])
        without = scene.loss(pixels, scene.back[pixels], scene.back_axis[pixels])
        gain = current - without
        if gain.sum() > 0:
            positive.add(scene.cells[index])
            if gain.sum() > 8 and not (gain < 0).any():
                clean.add(scene.cells[index])
    return positive, clean


def main():
    previous = {r['id']:r for r in read(viewer.ROOT/'window-wall-01/cases.json')}
    expected = {r['id']:r for r in read(viewer.ROOT/'facade-inset-round-01/combined.json')['rows']}
    _, history, packs = viewer.read_releases([viewer.RELEASE, viewer.ROOT/'smooth-local-placement-01',
                                               viewer.ROOT/'smooth-joint-placement-01', viewer.ROOT/'window-wall-01'])
    output = []
    for job in viewer.suite.jobs(viewer.suite.framed.EVALUATION):
        key = job[1]['id']
        if 'courtyard' not in key:
            continue
        base = previous[key]
        if key in packs:
            original = set(decoded_pack(BytesIO(packs[key])))
            digest = history[key][-1]['pack_sha256']
        else:
            original, digest, _, _ = viewer.current_pack(job)
        assert digest == base['previous_pack_sha256']
        image, parts, _, camera = viewer.latest_stages(job, expected[key])
        positive, clean = favorable_deletions(Scene(image, parts, camera, original))
        truth, _, transform, _ = ref_inputs(job)
        shift = np.array(base['result']['translation'])
        row = dict(id=key, source_hash=base['source_hash'], previous_pack_sha256=digest)
        for name, cells in [('positive',positive), ('clean',clean)]:
            coords = {tuple(shift+xyz) for xyz in registered(cells, transform)}
            row[name] = dict(correct_removed=len(coords & truth), extras_removed=len(coords-truth))
        output.append(row)
    save(viewer.ROOT/'courtyard-corner-01/removal-trial.json', output)


if __name__ == '__main__':
    main()
