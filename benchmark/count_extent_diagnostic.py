"""Compare two ground-truth descriptions of visible width and height, without running a predictor.

The reference raster labels the center of every four-by-four screenshot pixel square.
We map the outer corners of those squares into the known plane and round each span
to the nearest block. The comparison target counts every labeled block column and
row touched, including partially visible edge blocks. Occlusion, grazing views and
raster sampling can make these quantities differ even with a correct plane.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark.camera_round import read, save
from benchmark.count_metrics import reference_planes
from benchmark.normal_round import groups
from benchmark.surface_metrics import DOWNSAMPLE
from benchmark.surface_reference import recorded
from engine.geometry import UNIT, project
from runtime.mode import image_only


def picture(case):
    _, ref = recorded(case)
    planes = reference_planes(ref, read(case.path('labels'))['faces'])
    ids = ref['id_map']
    counts = Counter(eligible=0, equal=0, different=0, touching_frame=0, different_touching_frame=0)
    disagreements = []
    for pid, plane in planes.items():
        if not ref['patches'][pid]['eligible']:
            continue
        y, x = np.where(ids == pid)
        pixels = np.column_stack((x, y))
        corners = (pixels[:, None] + UNIT) * DOWNSAMPLE
        corners = np.minimum(corners.reshape(-1, 2), ref['image_size'])
        uv = project(np.linalg.inv(ref['charts'][pid]), corners)
        span = np.ptp(uv, axis=0)
        rounded = np.maximum(1, np.floor(span + .5).astype(int))
        extents = np.ptp(np.asarray(list(plane['faces'])), axis=0) + 1
        touches = bool((x == 0).any() or (x == ids.shape[1] - 1).any()
                       or (y == 0).any() or (y == ids.shape[0] - 1).any())
        equal = np.array_equal(rounded, extents)
        counts['eligible'] += 1
        counts['touching_frame'] += int(touches)
        counts['equal' if equal else 'different'] += 1
        if not equal:
            counts['different_touching_frame'] += int(touches)
            disagreements.append(dict(surface=pid, touched_block_extents=extents.tolist(),
                                      visible_span=span.tolist(), rounded=rounded.tolist(), touching_frame=touches))
    return dict(key=f'{case.floor}/{case.id}', counts=dict(counts), disagreements=disagreements)


def evaluate(cases, out):
    pictures = [picture(case) for case in cases]
    counts = Counter()
    for row in pictures:
        counts.update(row['counts'])
    result = dict(diagnostic='Reference raster vs touched-block extents', stride_px=DOWNSAMPLE,
                  protocol=__doc__.strip(), axis_order='Two world axes in X, Y, Z order excluding the plane normal axis',
                  all=dict(counts), pictures=pictures)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, result)
    return result


@image_only()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cohorts, _, _ = groups()
    result = evaluate([case for cases in cohorts.values() for case in cases], args.out)
    print(result['all'])


if __name__ == '__main__':
    main()
