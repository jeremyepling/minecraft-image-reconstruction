"""Compare the same repeat detector on masked native pixels and saved rectified samples."""
from collections import Counter

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, read, save
from benchmark.relative_face_round import SHARED
from engine.masked_count import _options, _select


def measure(path):
    rgba = np.asarray(Image.open(path).convert('RGBA'))
    sample = dict(rgb=rgba[:, :, :3], mask=rgba[:, :, 3] > 0)
    options = [_options(sample, axis)[0] for axis in (0, 1)]
    # Both controls omit the plane-aspect prior so only the input raster changes.
    return _select(options, None), [max(axis, key=lambda s: s['agreement'], default=None) for axis in options]


def main():
    result = {}
    availability = Counter()
    for case in read(SHARED / 'fixtures.json')['cases']:
        ident = case['id']
        folder = WORKTREES / 'floating-build-local-grid/runs/floating-build-grid-01/sky' / ident / 'source-handoff'
        counts = {r['id']: r for r in read(folder / 'counts.json')['surfaces']}
        result[ident] = {}
        for plane in read(folder / 'surfaces.json')['planes']:
            row = counts.get(plane['id'], {})
            crop = plane.get('source_crop', {}).get('path')
            if not crop:
                continue
            raw, raw_peak = measure(folder / crop)
            rectified, rectified_peak = measure(folder / row['sample']) if row.get('sample') else ([None]*2, [None]*2)
            result[ident][plane['id']] = dict(raw=raw, rectified=rectified, raw_peak=raw_peak,
                                               rectified_peak=rectified_peak)
            for key, values in [('raw', raw), ('rectified', rectified)]:
                availability[key] += sum(v is not None for v in values)
    save(ROOT / 'runs/native-crop-audit-01/pitch-controls.json', result)
    print(availability)
    print(next(iter(result.values()))['mask-3'])


if __name__ == '__main__':
    main()
