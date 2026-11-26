"""Diagnostic controls for visible markings on real and composited block faces."""
import argparse
import json
from pathlib import Path

import numpy as np

from engine.atlas import digest
from engine.blocker_shape import ShapeMatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas', required=True, type=Path)
    parser.add_argument('--tiles', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    matcher = ShapeMatcher(args.atlas)
    results = []
    with np.load(args.tiles) as data:
        for name, patch in zip(data['labels'], data['tiles']):
            results.append(dict(kind='real rendered', truth=str(name), **matcher.rank_patch(patch)))
    _, tile, alpha, _ = next(item for item in matcher.glass if item[0]=='glass')
    alpha = alpha[:,:,None]
    backgrounds = {'dark':50, 'sky':[105,170,225], 'noise':np.random.default_rng(42).uniform(0,195,(16,16,3))}
    for setting,bg in backgrounds.items():
        patch = tile*alpha + bg*(1-alpha)
        results.append(dict(kind='synthetic clear glass', background=setting, truth='glass', **matcher.rank_patch(patch)))
    results.append(dict(kind='synthetic flat gray',truth='unspecified opaque', **matcher.rank_patch(np.full((16,16,3),125))))
    args.out.write_text(json.dumps(dict(results=results,tiles_sha256=digest(args.tiles)),indent=2))


if __name__=='__main__':
    main()
