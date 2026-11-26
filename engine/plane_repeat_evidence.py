"""Rank plausible depths by comparing block-sized translations of visible textures."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from engine.plane_offset_features import NAMES, features, ranked_choice
from engine.visible_raster import SAMPLE_STRIDE


REPEAT_FEATURES = (*NAMES, 'repeat_high', 'repeat_low', 'repeat_atlas', 'repeat_edge', 'repeat_boundary')
WEIGHTS = Path(__file__).with_name('repeat_plane_weights.json')


@lru_cache(maxsize=2)
def _load_weights(path, fingerprint):
    model = json.loads(path.read_bytes())
    weights = np.asarray(model['weights'], dtype=float)
    if (model['schema'] != 1 or model['version'] != '26.3' or tuple(model['features']) != REPEAT_FEATURES
            or weights.shape != (len(REPEAT_FEATURES),) or not np.isfinite(weights).all()):
        raise ValueError('Unsupported repeat plane model')
    return weights


def frozen_repeat_weights():
    """Recheck the fingerprint so an asset repaired during the process can load."""
    try:
        stat = WEIGHTS.stat()
        return _load_weights(WEIGHTS, (stat.st_mtime_ns, stat.st_size))
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def near_levels(eye_coordinate, baseline):
    """Cover short planes missed by a distant accidental floor contact."""
    side = 1 if baseline > eye_coordinate else -1
    return [round(eye_coordinate) + side*i for i in range(1, 25)]


def feature_rows(rows, baseline):
    ordinary = features(rows,baseline)
    hi = np.array([r['repeat_high'] for r in rows])
    lo = np.array([r['repeat_low'] for r in rows])
    return np.column_stack((ordinary, hi, lo, hi*ordinary[:,0], hi*ordinary[:,4], lo*ordinary[:,5]))


def scored_rows(mask, P, eye, rays, axis, baseline, rgb, atlas, scene, *, extra_levels=()):
    from engine.atlas_plane_depth import options
    levels = set(near_levels(eye[axis], baseline)).union(extra_levels)
    rows = options(mask, P, eye, rays, axis, baseline, rgb, atlas, scene, radius=18, extra_levels=levels)
    repeats = repeat_scores(rgb, mask, P, eye, axis, [r['level'] for r in rows])
    for row, (hi, lo) in zip(rows, repeats, strict=True):
        row.update(repeat_high=hi, repeat_low=lo)
    return rows


def select_repeat(rows, baseline, stock_weights, repeat_weights, *, guarded=False):
    """Select one of the observed mask's candidate planes, with optional corroboration."""
    if not rows:
        return baseline
    if repeat_weights is not None:
        scores = feature_rows(rows, baseline) @ repeat_weights
        best = rows[int(np.argmax(scores))]
        strong_tile = best['atlas'] >= .70 and best['distinct_votes'] >= 2
        strong_repeat = best['repeat_high'] >= .24 and best['edge'] >= .12
        if not guarded or strong_tile or strong_repeat:
            return best['level']
    narrow = [row for row in rows if min(abs(row['level']-baseline), row['scene_delta']) <= 6]
    return ranked_choice(narrow, baseline, stock_weights)


def place(image, version, parts, counts, camera, *, guarded=False, atlas_path=None):
    from engine.atlas_plane_depth import _place
    weights = frozen_repeat_weights()
    def choose(part, mask, P, eye, rays, axis, baseline, rgb, atlas, scene, stock_weights):
        rows = scored_rows(mask, P, eye, rays, axis, baseline, rgb, atlas, scene)
        return select_repeat(rows, baseline, stock_weights, weights, guarded=guarded)
    return _place(image, version, parts, counts, camera, atlas_path, remove_occluded=True,
                  choose_plane=choose if weights is not None else None)


def guarded(image,version,parts,counts,camera):
    return place(image,version,parts,counts,camera,guarded=True)


def repeat_scores(rgb, mask, P, eye, axis, levels):
    from engine.atlas_plane_depth import _sample
    yy, xx = np.unravel_index(_sample(np.flatnonzero(mask), 600), mask.shape)
    # Native texel centers are sampled inside the observed surface.
    x = SAMPLE_STRIDE * xx + SAMPLE_STRIDE // 2
    y = SAMPLE_STRIDE * yy + SAMPLE_STRIDE // 2
    valid = (x < rgb.shape[1]) & (y < rgb.shape[0])
    x, y = x[valid], y[valid]
    if len(x) < 12:
        return [(0., 0.) for _ in levels]
    pixel = np.stack((x+.5, y+.5, np.ones(len(y))), axis=1)
    rays = pixel @ np.linalg.inv(P[:,:3]).T
    h,w = np.minimum(rgb.shape[:2], np.array(mask.shape)*SAMPLE_STRIDE)
    ref = rgb[y,x].astype(float)
    free = [j for j in range(3) if j != axis]
    output = []
    for level in levels:
        depth = np.divide(level-eye[axis], rays[:,axis],
                          out=np.zeros(len(rays)), where=abs(rays[:,axis])>1e-8)
        screen = depth[:,None]*pixel
        directions = []
        for d in free:
            # At a correct block pitch a one-block shift repeats more accurately
            # than either of two nearby noninteger displacements.
            hits, diffs = [], []
            for shift in (1., .8125, 1.1875):
                other = screen + shift*P[None,:,d]
                xy = np.floor(other[:,:2]/np.maximum(.1,other[:,2:])).astype(int)
                a,b = xy.T
                inside = (other[:,2]>1.) & (depth>1.) & (a>=0)&(a<w)&(b>=0)&(b<h)
                ix = np.flatnonzero(inside)
                inside[ix] &= mask[b[ix]//SAMPLE_STRIDE, a[ix]//SAMPLE_STRIDE]
                hits.append(inside)
                diffs.append(np.abs(rgb[b.clip(0,h-1),a.clip(0,w-1)].astype(float)-ref).mean(1))
            valid = np.logical_and.reduce(hits)
            if valid.sum() < 12:
                continue
            exact, offa, offb = [v[valid] for v in diffs]
            off = .5*(offa+offb)
            # An untextured segment supplies no evidence in either direction.
            gain = (off-exact) / (2.+off+exact)
            directions.append(float(gain.mean()) * min(1.,valid.sum()/48))
        output.append((max(directions,default=0.),min(directions,default=0.)))
    return output
