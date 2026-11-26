"""Test distinct material boundaries as RGB evidence for a one-block wall interval."""
from functools import lru_cache

import numpy as np

from benchmark.wall_texture_camera import (_abstain, _ground, _line_data, _optics, _result)
from engine.atlas_count import load_atlas, prepared_atlas_paths


@lru_cache(maxsize=2)
def _patterns(version):
    paths = prepared_atlas_paths(version)
    if not paths:
        return None
    at = load_atlas(paths[0], version=version)
    gray = at.tiles.mean(3).reshape(-1, 8, 2, 8, 2).mean((2, 4))
    tiles = np.stack([gray, gray[:, :, ::-1]], axis=1).reshape(len(gray), 2, -1)
    tiles -= tiles.mean(2, keepdims=True)
    tiles /= np.maximum(1., np.linalg.norm(tiles, axis=2, keepdims=True))
    return at, tiles.astype("float32")


def _match(crops, atlas):
    at, templates = atlas
    means = crops.mean((1, 2))
    gain = np.clip((means @ at.means.T) / np.maximum(1, (at.means**2).sum(1)), .32, 1.12)
    color = np.abs(gain[:, :, None]*at.means[None]-means[:, None]).mean(2)
    shortlist = np.argpartition(color, 23, axis=1)[:, :24]
    gray = crops.mean(3).reshape(len(crops), -1)
    gray -= gray.mean(1, keepdims=True)
    gray /= np.maximum(3, np.linalg.norm(gray, axis=1, keepdims=True))
    corr = np.einsum("nkod,nd->nko", templates[shortlist], gray, optimize=False).max(2)
    cost = np.take_along_axis(color, shortlist, axis=1)
    quality = corr-.023*cost
    idx = quality.argmax(1)
    return shortlist[np.arange(len(crops)), idx], quality[np.arange(len(crops)), idx], means


def _sample(rgb, mask, M, anchor, heights, atlas):
    import cv2

    # Each quad is an independently projected visible face. X indexes the wall
    # direction, Y points up from its measured contact with the floor.
    x = np.arange(-4, 4)
    y = np.arange(7)
    ix, iy = np.meshgrid(x, y)
    uv = (np.arange(8)+.5)/8
    u, v = np.meshgrid(uv, 1-uv)
    xyz = np.zeros((len(y), len(x), 8, 8, 3))
    xyz[..., anchor[1]] = ix[:, :, None, None]+u
    xyz[..., 1] = iy[:, :, None, None]+v
    relative = xyz.reshape(-1, 3) @ M.T
    offset = M @ anchor[3]
    out = []
    for height in heights:
        screen = relative+height*offset
        xy = (screen[:, :2]/screen[:, 2:3]).astype("float32").reshape(-1, 64, 2)
        xx, yy = xy[:, :, 0]-.5, xy[:, :, 1]-.5
        valid = cv2.remap(mask, xx, yy, cv2.INTER_NEAREST).mean(1)>.96
        width = np.linalg.norm(xy[:, 7]-xy[:, 0], axis=1)
        height_px = np.linalg.norm(xy[:, 56]-xy[:, 0], axis=1)
        valid &= (np.minimum(width,height_px)>8) & (screen[:, 2].reshape(-1, 64).min(1)>0)
        if valid.sum()<2:
            continue
        crops = cv2.remap(rgb, xx[valid], yy[valid], cv2.INTER_LINEAR)
        ids, q, colors = _match(crops.reshape(-1, 8, 8, 3), atlas)
        labels = np.full(ix.size, -1)
        scores = np.zeros(ix.size)
        means = np.zeros((ix.size, 3))
        labels[valid], scores[valid], means[valid] = ids, q, colors
        labels, scores, means = labels.reshape(ix.shape), scores.reshape(ix.shape), means.reshape(*ix.shape, 3)
        # Distinct labels plus measured color change between neighboring cells;
        # either cell may occur exactly once in the entire photo.
        pairs = []
        for axis in (0, 1):
            p = (slice(None, -1), slice(None)) if axis==0 else (slice(None), slice(None, -1))
            n = (slice(1, None), slice(None)) if axis==0 else (slice(None), slice(1, None))
            strength = np.minimum(scores[p], scores[n])
            different = (labels[p]!=labels[n]) & (abs(means[p]-means[n]).mean(2)>8)
            for i, j in zip(*np.nonzero(different & (strength>.39))):
                k, l = (i+1,j) if axis==0 else (i,j+1)
                pairs.append((float(strength[i,j]), [int(j+x[0]),int(i)], [int(l+x[0]),int(k)],
                              atlas[0].labels[labels[i,j]], atlas[0].labels[labels[k,l]]))
        if pairs:
            pairs.sort(reverse=True)
            out.append((pairs[0][0], height, anchor, pairs[:4]))
    return out


def predict(image, version="26.3"):
    atlas = _patterns(version)
    if atlas is None:
        return _abstain("atlas_unavailable")
    data = _line_data(image)
    if data is None:
        return _abstain("missing_lines")
    rgb, mask, seg, horizon = data
    M = _optics(seg, horizon, image.size)
    if M is None:
        return _abstain("missing_manhattan_pencils")
    anchors = _ground(rgb, mask, seg, M)
    if not anchors:
        return _abstain("no_ground_contact")
    accepted, proposals = [], []
    for a in anchors:
        if any(a[1]==other[1] and np.linalg.norm(a[2]-other[2])<6 for other in accepted):
            continue
        accepted.append(a)
        proposals.extend(_sample(rgb, mask, M, a, np.geomspace(.8, 21, 68), atlas))
    if not proposals:
        return _abstain("no_adjacent_atlas_pair")
    strength, height, anchor, pairs = max(proposals, key=lambda item: item[0])
    refined = _sample(rgb, mask, M, anchor, height*np.linspace(.974, 1.026, 11), atlas)
    if refined:
        strength, height, anchor, pairs = max(refined, key=lambda item: item[0])
    if strength < .64:
        return _abstain("weak_adjacent_atlas_pair", best=strength, pairs=pairs)
    return _result(M, anchor, height, dict(height=height, transition_score=strength, transitions=pairs,
                   repeat_votes=0, anchor=anchor[2].tolist(), axis=anchor[1]))

