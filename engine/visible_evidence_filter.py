"""Evaluate the frozen visible-cell rejection tree on observed RGB and proposed blocks."""
import hashlib
import json
from pathlib import Path

import numpy as np

from engine.camera_atlas_floor import _atlas
from engine.surface_model import intrinsic
from engine.visible_raster import SAMPLE_STRIDE, observed_support


MODEL = Path(__file__).parent / 'data' / 'visible-cell-filter-26.3.json'
MODEL_DIGEST = '496d0c550c37982b7419ca671512e870b87e11810a215e25976612a20d2291cd'


def _model_paths():
    try:
        model = json.loads(MODEL.read_bytes())
        canonical = json.dumps(model, sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(canonical).hexdigest() == MODEL_DIGEST:
            return model['paths']
    except (OSError, ValueError, TypeError):
        pass
    return ()


def _texture_agreement(samples, supported, atlas):
    """Return the best normalized sprite correlation and its lead over the runner-up."""
    scores = np.zeros(len(samples))
    margin = np.zeros(len(samples))
    if atlas is None:
        return scores, margin
    for i in np.flatnonzero(supported.sum(axis=1) >= 96):
        keep = supported[i]
        colors = samples[i, keep]
        mean = colors.mean(axis=0)
        gain = np.clip((atlas.means @ mean) / np.maximum(1, (atlas.means**2).sum(1)), .3, 1.15)
        err = np.abs(atlas.means*gain[:, None] - mean).mean(1) + 2*np.abs(np.log(gain/.8))
        indices = np.argpartition(err, 24)[:24]
        templates = atlas.patterns[indices].reshape(-1, 256)[:, keep]
        templates = templates - templates.mean(1, keepdims=True)
        observed = colors.mean(1)
        observed -= observed.mean()
        if observed.std() < 1.:
            continue
        denom = np.linalg.norm(templates, axis=1)*np.linalg.norm(observed)
        value = (templates @ observed) / np.maximum(1., denom)
        by_texture = value.reshape(len(indices), 8).max(1)
        best = np.partition(by_texture, -2)[-2:]
        scores[i] = best[1]
        margin[i] = best[1] - best[0]
    return scores, margin


def _face_evidence(rgb, P, eye, xyz, axis, axis_mask, owner, atlas):
    h, w = rgb.shape[:2]
    positive = eye[axis] >= xyz[:, axis]+1
    level = xyz[:, axis] + positive
    facing = positive | (eye[axis] <= xyz[:, axis])
    u, v = np.meshgrid((np.arange(16)+.5)/16, (np.arange(16)+.5)/16)
    offsets = np.zeros((256, 3))
    offsets[:, [a for a in range(3) if a != axis]] = np.column_stack((u.ravel(), v.ravel()))
    n = len(xyz)
    support_count = np.zeros(n, dtype=int)
    art = np.zeros(n)
    margin = np.zeros(n)
    contrast = np.zeros(n)
    visible_samples = np.zeros(n)
    # Limit intermediate memory as the number of proposed blocks grows.
    for start in range(0, n, 256):
        sl = slice(start, min(start+256, n))
        world = xyz[sl, None, :] + offsets
        world[:, :, axis] = level[sl, None]
        q = world @ P[:, :3].T + P[:, 3]
        xy = q[:, :, :2] / np.maximum(q[:, :, 2, None], 1.e-6)
        pix = np.floor(xy)
        valid = facing[sl, None] & (q[:, :, 2] > 1.e-6) & (pix >= 0).all(2) & (pix < (w, h)).all(2)
        px, py = np.clip(pix, [0, 0], [w-1, h-1]).astype(int).transpose(2, 0, 1)
        sx, sy = px//SAMPLE_STRIDE, py//SAMPLE_STRIDE
        visible = valid & (owner[sy, sx] == np.arange(sl.start, sl.stop)[:, None])
        support = visible & axis_mask[sy, sx]
        support_count[sl] = support.sum(1)
        samples = rgb[py, px].astype(np.float32)
        art[sl], margin[sl] = _texture_agreement(samples, support, atlas)
        gray = samples.mean(2).reshape(-1, 16, 16)
        contrast[sl] = np.abs(np.diff(gray, axis=1)).mean((1, 2)) + np.abs(np.diff(gray, axis=2)).mean((1, 2))
        a, b = xy[:, 15]-xy[:, 0], xy[:, 240]-xy[:, 0]
        footprint = np.abs(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0])/SAMPLE_STRIDE**2
        visible_samples[sl] = footprint * visible.mean(1)

    plane_atlas = np.zeros(n)
    plane_median = np.zeros(n)
    for height in np.unique(level):
        same = level == height
        usable = same & (support_count >= 96)
        if usable.any():
            plane_atlas[same] = np.mean(art[usable] > .65)
            plane_median[same] = np.median(art[usable])
    fields = dict(plane_atlas=plane_atlas, plane_median_atlas=plane_median, edge_contrast=contrast,
                  atlas_separation=margin, visible_samples=visible_samples)
    return support_count, fields


def filter_cells(image, parts, camera, cells, atlas_path=None):
    """Keep source-visible proposals accepted by the tree, or use the input set if unavailable."""
    if not cells or not parts or not camera or camera.get('P') is None:
        return cells
    paths = _model_paths()
    if not paths:
        return cells
    try:
        P = np.asarray(camera['P'], dtype=float)
    except (TypeError,ValueError):
        return cells
    if P.shape != (3, 4) or not np.isfinite(P).all() or np.linalg.norm(P[2, :3]) < 1.e-8:
        return cells
    P = P / np.linalg.norm(P[2, :3])
    focal = np.linalg.norm(P[0, :3] - image.width/2*P[2, :3])
    if focal < 1.e-6:
        return cells
    try:
        eye = np.linalg.solve(P[:, :3], -P[:, 3])
        rotation = np.linalg.solve(intrinsic(image.size, focal), P[:, :3])
    except np.linalg.LinAlgError:
        return cells
    shape = tuple((n+SAMPLE_STRIDE-1)//SAMPLE_STRIDE for n in image.size[::-1])
    assigned = np.zeros((3, *shape), dtype=bool)
    for part in parts:
        normal = part.get('normal_camera')
        if normal is None or not part.get('support'):
            continue
        normal=np.asarray(normal,dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1.e-8:
            continue
        axis = int(np.abs(rotation.T @ normal).argmax())
        assigned[axis] |= observed_support(part['support'], shape, image.size)
    if not assigned.any():
        return cells
    atlas = _atlas('26.3', atlas_path)
    if atlas is None:
        return cells
    from engine.visible_placement import _frontmost_cells
    points = sorted(cells)
    xyz = np.asarray(points, dtype=int)
    owner = _frontmost_cells(points, P, eye, shape)
    rgb = np.asarray(image.convert('RGB'))
    views = [_face_evidence(rgb, P, eye, xyz, i, assigned[i], owner, atlas) for i in range(3)]
    axis = np.argmax([n for n, _ in views], axis=0)
    rows = np.arange(len(points))
    fields = {k: np.stack([f[k] for _, f in views], axis=1)[rows, axis] for k in views[0][1]}
    fields['horizontal_face'] = axis == 1
    drop = np.zeros(len(points), dtype=bool)
    for path in paths:
        match = np.ones(len(points), dtype=bool)
        for key, op, value in path:
            match &= fields[key] <= value if op == 'le' else fields[key] > value
        drop |= match
    return {points[i] for i in np.flatnonzero(~drop)}
