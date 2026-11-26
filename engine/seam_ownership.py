"""Test whether a neighboring cube's side explains an alleged second front."""
from functools import lru_cache
import numpy as np

from engine.visible_raster import SAMPLE_STRIDE

V, U = (np.indices((16, 16)) + .5) / 16


def _face_samples(gray, projection, cell, axis, eye, owner, sample_face):
    """Rectify the vertical face with one texture row per world-height sample."""
    sign = 1 if eye[axis] >= cell[axis] + 1 else -1
    if cell[axis] <= eye[axis] <= cell[axis] + 1:
        return None
    hit = sample_face(projection, cell, axis, sign, owner.shape)
    if hit is None:
        return None
    bounds, mask, _, _, quad = hit
    quad = quad * SAMPLE_STRIDE
    height, width = gray.shape
    if (quad.min(axis=0) < 0).any() or (quad.max(axis=0) >= [width, height]).any():
        return None
    if np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1).min() < 5:
        return None
    xyz = np.broadcast_to(cell, (16, 16, 3)).astype(float).copy()
    xyz[:, :, axis] += int(sign > 0)
    xyz[:, :, 1] += V
    xyz[:, :, 2 - axis] += U
    q = xyz @ projection[:, :3].T + projection[:, 3]
    xy = (q[:, :, :2] / q[:, :, 2, None]).astype(int)
    return gray[xy[:, :, 1], xy[:, :, 0]], owner[bounds][mask]


def _correlation(a, b):
    # The two vertical sides can have opposite horizontal texture directions.
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1.e-8:
        return 0.
    return max(np.sum(a * b), np.sum(a * b[:, ::-1])) / denominator


def _turn_fits(front, side, second):
    if min(front.std(), side.std()) < 5 or not .60 * front.mean() <= side.mean() <= .98 * front.mean():
        return False
    side_match = _correlation(front, side)
    return side_match >= .75 and side_match >= _correlation(front, second) + .25


def textured_side_owners(image, projection, eye, cells, owner, accepted, strict, sample_face):
    """Identify neighboring cells whose projected side explains this alleged front."""
    gray = np.asarray(image.convert('L'), dtype=float)
    index = {p: i for i, p in enumerate(cells)}

    @lru_cache(None)
    def sample(p, axis):
        return _face_samples(gray, projection, p, axis, eye, owner, sample_face)

    possible = {}
    for extra in sorted(accepted - strict):
        for axis in (0, 2):
            donor = list(extra)
            donor[axis] += -1 if eye[axis] > extra[axis] + .5 else 1
            donor = tuple(donor)
            if donor not in accepted:
                continue
            front_axis = 2 - axis
            front = sample(donor, front_axis)
            side = sample(donor, axis)
            second = sample(extra, front_axis)
            if front is None or side is None or second is None:
                continue
            if np.mean(front[1] == index[donor]) < .6 or np.mean(second[1] == index[extra]) < .6:
                continue
            if np.mean((side[1] == index[donor]) | (side[1] == index[extra])) < .6:
                continue
            if _turn_fits(front[0], side[0], second[0]):
                possible.setdefault(extra, set()).add(donor)
    return possible


def resolve_side_owners(textured, grounded, strict):
    """Remove unambiguous duplicates once, keeping all cells that explain another face."""
    owners = set().union(*textured.values(), *grounded.values())
    duplicates = set()
    for cell in (textured.keys() | grounded.keys()) - strict - owners:
        explanations = textured.get(cell, set()) | grounded.get(cell, set())
        if len(explanations) == 1:
            duplicates.add(cell)
    return duplicates
