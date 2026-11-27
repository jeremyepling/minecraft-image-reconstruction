"""Render 16-texel vanilla sprites through a camera on demand."""
import numpy as np


def footprint_uv(P, xy, cell, axis, *, samples=2):
    """Return wrapped UV indices for a uniform pixel quadrature on a cube face.

    xy are source pixel centers, including the half-pixel offset. Both free
    world axes follow engine.projected_appearance.Scene.uv ordering.
    """
    matrix = np.asarray(P, dtype=float)
    inverse = np.linalg.inv(matrix[:, :3])
    eye = -inverse @ matrix[:, 3]
    level = cell[axis] + int(eye[axis] > cell[axis] + 1)
    offsets = (np.arange(samples) + .5) / samples - .5
    offsets = np.stack(np.meshgrid(offsets, offsets), axis=-1).reshape(-1, 2)
    positions = np.asarray(xy, dtype=float)[:, None, :] + offsets
    rays = positions @ inverse[:, :2].T + inverse[:, 2]
    free = [d for d in range(3) if d != axis]
    coords = eye[free] + (level - eye[axis]) * rays[..., free] / rays[..., axis, None]
    uv = np.floor(coords * 16).astype(np.int64) % 16
    return uv[..., 0] + 16 * uv[..., 1]


def draw_templates(tiles, uv):
    """Return per-pixel quadrature means for gray/RGB tiles (T,16,16,...)."""
    tiles = np.asarray(tiles)
    flat = tiles.reshape(len(tiles), 256, *tiles.shape[3:])
    return flat[:, uv].mean(axis=2, dtype=np.float32)
