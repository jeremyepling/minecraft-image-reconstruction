"""Render source-mask outlines and the visible face corners of a decoded placement export."""
from itertools import product

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine.pillar_seams import native_mask
from engine.visible_grid import _carriers
from engine.visible_placement import _charts, _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE, observed_support


def edge_overlay(size, parts, camera, cells):
    """Use the estimated camera and local export coordinates, before evaluation registration."""
    width, height = size
    outlines = np.zeros((height, width), bool)
    for part in parts:
        mask = native_mask(part, size)
        if mask.any():
            interior = np.asarray(Image.fromarray(mask).filter(ImageFilter.MinFilter(3)))
            outlines |= mask & ~interior
    surface = Image.fromarray(outlines.astype(np.uint8))
    surface.putpalette([0, 0, 0, 255, 162, 51])
    cells = sorted(cells)
    pixels = np.zeros((height, width), np.uint8)
    points = set()
    faces = 0
    if cells and camera.get('P') is not None:
        p = np.array(camera['P'], float)
        p /= np.linalg.norm(p[2, :3])
        eye = -np.linalg.solve(p[:, :3], p[:, 3])
        raster = p.copy()
        raster[:2] *= SAMPLE_STRIDE
        owner, axis = _frontmost_cells(cells, raster, eye, (height, width), face_axes=True)
        front = owner >= 0
        code = np.where(front, owner * 3 + axis, -1)
        boundaries = np.zeros_like(front)
        for dy, dx in ((1, 0), (0, 1)):
            first = np.s_[:height-dy, :width-dx]
            second = np.s_[dy:, dx:]
            changed = code[first] != code[second]
            boundaries[first] |= changed & front[first]
            boundaries[second] |= changed & front[second] & ~front[first]
        pixels[boundaries] = 1
        visible = np.unique(code[front])
        faces = len(visible)
        for value in visible:
            index, normal = divmod(int(value), 3)
            cell = cells[index]
            free = [a for a in range(3) if a != normal]
            for offsets in product((0, 1), repeat=2):
                vertex = np.array(cell)
                vertex[normal] += int(eye[normal] > cell[normal] + 1)
                vertex[free] += offsets
                uv = p[:, :3] @ vertex + p[:, 3]
                if uv[2] <= 0:
                    continue
                x, y = np.floor(uv[:2] / uv[2]).astype(int)
                if 0 <= x < width and 0 <= y < height:
                    if (code[max(0, y-2):y+3, max(0, x-2):x+3] == value).any():
                        points.add((int(x), int(y)))
    wire = Image.fromarray(pixels)
    wire.putpalette([0, 0, 0, 22, 217, 208, 208, 255, 247])
    draw = ImageDraw.Draw(wire)
    for x, y in points:
        draw.ellipse((x-1, y-1, x+1, y+1), fill=2)
    return surface, wire, faces


def corner_pair(image, parts, counts, camera, record):
    """Reproject the two whole counted faces on the tested and original chart planes."""
    anchors = record.get('corners', {})
    if len(anchors) != 2 or camera.get('P') is None:
        return None
    charts = _charts(image, '26.3', parts, counts)
    shape = tuple((n + (SAMPLE_STRIDE-1)//2)//SAMPLE_STRIDE for n in image.size[::-1])
    masks = [observed_support(part['support'], shape, image.size) if part.get('support')
             else np.zeros(shape, bool) for part in parts]
    p = np.asarray(camera['P'], float)
    carriers = {id(mask): plane for mask, plane, _ in _carriers(image, charts, masks, p)}
    pair = [(chart, anchors[part['id']], carriers.get(id(mask))) for chart, part, mask in zip(charts, parts, masks)
            if part['id'] in anchors]
    if len(pair) != 2:
        return None

    def polygons(planes):
        result = []
        for i, (chart, anchor, _) in enumerate(pair):
            if planes[i] is None or planes[1-i] is None:
                return []
            axis, level, sign = planes[i]
            other, start, side = planes[1-i]
            if (axis, sign) != (anchor[0], anchor[2]) or {axis, other} != {0, 2}:
                return []
            width, height = np.rint(chart['pitch']['counts']).astype(int)
            end = start - side * width
            quad = np.zeros((4, 3))
            quad[:, axis] = level
            quad[:, other] = [start, end, end, start]
            quad[:, 1] = [0, 0, height, height]
            uv = quad @ p[:, :3].T + p[:, 3]
            if (uv[:, 2] <= 0).any():
                return []
            result.append((uv[:, :2] / uv[:, 2:]).round(2).tolist())
        return result

    return dict(ground=polygons([a for _, a, _ in pair]), chart=polygons([c for _, _, c in pair]),
                revised=bool(record['added'] or record['removed']))
