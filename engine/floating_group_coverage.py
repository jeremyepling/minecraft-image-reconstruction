"""Rank connected Floating groups by the visible image area their cells cover."""
import cv2
import numpy as np

from benchmark.floating_package_rechecks import native_mask
from benchmark.mobile_surface_evidence import parent
from engine.geometry import UNIT, project


def coverage_scores(groups, charts, planes, size):
    """Count projected source-supported pixels at one quarter linear resolution."""
    width, height = size
    raster_width, raster_height = (width + 3) // 4, (height + 3) // 4
    plane_by_id = {plane['id']: plane for plane in planes}
    selected_ids = {chart_id for group in groups for chart_id in group['origins']}
    support = {}
    footprints = {}
    scale = np.array((raster_width / width, raster_height / height)) * 16
    for chart in charts:
        chart_id = chart['id']
        if chart_id not in selected_ids:
            continue
        plane_id = parent(chart_id)
        cells = chart['export_uv'] if 'export_uv' in chart else [[c['u'], c['v']] for c in chart['cells']]
        area = np.zeros((raster_height, raster_width), dtype=np.uint8)
        if cells and plane_id in plane_by_id:
            if plane_id not in support:
                mask = native_mask(plane_by_id[plane_id], size).astype(np.uint8)
                support[plane_id] = cv2.resize(mask, (raster_width, raster_height),
                                              interpolation=cv2.INTER_NEAREST_EXACT)
            quads = project(np.asarray(chart['H']), np.asarray(cells)[:, None, :] + UNIT)
            finite = np.isfinite(quads).all(axis=(1, 2))
            coords = np.rint(np.clip(quads[finite] * scale, -2**28, 2**28)).astype(np.int32)
            for quad in coords:
                cv2.fillConvexPoly(area, quad, 1, shift=4)
            area &= support[plane_id]
        footprints[chart_id] = area

    scores = []
    for group in groups:
        coverage = np.zeros((raster_height, raster_width), dtype=np.uint8)
        for chart_id in group['origins']:
            coverage |= footprints[chart_id]
        scores.append(int(coverage.sum()))
    return scores


def select_group(groups, charts, planes, size):
    scores = coverage_scores(groups, charts, planes, size)
    incumbent = groups[0]
    best = max(range(len(groups)), key=scores.__getitem__)
    alternative = groups[best]
    if best and 4 * len(alternative['blocks']) <= len(incumbent['blocks']):
        return alternative
    return incumbent
