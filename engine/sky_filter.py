"""Exclude clear-day sky and its blue-tinted vanilla clouds from visible surface masks."""
import numpy as np
from PIL import Image

from engine.camera_lines import _runs
from engine.plane_consensus import row_runs
from engine.surface_perimeter import _native_alpha


METHOD = 'clear-noon-sky-v1'
SETTINGS = dict(min_red=115, min_green=159, min_blue=198, min_green_minus_red=12,
                max_green_minus_red=52, max_chroma_error=12, max_sky_curve_error=4,
                max_seed_height_fraction=.5, min_clear_pixels=32, min_blue_span=6)


def sky_masks(image):
    """Require an exterior-connected sky region with a clear-sky color gradient.

    Clear-noon vanilla sky reaches blue saturation and follows a narrow red/green
    gradient. The rendered clouds in these captures retain that blue cast but
    have a darker blue channel. Flat neutral whites and grays stay unclassified.
    """
    r, g, b = np.asarray(image.convert('RGB'), dtype=np.int16).transpose(2, 0, 1)
    s = SETTINGS
    warm = g-r
    cool = b-g
    daylight = ((r >= s['min_red']) & (g >= s['min_green']) & (b >= s['min_blue']) &
                (warm >= s['min_green_minus_red']) & (warm <= s['max_green_minus_red']) &
                (np.abs(cool-2*warm) <= s['max_chroma_error']))
    clear = daylight & (b == 255) & (np.abs(g-(.69*r+84)) <= s['max_sky_curve_error'])
    tinted = daylight & (b < 255)
    accepted = np.zeros_like(clear)
    height, width = clear.shape
    for runs in _runs(clear | tinted):
        if not any(y == 0 or (y < s['max_seed_height_fraction']*height and (a == 0 or z == width))
                   for y, a, z in runs):
            continue
        extent = np.zeros_like(clear)
        for y, a, z in runs:
            extent[y, a:z] = True
        seeds = r[extent & clear]
        if seeds.size >= s['min_clear_pixels'] and np.ptp(seeds) >= s['min_blue_span']:
            accepted |= extent
    return accepted & clear, accepted & tinted


def filter_surface(image, parts, masks=None):
    """Apply after all proposal-generating stages and before outlines, crops and lift."""
    blue, cloud = sky_masks(image) if masks is None else masks
    excluded = blue | cloud
    output = []
    log = dict(blue_pixels=int(blue.sum()), cloud_pixels=int(cloud.sum()), removed=[],
               trimmed=[], removed_blue_pixels=0, removed_cloud_pixels=0)
    for part in parts:
        native = _native_alpha(part['support'], image.size)
        if native is None:
            log['removed'].append(part['id'])
            continue
        alpha, (x0, y0, x1, y1) = native
        owned = np.asarray(alpha) != 0
        crop = np.s_[y0:y1, x0:x1]
        rejected = owned & excluded[crop]
        if not rejected.any():
            output.append(part)
            continue
        log['removed_blue_pixels'] += int(np.count_nonzero(rejected & blue[crop]))
        log['removed_cloud_pixels'] += int(np.count_nonzero(rejected & cloud[crop]))
        kept = owned & ~rejected
        if not kept.any():
            log['removed'].append(part['id'])
            continue
        log['trimmed'].append(part['id'])
        runs = np.asarray(row_runs(kept)) + [y0, x0, x0]
        cleaned = {k:v for k,v in part.items() if k not in
                   ('outline','corner_junctions','source_crop','context_crop','sampling_chart')}
        cleaned.update(support=dict(size=list(image.size), runs=runs.tolist()), support_area=int(kept.sum()))
        # An existing chart is a sampling window. Retain its orientation when its
        # corners still have image evidence; holes within it are masked by the sampler.
        if 'quad' in cleaned:
            xy = np.floor(cleaned['quad']).astype(int)
            x, y = xy.T
            inside = (x>=0) & (x<image.width) & (y>=0) & (y<image.height)
            if np.any(excluded[y[inside], x[inside]]) or kept.sum() < rejected.sum():
                del cleaned['quad']
                cleaned['geometry_status'] = 'mask only'
        output.append(cleaned)
    affected = set(log['removed'] + log['trimmed'])
    if affected:
        output = [dict(p, corner_junctions=[j for j in p['corner_junctions'] if j['neighbor'] not in affected])
                  if 'corner_junctions' in p else p for p in output]
    return output, log
