"""Undo a photographed diagonal cut between siblings on the same upright wall."""

from itertools import combinations

import numpy as np
from PIL import ImageFilter

from engine.plane_consensus import with_mask
from engine.surface_model import SETTINGS as LENS_SETTINGS, intrinsic
from engine.surface_perimeter import _native_alpha


METHOD = 'photographic-diagonal-sibling-join'
SETTINGS = dict(min_boundary_pixels=96, max_median_color_step=6.,
                min_angle_degrees=18., min_line_ratio=15., min_normal_dot=.985)


def _tangents(normal, size):
    # With zero camera roll, projected world-up is in the camera y/z plane.
    # A wall normal is perpendicular to world-up. An edge-on wall cannot
    # reliably constrain pitch and a near-horizontal surface is outside scope.
    if abs(normal[2]) < .18 or abs(normal[1]) > .50:
        return None
    up = np.array([0., -1., normal[1] / normal[2]])
    if abs(up[2]) > .85:
        return None
    focal = size[1] / (2 * np.tan(np.radians(LENS_SETTINGS['assumed_vertical_fov_degrees']) / 2))
    return intrinsic(size, focal) @ np.stack((up, np.cross(normal, up)), axis=1)


def _on_box(patch, box):
    alpha, (x, y, r, b) = patch
    xx, yy, rr, bb = box
    left, top, right, bottom = max(x, xx), max(y, yy), min(r, rr), min(b, bb)
    mask = np.zeros((bb-yy, rr-xx), bool)
    if left < right and top < bottom:
        mask[top-yy:bottom-yy, left-xx:right-xx] = np.asarray(alpha, bool)[top-y:bottom-y, left-x:right-x]
    return mask


def _contact(first, second, smooth, size, vps):
    x, y, r, b = first[1]
    xx, yy, rr, bb = second[1]
    if min(r,rr) < max(x,xx)-5 or min(b,bb) < max(y,yy)-5:
        return None
    box = (max(0,max(x,xx)-13), max(0,max(y,yy)-13),
           min(size[0],min(r,rr)+13), min(size[1],min(b,bb)+13))
    a, other = _on_box(first, box), _on_box(second, box)
    rgb = np.asarray(smooth.crop(box), np.float32)
    h, w = a.shape
    points, left, right, local = [], [], [], []
    for dy, dx in ((0,1),(1,0),(0,-1),(-1,0)):
        for gap in (1,3,5):
            ys, xs = np.nonzero(a & np.roll(other, (-dy*gap,-dx*gap), (0,1)))
            keep = (xs>=10)&(xs<w-10)&(ys>=10)&(ys<h-10)
            xs, ys = xs[keep], ys[keep]
            # Both cross-seam samples and their local controls must actually
            # belong to the respective surfaces, including around holes.
            keep = (~a[ys+dy,xs+dx] & a[ys-2*dy,xs-2*dx] & a[ys-6*dy,xs-6*dx] &
                    other[ys+(gap+2)*dy,xs+(gap+2)*dx] & other[ys+(gap+5)*dy,xs+(gap+5)*dx])
            xs, ys = xs[keep], ys[keep]
            if not xs.size:
                continue
            a_rgb = rgb[ys-2*dy,xs-2*dx]
            b_rgb = rgb[ys+(gap+2)*dy,xs+(gap+2)*dx]
            points.append(np.column_stack((xs,ys)))
            left.append(a_rgb)
            right.append(b_rgb)
            local.append((abs(a_rgb-rgb[ys-6*dy,xs-6*dx]) +
                          abs(b_rgb-rgb[ys+(gap+5)*dy,xs+(gap+5)*dx])) / 2)
    if not points:
        return None
    xy = np.unique(np.concatenate(points), axis=0)
    if len(xy) < SETTINGS['min_boundary_pixels']:
        return None
    vals, basis = np.linalg.eigh(np.cov(xy.T))
    if vals[1] < SETTINGS['min_line_ratio'] * max(1., vals[0]):
        return None
    center = xy.mean(axis=0) + box[:2]
    directions = vps[:2] - center[:,None]*vps[2]
    norms = np.linalg.norm(directions, axis=0)
    if min(norms) < 1.e-8:
        return None
    angles = np.degrees(np.arccos(np.clip(abs(basis[:,-1] @ directions)/norms, 0, 1)))
    if min(angles) < SETTINGS['min_angle_degrees']:
        return None
    a_rgb, b_rgb = np.concatenate(left), np.concatenate(right)
    step = float(max(abs(np.median(a_rgb,axis=0) - np.median(b_rgb,axis=0))))
    contrast = float(abs(a_rgb-b_rgb).mean())
    within = float(np.concatenate(local).mean())
    if step > SETTINGS['max_median_color_step'] or contrast > max(4., 1.5*within):
        return None
    return dict(border_pixels=len(xy), minimum_angle_deg=round(float(min(angles)),2),
                median_step=round(step,2), contrast=round(contrast,2), within=round(within,2))


def _remap_junctions(parts, output, retired):
    """Keep an existing corner contact only when both endpoints still exist."""
    old = {p['id']:p for p in parts}
    new = {p['id']:p for p in output}
    links = {}

    def endpoint(key, index):
        part = old.get(key)
        if not part or 'quad' not in part or not 0 <= index < len(part['quad']):
            return None
        if key not in retired:
            return key, index
        result = new[retired[key]]
        if 'quad' not in result:
            return None
        delta = np.asarray(result['quad']) - part['quad'][index]
        distances = np.linalg.norm(delta, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 2.:
            return result['id'], nearest
        return None

    for part in parts:
        for link in part.get('corner_junctions',[]):
            key, neighbor = part['id'], link['neighbor']
            if key not in retired and neighbor not in retired:
                links.setdefault(key,[]).append(link)
                continue
            a = endpoint(key, link['corner'])
            b = endpoint(neighbor, link['neighbor_corner'])
            if a is None or b is None or a[0] == b[0]:
                continue
            distance = np.linalg.norm(np.asarray(new[a[0]]['quad'][a[1]]) - new[b[0]]['quad'][b[1]])
            if distance > max(2., link.get('distance_px',0.) + 2.):
                continue
            updated = dict(link, corner=a[1], neighbor=b[0], neighbor_corner=b[1], distance_px=round(float(distance),2))
            outgoing = links.setdefault(a[0],[])
            if updated not in outgoing:
                outgoing.append(updated)
    return [dict(p,corner_junctions=links.get(p['id'],[]))
            if p['id'] in links or 'corner_junctions' in p else p for p in output]


def refine(image, parts):
    """Keep unknown pixels unknown. Each joined support is an exact native union."""
    groups = {}
    for i, part in enumerate(parts):
        if part.get('parent_id') and part.get('support') and part.get('normal_camera') is not None:
            groups.setdefault(part['parent_id'], []).append(i)
    pairs = []
    patches = {}
    smooth = None
    for siblings in groups.values():
        for j, i in combinations(siblings, 2):
            normals = np.asarray([parts[k]['normal_camera'] for k in (i,j)], float)
            if normals.shape != (2,3) or not np.all(np.isfinite(normals)):
                continue
            lengths = np.linalg.norm(normals, axis=1)
            if min(lengths) < 1.e-8:
                continue
            normal, other = normals / lengths[:,None]
            if normal @ other < SETTINGS['min_normal_dot']:
                continue
            vps = _tangents(normal, image.size)
            if vps is None:
                continue
            for k in (i, j):
                if k not in patches:
                    patches[k] = _native_alpha(parts[k]['support'], image.size)
            if patches[i] is None or patches[j] is None:
                continue
            if smooth is None:
                smooth = image.convert('RGB').filter(ImageFilter.GaussianBlur(1.35))
            obs = _contact(patches[i], patches[j], smooth, image.size, vps)
            if obs:
                pairs.append((obs['contrast']-obs['within'], i, j, obs))
    if not pairs:
        return parts, []
    gray = np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)), np.float32)[...,None]
    seen, joined, events = set(), [], []
    box = (0, 0, *image.size)
    for _, i, j, obs in sorted(pairs):
        if i in seen or j in seen:
            continue
        first, second = _on_box(patches[i],box), _on_box(patches[j],box)
        areas = [int(first.sum()), int(second.sum())]
        base = parts[i] if areas[0] > areas[1] else parts[j]
        # The normal and depth estimate come from the larger observed sibling.
        # The old split and chart-specific contacts do not describe the union.
        inherited = base['id']
        base = {k:v for k,v in base.items() if k not in
                ('source_crop','context_crop','corner_junctions','junction_split')}
        result = with_mask(base, first|second, image.size, gray, np.ones(2))
        result['id'] += '-joined'
        result['diagonal_join'] = dict(inputs=[parts[i]['id'],parts[j]['id']], normal_depth_source=inherited)
        joined.append(result)
        events.append(dict(inputs=[parts[i]['id'],parts[j]['id']], output=result['id'],
                           input_areas_px=areas, output_area_px=result['support_area'], **obs))
        seen.update((i,j))
    output = [p for i,p in enumerate(parts) if i not in seen]+joined
    if any(p.get('corner_junctions') for p in parts):
        retired = {key:e['output'] for e in events for key in e['inputs']}
        output = _remap_junctions(parts, output, retired)
    return output, events
