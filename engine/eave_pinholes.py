"""Repair tiny unassigned holes surrounded by a continuous visible eave underside."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from engine.entrance_regions import _components
from engine.plane_consensus import row_runs
from engine.surface_perimeter import _native_alpha, with_outline

METHOD = 'enclosed-underside-texture'
RADIUS = 3


def row_background(rgb, y, x):
    """Compare with both 2% side margins at the same height in the source image."""
    width = max(2, rgb.shape[1]//50)
    pixels = rgb[y, x].astype(np.int16)
    distances = []
    for margin in (slice(0, width), slice(-width, None)):
        band = rgb[y, margin].astype(np.int16)
        distances.append(np.max(abs(band-pixels[:,None]), axis=2).min(axis=1))
    return float(np.mean(np.maximum(*distances) <= 3))


def unassigned_holes(mask):
    # Flood the space outside the outline from a padded corner so open notches
    # and the exterior are removed from the list of potential interior holes.
    padded = Image.fromarray(np.pad(mask.astype('uint8'), 1))
    ImageDraw.floodfill(padded, (0,0), 2)
    return np.asarray(padded)[1:-1,1:-1] == 0


def refine(image, parts):
    result = list(parts)
    observations = []
    patches = [_native_alpha(p['support'], image.size) if p.get('support') else None for p in parts]
    rgb = np.asarray(image.convert('RGB'))
    for i, (part, patch) in enumerate(zip(parts,patches)):
        normal = part.get('normal_camera')
        if patch is None or normal is None or normal[1] < .85:
            continue
        alpha, (left,top,right,bottom) = patch
        if right-left < 4*(bottom-top):
            continue
        mask = np.asarray(alpha,bool)
        holes = unassigned_holes(mask)
        if not holes.any():
            continue
        occupied = np.zeros_like(mask)
        for j, other in enumerate(patches):
            if i == j or other is None:
                continue
            support, (x,y,r,b) = other
            l,t,r,b = max(x,left), max(y,top), min(r,right), min(b,bottom)
            if l<r and t<b:
                occupied[t-top:b-top,l-left:r-left] |= np.asarray(support,bool)[t-y:b-y,l-x:r-x]
        corrected = mask.copy()
        for cells in _components(holes):
            if not 4 <= len(cells) <= 1024:
                continue
            ys,xs = cells.T
            if occupied[ys,xs].any():
                continue
            y,x = cells.min(0)-RADIUS
            b,r = cells.max(0)+RADIUS+1
            if y<0 or x<0 or b>mask.shape[0] or r>mask.shape[1]:
                continue
            inner = np.zeros((b-y,r-x),bool)
            inner[ys-y,xs-x] = True
            outer = np.asarray(Image.fromarray(inner.astype('uint8')).filter(ImageFilter.MaxFilter(2*RADIUS+1)),bool)
            ring = outer & ~inner
            if not mask[y:b,x:r][ring].all() or occupied[y:b,x:r][ring].all():
                continue
            gy,gx = ys+top,xs+left
            if row_background(rgb,gy,gx)>=.15:
                continue
            rim = rgb[top+y:top+b,left+x:left+r][ring]
            step = np.abs(np.quantile(rgb[gy,gx],[.25,.5,.75],axis=0)-np.quantile(rim,[.25,.5,.75],axis=0)).max()
            if step>8:
                continue
            corrected[ys,xs] = True
            observations.append(dict(host=part['id'], pixels=len(cells), box=[int(gx.min()),int(gy.min()),int(gx.max()+1),int(gy.max()+1)]))
        if not np.array_equal(corrected,mask):
            runs = [[y+top, x+left, r+left] for y,x,r in row_runs(corrected)]
            result[i] = dict(part, support=dict(size=list(image.size),runs=runs), support_area=int(corrected.sum()),
                             measured_eave_support=part['support'])
            if 'support_area' in part:
                result[i]['measured_eave_support_area'] = part['support_area']
            result[i].pop('outline',None)
    return result, observations


def measurement_parts(parts):
    """Keep the measured grid and depth evidence for an outline-only repair."""
    measured = []
    for part in parts:
        original = part.get('measured_eave_support')
        if original is None:
            measured.append(part)
        else:
            candidate = dict(part, support=original)
            if 'measured_eave_support_area' in part:
                candidate['support_area'] = part['measured_eave_support_area']
            else:
                candidate.pop('support_area',None)
            measured.append(with_outline(candidate))
    return measured
