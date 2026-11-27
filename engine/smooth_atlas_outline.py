"""Extend a visible smooth wall only when source, host and atlas agree away from its rim."""
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter

from engine import atlas_count
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs
from engine.smooth_extent_placement import proposals
from engine.surface_perimeter import with_outline


METHOD = 'atlas-smooth-visible-outline'


def material(pixels, atlas, smooth):
    # Use the same per-sprite shading and contrast fit as the ordinary atlas matcher.
    pixels = pixels[::max(1, len(pixels)//512)]
    _, loss = atlas_count._shortlist(pixels, np.ones(len(pixels), bool), atlas)
    si = np.flatnonzero(smooth)[np.argmin(loss[smooth])]
    ti = np.flatnonzero(~smooth)[np.argmin(loss[~smooth])]
    return dict(smooth=atlas.labels[si], textured=atlas.labels[ti],
                margin=float(loss[ti]-loss[si]), median=np.median(pixels, axis=0).tolist())


def host_witness(rgb, host, box, target):
    # Find an independently visible, nearly full 8x8 host patch. Its pixels have
    # never been filtered by the proposed color; strong local texture can overrule it.
    x0,y0,x1,y1 = box
    height,width = host.shape
    best = None
    for y in range(max(0,y0-20),min(height,y1+20)-7,6):
        for x in range(max(0,x0-20),min(width,x1+20)-7,6):
            seen = host[y:y+8,x:x+8]
            if seen.sum()<48:
                continue
            pixels = rgb[y:y+8,x:x+8][seen]
            error = float(abs(np.median(pixels,axis=0)-target).max())
            if best is None or (error,x,y)<best[:3]:
                best = error,x,y,pixels
    return best


def margin_witness(rgb, box, target):
    # Compare signed channel differences against both frame margins at the same height.
    _,y0,_,y1 = box
    height,width,_ = rgb.shape
    if width<48:
        return None
    best = None
    for y in range(max(0,y0-3),min(height-7,y1),4):
        for x in (0,8,16,width-24,width-16,width-8):
            pixels = rgb[y:y+8,x:x+8].reshape(-1,3)
            error = float(abs(np.median(pixels,axis=0)-target).max())
            if best is None or error<best[0]:
                best = error,x,y,pixels
    return best


def supported_runs(host, runs, box, radius):
    # Erode the union rather than the new pixels alone, preserving the seam
    # where the old and new portions of this same surface meet.
    if radius==0:
        return runs
    height,width = host.shape
    x0,y0,x1,y1 = box
    x0,y0,x1,y1 = max(0,x0-radius),max(0,y0-radius),min(width,x1+radius),min(height,y1+radius)
    extra = np.zeros((y1-y0,x1-x0),bool)
    for y,left,right in runs:
        extra[y-y0,left-x0:right-x0] = True
    full = np.pad(host[y0:y1,x0:x1] | extra, radius)
    interior = Image.fromarray(full.astype('uint8')).filter(ImageFilter.MinFilter(2*radius+1))
    good = extra & np.asarray(interior,bool)[radius:-radius,radius:-radius]
    return [(y+y0,left+x0,right+x0) for y,left,right in row_runs(good)]


def checked_regions(image, parts, regions, atlas, *, radius=2):
    rgb = np.asarray(image.convert('RGB'),np.int16)
    smooth = atlas.spreads.max(axis=1)<=3
    lookup = {p['id']:p for p in parts}
    hosts = {}
    checked = []
    for region in regions:
        key,runs = region['surface'],region['runs']
        box = [min(x for _,x,_ in runs),min(y for y,_,_ in runs),
               max(r for _,_,r in runs),max(y for y,_,_ in runs)+1]
        if key not in hosts:
            hosts[key] = native_mask(lookup[key],image.size)
        sampled = np.concatenate([rgb[y,left:right] for y,left,right in runs])
        fit = material(sampled,atlas,smooth)
        if fit['margin']<1:
            continue
        target = np.array(fit['median'])
        sponsor = host_witness(rgb,hosts[key],box,target)
        if sponsor is None or sponsor[0]>3 or material(sponsor[3],atlas,smooth)['margin']<1:
            continue
        margin = margin_witness(rgb,box,target)
        if margin is None or (margin[0]<=3 and material(margin[3],atlas,smooth)['margin']>=1):
            continue
        keep = supported_runs(hosts[key],runs,box,radius)
        if keep:
            checked.append(dict(surface=key, runs=keep, pixels=sum(r-l for _,l,r in keep), box=box,
                                smooth_sprite=fit['smooth'], rival_sprite=fit['textured']))
    return checked


def apply_regions(image, parts, checked):
    additions = defaultdict(list)
    for region in checked:
        additions[region['surface']].extend(region['runs'])
    updated = []
    for part in parts:
        runs = additions.get(part['id'])
        if not runs:
            updated.append(part)
            continue
        mask = native_mask(part,image.size)
        for y,left,right in runs:
            mask[y,left:right] = True
        item = dict(part,support_area=int(mask.sum()),support=dict(size=list(image.size),runs=row_runs(mask)),
                    measured_smooth_support=part['support'])
        if 'support_area' in part:
            item['measured_smooth_support_area'] = part['support_area']
        item.pop('outline',None)
        updated.append(item)
    return updated


def measurement_parts(parts):
    measured = []
    for part in parts:
        if 'measured_smooth_support' not in part:
            measured.append(part)
            continue
        item = dict(part,support=part['measured_smooth_support'])
        if 'measured_smooth_support_area' in part:
            item['support_area'] = part['measured_smooth_support_area']
        else:
            item.pop('support_area',None)
        measured.append(with_outline(item))
    return measured


def extra_interior(image, common, baseline):
    """Keep all previously accepted pixels with their original owner."""
    if not common or not baseline:
        return common
    occupied = np.zeros(image.size[::-1],bool)
    for row in baseline:
        for y,l,r in row['runs']:
            occupied[y,l:r] = True
    result=[]
    for row in common:
        keep=[]
        for y,l,r in row['runs']:
            for _,a,b in row_runs((~occupied[y,l:r])[None,:]):
                keep.append((y,a+l,b+l))
        if keep:
            result.append(dict(row,runs=keep,pixels=sum(b-a for _,a,b in keep)))
    return result


def with_common_wall(image, parts, version, *, atlas_loader):
    original = proposals(image,parts)
    extra = proposals(image,parts,common_wall=True)
    proposal_count = len(original)+len(extra)
    if not proposal_count:
        return parts, dict(events=[],proposals=0), []
    atlas=atlas_loader()
    if atlas is None:
        return parts, dict(events=[],proposals=proposal_count,status='atlas unavailable'), []
    if atlas.provenance['version']!=version:
        raise ValueError('Surface and texture atlas versions must match')
    baseline=checked_regions(image,parts,original,atlas)
    common=extra_interior(image,checked_regions(image,parts,extra,atlas),baseline)
    common=[dict(r,common_wall=True) for r in common]
    combined=baseline+common
    events=[{k:v for k,v in r.items() if k!='runs'} for r in combined]
    return apply_regions(image,parts,combined),dict(events=events,proposals=proposal_count,
                        atlas_sha256=atlas.provenance['atlas_sha256']),common


def refine(image, parts, version, *, atlas_loader):
    updated,status,_=with_common_wall(image,parts,version,atlas_loader=atlas_loader)
    return updated,status
