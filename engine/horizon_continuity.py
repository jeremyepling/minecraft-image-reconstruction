"""Identify narrow surface supports that follow the sky/ground edge into the frame margins."""
import numpy as np

from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs


def margin_continuation(rgb, top, bottom):
    h, w, _ = rgb.shape
    pad = max(8, h//90)
    if top < 2*pad or bottom+3*pad >= h:
        return None
    border = max(8, w//5)
    x = np.arange(w)
    outer = np.r_[x[:border], x[-border:]]
    sky = np.median(rgb[top-pad, outer], axis=0)
    # A near-constant, blue sky above both independent margin strips.
    if sky[2]-sky[0] < 20 or np.mean(abs(rgb[top-pad, outer]-sky).max(axis=1)<=4) < .9:
        return None
    y0 = top-2*pad
    band = rgb[y0:bottom+pad+1]
    is_sky = abs(band-sky).max(axis=2) <= 7
    departures = (~is_sky).argmax(axis=0)
    edge = y0+departures
    has_edge = is_sky[0] & (~is_sky).any(axis=0)
    offsets = [3, 5, 7, 10]
    profiles = rgb[edge[:,None]+offsets, x[:,None]]
    valid = has_edge & (abs(profiles[:,1]-sky).max(axis=1)>24)
    # A foreground column stops the trace when its full below-edge profile
    # diverges from the distant ground independently observed in the margin.
    reached = np.zeros(w, bool)
    color_ok = np.zeros((bottom-top,w),bool)
    for indices in (x[:border], x[-border:]):
        open_sky = is_sky[0] & (abs(rgb[top-pad]-sky).max(axis=1) <= 7)
        if indices[0] == 0:
            connected_sky = np.logical_and.accumulate(open_sky)
        else:
            connected_sky = np.logical_and.accumulate(open_sky[::-1])[::-1]
        witnesses = indices[valid[indices] & connected_sky[indices]]
        if len(witnesses) < border//3:
            continue
        template = np.median(profiles[witnesses], axis=0)
        similar = valid & connected_sky & (abs(profiles-template).max(axis=(1,2)) <= 5)
        if similar[indices].sum() < border//3:
            continue
        ground = indices[similar[indices]]
        # Check each pixel too. A thin contrasting rail can occupy a row
        # between two matching ground-profile samples.
        offset = np.arange(top,bottom)[:,None] - edge
        for distance in np.unique(offset[:,similar]):
            sample_y = np.clip(edge[ground,None]+distance+[-1,0,1],0,h-1)
            witness = rgb[sample_y,ground[:,None]].reshape(-1,3)
            lo, hi = np.percentile(witness, [5,95], axis=0)
            ys, xs = np.nonzero((offset == distance) & similar)
            observed = rgb[top+ys,xs]
            color_ok[ys,xs] |= ((observed >= lo-2) & (observed <= hi+2)).all(axis=1)
        # Follow only uninterrupted columns. On one side the visible horizon
        # may end early due to distant terrain, but the other side remains valid.
        cuts = np.flatnonzero(np.diff(similar.astype(int), prepend=0, append=0))
        for left,right in cuts.reshape(-1,2):
            if min(right,indices[-1]+1) - max(left,indices[0]) < border//5:
                continue
            # Leave a few source pixels around the first non-background column.
            reached[left + (3 if left else 0):right - (3 if right<w else 0)] = True
    return reached, edge, color_ok


def refine(image, parts):
    rgb = None
    output, events = [], []
    for part in parts:
        support = part.get('support')
        if not support or not support.get('runs') or part.get('quad') is not None:
            output.append(part)
            continue
        runs = np.asarray(support['runs'])
        sx, sy = np.asarray(image.size)/support['size']
        span = (runs[:,2].max()-runs[:,1].min())*sx
        height = (runs[:,0].max()+1-runs[:,0].min())*sy
        if span < max(48,image.width//18,8*height) or height > image.height/40:
            output.append(part)
            continue
        top = int(np.floor(runs[:,0].min()*sy))
        bottom = int(np.ceil((runs[:,0].max()+1)*sy))
        if rgb is None:
            rgb = np.asarray(image.convert('RGB'), np.int16)
        boundary = margin_continuation(rgb, top, bottom)
        if boundary is None:
            output.append(part)
            continue
        reached, edge, color_ok = boundary
        mask = native_mask(part,image.size)
        y,x = np.nonzero(mask)
        clear = reached[x] & color_ok[y-top,x] & (abs(edge[x]-y) <= max(7, int(height)+2))
        if clear.sum() < .85*len(clear):
            output.append(part)
            continue
        # Around the end of a traced run, distinguish actual roof pixels
        # from ground-colored pixels at the same row of the distant horizon.
        remaining = np.flatnonzero(~reached[x])
        for delta in (-4, 4):
            peers = x[remaining,None] + delta + np.sign(delta)*np.arange(3)
            inside = (peers.min(axis=1) >= 0) & (peers.max(axis=1) < image.width)
            tested, peers = remaining[inside], peers[inside]
            verified = (reached[peers] & color_ok[y[tested,None]-top,peers]).all(axis=1)
            stable = abs(rgb[y[tested,None],peers]-rgb[y[tested],x[tested]][:,None]).max(axis=(1,2)) <= 1
            clear[tested[verified & stable]] = True
        mask[y[clear],x[clear]] = False
        if mask.any():
            revised = dict(part,support=dict(size=list(image.size),runs=row_runs(mask)),support_area=int(mask.sum()))
            revised.pop('outline',None)
            output.append(revised)
        events.append(dict(id=part['id'],removed=int(clear.sum()),retained=int(mask.sum())))
    if events:
        changed = {event['id'] for event in events}
        output = [dict(part, corner_junctions=[junction for junction in part['corner_junctions']
                                               if junction['neighbor'] not in changed])
                  if part.get('corner_junctions') else part for part in output]
    return output, events
