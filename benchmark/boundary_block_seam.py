"""Locate a missing grid origin from repeated straight edges inside the visible chart."""
import numpy as np


def boundary_phase(mask, count):
    if count is None or count < 1.8:
        return None
    height, width = mask.shape
    pitch = width/count
    if pitch < 6 or height < 7:
        return None
    # A sign change is a visible mask edge. Discard the sampling window's own edges.
    changes = np.diff(mask.astype(np.int8), axis=1)
    margin = max(3., .08*pitch)
    locations, votes = [], []
    for x in np.flatnonzero(np.any(changes,axis=0)):
        coord = x+1.
        if min(coord, width-coord) <= margin:
            continue
        column = changes[:,x]
        ends = np.r_[0, np.where(np.diff(column))[0]+1, height]
        runs = np.diff(ends)
        long = (column[ends[:-1]]!=0) & (runs>=max(5, min(18, .12*height)))
        if long.any():
            locations.append(coord/pitch)
            votes.append(float(runs[long].sum()))
    if len(locations) < 2:
        return None
    loc, weight = np.array(locations), np.array(votes)
    # Long edges can drift by a pixel. Cluster by position within one repeat.
    phases = np.arange(64)/64.
    distance = abs((loc[None,:]-phases[:,None]+.5)%1.-.5)
    width_blocks = min(.15,max(.07,1.5/pitch))
    scores = (np.maximum(0.,1.-distance/width_blocks)*weight).sum(axis=1)
    best = int(np.argmax(scores))
    same = distance[best] <= width_blocks
    if weight[same].sum() < .75*weight.sum() or np.ptp(loc[same]) < .75:
        return None
    offset = (loc[same]-phases[best]+.5)%1.-.5
    return float((phases[best]+np.average(offset,weights=weight[same]))%1.)


def predict(sample, baseline):
    phases = list(baseline['phase'])
    if sample is not None:
        for axis,mask in enumerate((sample['mask'],sample['mask'].T)):
            if phases[axis] is None:
                phases[axis] = boundary_phase(mask,baseline['counts'][axis])
    return dict(counts=list(baseline['counts']),phase=phases)
