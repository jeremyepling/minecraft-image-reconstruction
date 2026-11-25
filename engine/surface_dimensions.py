"""Measure the visible extent of a surface using its measured texture spacing."""
import numpy as np

from engine.geometry import transform


def dimensions(region, quad, image_size, counts):
    result = dict(spans=[None,None], blocks=[None,None])
    if quad is None or counts is None:
        return result
    support = region.get('support')
    if support is None:
        extent = np.ones(2)
    else:
        runs = np.asarray(support['runs'])
        if not len(runs):
            return result
        y,lo,hi = runs.T
        # Include the full outside edges of observed pixels, even beyond the metering quad.
        corners = np.column_stack((np.r_[lo,hi,lo,hi],np.r_[y,y,y+1,y+1]))
        xy = corners * (np.asarray(image_size) / support['size'])
        try:
            H = np.linalg.inv(transform(quad))
        except np.linalg.LinAlgError:
            return result
        coords = xy @ H[:,:2].T + H[:,2]
        if not (np.all(coords[:,2]>1e-8) or np.all(coords[:,2]<-1e-8)):
            return result
        extent = np.ptp(coords[:,:2]/coords[:,2,None],axis=0)
    for k,n in enumerate(counts):
        if n is not None and np.isfinite(n) and n>0:
            result['spans'][k] = float(extent[k]*n)
            result['blocks'][k] = max(1,int(np.floor(result['spans'][k]+.5)))
    return result
