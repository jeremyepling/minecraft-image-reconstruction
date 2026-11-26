"""Remove proposals with no visible pixel in the proposed reconstruction."""
import numpy as np

from engine.visible_raster import SAMPLE_STRIDE


def fully_occluded_cells(size, projection, cells):
    """Confirm occlusion at every native pixel center before dropping a block."""
    if not cells:
        return set()
    from engine.visible_placement import _frontmost_cells
    points=sorted(map(tuple,cells))
    P=np.asarray(projection,dtype=float)
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    # Like the placement masks, sample only coarse centers inside the actual viewport.
    shape=tuple((n+(SAMPLE_STRIDE-1)//2)//SAMPLE_STRIDE for n in size[::-1])
    coarse=_frontmost_cells(points,P,eye,shape)
    emitted=np.bincount(coarse[coarse>=0],minlength=len(points))>0
    if emitted.all():
        return set()
    # _frontmost_cells rasterizes onto a SAMPLE_STRIDE grid. Scale screen x,y
    # so each sample here lands at an original input pixel center.
    native_P=P.copy()
    native_P[:2]*=SAMPLE_STRIDE
    fine=_frontmost_cells(points,native_P,eye,size[::-1])
    emitted |= np.bincount(fine[fine>=0],minlength=len(points))>0
    return {points[i] for i in np.flatnonzero(~emitted)}
