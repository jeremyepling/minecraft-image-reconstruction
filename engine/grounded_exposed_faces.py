"""Verify atlas matches on original photo pixels exposed by rejecting misplaced blocks."""
import numpy as np
from engine import thin_atlas_faces
from engine.camera_atlas_floor import _atlas
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE


def newly_exposed(image, parts, camera, original, pruned, proposals):
    if not proposals:
        return set()
    P=np.array(camera['P'],dtype=float,copy=True)
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    # A cell counts only if its projected horizontal face now owns a native
    # photo pixel previously covered by a cell removed in this photograph.
    P[:2]*=SAMPLE_STRIDE
    before=sorted(original)
    after=sorted(pruned|proposals)
    old=_frontmost_cells(before,P,eye,image.size[::-1])
    front,axis=_frontmost_cells(after,P,eye,image.size[::-1],face_axes=True)
    removed=np.array([p not in pruned for p in before]+[False])
    # The newly claimed pixels must themselves belong to a detected surface.
    y,x=np.nonzero(removed[old] & (axis==1) & (front>=0))
    supported=np.zeros(len(y),bool)
    for part in parts:
        support=part.get('support')
        if support and len(y):
            width,height=support['size']
            local=np.zeros((height,width),bool)
            for row,left,right in support['runs']:
                local[row,left:right]=True
            supported |= local[((y+.5)*height/image.height).astype(int),((x+.5)*width/image.width).astype(int)]
    hits=front[y[supported],x[supported]]
    count=np.bincount(hits,minlength=len(after))
    return {after[i] for i in np.flatnonzero(count>=4) if after[i] in proposals}


def recover(image, parts, camera, original, pruned, *, atlas_path=None):
    if original==pruned or not pruned or camera.get('P') is None:
        return pruned
    atlas=_atlas('26.3',atlas_path)
    candidates=thin_atlas_faces.recover(image,parts,camera,pruned,atlas,mode='rank75')-original
    return pruned|newly_exposed(image,parts,camera,original,pruned,candidates)
