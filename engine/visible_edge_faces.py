"""Seek misplaced or dropped mask-edge samples on already estimated visible planes."""
import numpy as np
from PIL import Image
from engine.partial_atlas_face import verify
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE, observed_support


def neighboring(cell, axis, cells):
    for dim in range(3):
        if dim == axis:
            continue
        for offset in (-1,1):
            point=list(cell)
            point[dim]+=offset
            if tuple(point) in cells:
                return True
    return False


def native_support(source,size):
    width,height=source['size']
    mask=np.zeros((height,width),bool)
    for y,left,right in source['runs']:
        mask[y,left:right]=True
    if size!=(width,height):
        mask=np.asarray(Image.fromarray(mask).resize(size,Image.Resampling.NEAREST))
    return mask


def extend(image,parts,P,cells,offsets,atlas,*,strict=False):
    if not cells or not offsets or atlas is None:
        return set()
    stride=SAMPLE_STRIDE
    shape=tuple((n+(stride-1)//2)//stride for n in image.size[::-1])
    P=np.asarray(P,float)
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    owner=_frontmost_cells(sorted(cells),P,eye,shape)
    by_id={row['id']:row for row in offsets}
    rgb=np.asarray(image.convert('RGB'))
    inverse=np.linalg.inv(P[:,:3])
    proposals={}
    for part in parts:
        row=by_id.get(part['id'])
        if row is None or not part.get('support'):
            continue
        axis,level=row['axis'],row['level']
        mask=observed_support(part['support'],shape,image.size) & (owner<0)
        y,x=np.nonzero(mask)
        if len(y)<2:
            continue
        rays=np.column_stack(((x+.5)*stride,(y+.5)*stride,np.ones(len(x))))@inverse.T
        scale=np.divide(level-eye[axis],rays[:,axis],out=np.zeros(len(x)),where=abs(rays[:,axis])>1.e-7)
        good=(scale>1.5)&(scale<150)
        xyz=np.floor(eye+scale[good,None]*rays[good]).astype(int)
        sign=1 if eye[axis]>level else -1
        xyz[:,axis]=level-int(sign>0)
        candidates,counts=np.unique(xyz,axis=0,return_counts=True)
        native=native_support(part['support'],image.size)
        for point in candidates[counts>=2]:
            cell=tuple(point.tolist())
            if cell[1]<0 or cell in cells or (strict and not neighboring(cell,axis,cells)):
                continue
            if verify(rgb,atlas,mask,P,cell,axis,sign,native_mask=native,
                      threshold=.72 if strict else .60,minimum=32 if strict else 42):
                proposals.setdefault(cell,[]).append((axis,mask))
    if not proposals:
        return set()
    all_cells=sorted(cells | proposals.keys())
    front,axes=_frontmost_cells(all_cells,P,eye,shape,face_axes=True)
    added=set()
    for index,cell in enumerate(all_cells):
        if cell in proposals and any(np.count_nonzero((front==index)&(axes==axis)&mask)>=2
                                     for axis,mask in proposals[cell]):
            added.add(cell)
    return added
