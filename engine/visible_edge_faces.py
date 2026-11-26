"""Seek misplaced or dropped mask-edge samples on already estimated visible planes."""
import numpy as np
from PIL import Image
from engine.partial_atlas_face import verify
from engine.visible_placement import _frontmost_cells, _sample_face
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


def neighbors(cells):
    points=set()
    for cell in cells:
        for dim in range(3):
            for delta in (-1,1):
                xyz=list(cell)
                xyz[dim]+=delta
                if xyz[1]>=0:
                    points.add(tuple(xyz))
    return points-cells


def verify_neighbors(cells,parts,image,rgb,P,eye,owner,atlas,proposals):
    """Look for unmatched photographed faces where detected masks touch across a crease."""
    available=np.zeros(owner.shape,bool)
    native=np.zeros(rgb.shape[:2],bool)
    for part in parts:
        if support:=part.get('support'):
            available|=observed_support(support,owner.shape,image.size)
            native|=native_support(support,image.size)
    available &= owner<0
    for cell in sorted(neighbors(cells)):
        for axis in range(3):
            sign=1 if eye[axis]>cell[axis]+1 else -1
            if cell[axis]<=eye[axis]<=cell[axis]+1 or not neighboring(cell,axis,cells):
                continue
            hit=_sample_face(P,cell,axis,sign,owner.shape)
            if hit is None:
                continue
            box,pixels,*_=hit
            if np.count_nonzero(pixels&available[box])<max(4,.20*pixels.sum()):
                continue
            if verify(rgb,atlas,available,P,cell,axis,sign,native_mask=native,threshold=.75,minimum=56):
                proposals.setdefault(cell,[]).append((axis,available))


def extend(image,parts,P,cells,offsets,atlas,*,strict=False,search=False,creases=False):
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
        native=native_support(part['support'],image.size)
        for plane in range(level-int(search)*2,level+int(search)*2+1):
            scale=np.divide(plane-eye[axis],rays[:,axis],out=np.zeros(len(x)),where=abs(rays[:,axis])>1.e-7)
            good=(scale>1.5)&(scale<150)
            xyz=np.floor(eye+scale[good,None]*rays[good]).astype(int)
            sign=1 if eye[axis]>plane else -1
            xyz[:,axis]=plane-int(sign>0)
            candidates,counts=np.unique(xyz,axis=0,return_counts=True)
            matches=set()
            for point in candidates[counts>=2]:
                cell=tuple(point.tolist())
                if cell[1]<0 or cell in cells or (strict and not neighboring(cell,axis,cells)):
                    continue
                if verify(rgb,atlas,mask,P,cell,axis,sign,native_mask=native,
                          threshold=.72 if strict or plane!=level else .60,minimum=32 if strict else 42):
                    matches.add(cell)
            for cell in matches:
                if plane==level or neighboring(cell,axis,matches) or neighboring(cell,axis,cells):
                    proposals.setdefault(cell,[]).append((axis,mask))
    if creases:
        verify_neighbors(cells,parts,image,rgb,P,eye,owner,atlas,proposals)
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
