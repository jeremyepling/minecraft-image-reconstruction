"""Dense atlas search on independent metric wall charts using only source image evidence."""
import numpy as np
import cv2
from benchmark.full_block_lattice import rectified_sprite_hits
from engine.geometry import project, transform, UNIT
from engine.surface_model import intrinsic


def windows(mask):
    h,w = mask.shape
    def starts(n):
        return sorted(set([*range(0,max(0,n-92),72),max(0,n-92)]))
    for y in starts(h):
        for x in starts(w):
            sub=mask[y:y+92,x:x+92]
            if sub.sum()>=100:
                yield x,y,sub


def metric_patch(rgb, mask, basis):
    yy,xx=np.nonzero(mask)
    if not len(xx):
        return
    xy=(np.column_stack((xx,yy))+.5)*4
    inv=np.linalg.inv(basis)
    unit=project(inv,xy)
    if not np.isfinite(unit).all():
        return
    low,high=unit.min(0),unit.max(0)
    span=high-low
    if min(span)<1.e-6:
        return
    # Preserve local texture resolution, bounding large regions to a 512-pixel search.
    centre=np.median(unit,axis=0)
    base=project(basis,centre)
    delta=span.min()*.001
    scale=max(np.linalg.norm(project(basis,centre+step*delta)-base)/delta for step in np.eye(2))
    scale=min(scale,510/span.max())
    shape=(span*scale+2).astype(int)
    if min(shape)<9:
        return
    H=basis @ np.array([[1/scale,0,low[0]],[0,1/scale,low[1]],[0,0,1]])
    x,y=np.meshgrid(np.arange(shape[0])+.5,np.arange(shape[1])+.5)
    xy=project(H,np.stack([x,y],axis=-1)).astype('f')
    crop=cv2.remap(rgb,xy[:,:,0]-.5,xy[:,:,1]-.5,cv2.INTER_LINEAR)
    valid=cv2.remap(mask.astype('B'),xy[:,:,0]/4-.5,xy[:,:,1]/4-.5,cv2.INTER_NEAREST)
    return H,crop,valid


def tangent_bases(q, normal, size, aspect, vert):
    # The local upward pencil only measures the orientation of photographed verticals.
    # Per-wall depth, translation, and scale remain independent.
    K=intrinsic(size,size[1])
    inv=np.linalg.inv(K)
    n=np.array(normal,dtype=float)
    n/=np.linalg.norm(n)
    up=inv@vert
    up-=n.dot(up)*n
    up/=max(1.e-9,np.linalg.norm(up))
    right=np.cross(n,up)
    center=q.mean(0)
    center_h=np.r_[center,1.]
    if (K@right)[0] - center[0]*right[2]<0:
        right=-right
    if (K@up)[1] - center[1]*up[2]<0:
        up=-up
    yield np.column_stack((K@right,K@up,center_h))
    if aspect:
        yield transform(q)@np.diag([1/aspect,1,1])


def vertical_pencil(lines,size):
    # Fit the native vertical edge family only. Near-parallel lines vote at infinity.
    ends=np.array(lines).reshape(-1,2,2)
    d=ends[:,1]-ends[:,0]
    keep=abs(d[:,0])<.25*abs(d[:,1])
    ends,d=ends[keep],d[keep]
    if len(ends)<3:
        return np.array([0.,1.,0.])
    center=np.array(size)/2
    p=(ends-center)/size[1]
    a,b=p[:,0],p[:,1]
    eq=np.cross(np.column_stack((a,np.ones(len(a)))),np.column_stack((b,np.ones(len(b)))))
    length=np.linalg.norm(eq[:,:2],axis=1)
    eq/=length[:,None]
    pairs=np.triu_indices(len(eq),1)
    dirs=np.cross(eq[pairs[0]],eq[pairs[1]])
    dirs=np.vstack((dirs,[0,1,0]))
    mid=p.mean(1)
    rays=dirs[:,None,:2]-mid*dirs[:,None,2:]
    residual=abs(dirs@eq.T)/np.maximum(1.e-9,np.linalg.norm(rays,axis=2))
    votes=np.maximum(0,1-(residual/.026)**2)
    best=np.argmax(votes@length)
    good=votes[best]>.5
    if sum(good)<3:
        return np.array([0.,1.,0.])
    _,_,v=np.linalg.svd(eq[good]*np.sqrt(length[good,None]))
    axis=v[-1]
    return np.r_[axis[:2]*size[1]+center*axis[2],axis[2]]


def dense_charts(rgb,q,mask,normal,aspect,vert):
    from benchmark.relative_source_charts import grow
    seen=[]
    for basis in tangent_bases(q,normal,rgb.shape[1::-1],aspect,vert):
        patch=metric_patch(rgb,mask,basis)
        if patch is None:
            continue
        H,crop,valid=patch
        for x,y,sub in windows(valid):
            origin=H@np.array([[1,0,x],[0,1,y],[0,0,1]])
            for seed in rectified_sprite_hits(crop[y:y+92,x:x+92],sub,origin,rgb):
                corners=np.asarray(seed['quad'])
                center=corners.mean(0)
                if any(np.linalg.norm(center-old)<2 for old in seen):
                    continue
                seen.append(center)
                h_up=transform(corners[[3,2,1,0]])
                cells=grow(rgb,h_up,mask)
                if len(cells)>=2 or (cells and cells[0]['confidence']>=.83):
                    yield h_up,cells
