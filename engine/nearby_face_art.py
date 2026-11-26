"""Measure vanilla tile phase on observed parts of each projected candidate face."""
import numpy as np

from engine.camera_atlas_floor import _atlas
from engine.nearby_face_decision import _project


def scores(points, axis, rgb, P, eye, mask, atlas):
    # Interleave texels across the tile. Sample at their native projected centers.
    idx=np.arange(0,256,4)+(np.arange(64)//4)%4
    uv=(np.stack((idx%16,idx//16),1)+.5)/16
    free=[d for d in range(3) if d!=axis]
    xyz=np.repeat(points[:,None,:],len(uv),axis=1).astype(float)
    xyz[:,:,axis]+=(eye[axis]>points[:,axis]+1)[:,None]
    xyz[:,:,free]+=uv
    xy,z=_project(P,xyz)
    h,w=mask.shape
    x,y=np.floor(xy).astype(int).transpose(2,0,1)
    valid=(x>=0)&(y>=0)&(x<4*w)&(y<4*h)&(z>1)
    valid &= ((eye[axis]<points[:,axis])|(eye[axis]>points[:,axis]+1))[:,None]
    valid &= mask[(y//4).clip(0,h-1),(x//4).clip(0,w-1)]
    arr=rgb[y.clip(0,rgb.shape[0]-1),x.clip(0,rgb.shape[1]-1)]
    count=np.maximum(1,valid.sum(1))
    mean=(arr*valid[:,:,None]).sum(1)/count[:,None]
    gain=np.clip((mean@atlas.means.T)/np.maximum(1,(atlas.means**2).sum(1)),.30,1.2)
    loss=abs(mean[:,None]-gain[:,:,None]*atlas.means).mean(2)+3*abs(np.log(gain/.80))
    short=np.argpartition(loss,12,axis=1)[:,:12]
    patterns=atlas.patterns[short].reshape(len(points),12,8,256)
    gray=arr.mean(2)
    gray=(gray-(gray*valid).sum(1)[:,None]/count[:,None])*valid
    norm=np.sqrt((gray**2).sum(1))
    def corr(ids):
        ref=patterns[:,:,:,ids]
        ref=ref-(ref*valid[:,None,None]).sum(-1,keepdims=True)/count[:,None,None,None]
        num=np.einsum('naod,nd->nao',ref,gray)
        den=np.sqrt(np.einsum('naod,nd->nao',ref**2,valid))*norm[:,None,None]
        return num/np.maximum(1,den)
    actual=corr(idx)
    penalized=actual-np.take_along_axis(loss,short,axis=1)[:,:,None]/60
    choice=penalized.reshape(len(points),-1).argmax(1)
    take=np.arange(len(points)),choice//8,choice%8
    rival=corr((idx//16)*16+(idx%16+4)%16)[take]
    rival=np.maximum(rival,corr((idx+64)%256)[take])
    enough=(count>=9)&(norm>2)
    return np.column_stack([np.where(enough,penalized[take],0),np.where(enough,actual[take]-rival,0)])


def enrich(image, parts, camera, data, *, atlas_path=None):
    from engine.visible_horizontal_faces import _surface_masks
    from engine.surface_model import intrinsic
    if not len(data['edges']):
        return data
    atlas=_atlas('26.3',atlas_path)
    if atlas is None:
        return None
    points=data['points']
    P=np.asarray(camera['P'],float)
    P=P/np.linalg.norm(P[2,:3])
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    focal=np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
    rot=np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    _,masks=_surface_masks(parts,image.size,np.arange(2,image.width,4),np.arange(2,image.height,4),rot)
    rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
    art=np.stack([np.concatenate([scores(points[i:i+48],a,rgb,P,eye,masks[a],atlas)
                                  for i in range(0,len(points),48)]) for a in range(3)],axis=1)
    src,dst,axis=data['edges'].T
    x=data['features']
    return dict(data,features=np.column_stack([x,art[src,axis],art[dst,axis],art[dst,axis]-art[src,axis],
                                               art[src].max(1),art[dst].max(1)]).astype(np.float32))

