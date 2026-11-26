"""Check whether observed pixels of a partial grid cell agree with a vanilla face."""
import numpy as np

from engine.visible_raster import SAMPLE_STRIDE


def verify(rgb,atlas,mask,P,cell,axis,sign,*,native_mask=None,threshold=.6,minimum=42):
    if atlas is None:return False
    free=[d for d in range(3) if d!=axis]
    u,v=np.meshgrid((np.arange(16)+.5)/16,(np.arange(16)+.5)/16)
    xyz=np.repeat(np.asarray(cell,float)[None,:],256,axis=0)
    xyz[:,axis]+=int(sign>0)
    xyz[:,free[0]]+=u.ravel()
    xyz[:,free[1]]+=v.ravel()
    p=xyz@P[:,:3].T+P[:,3]
    if np.min(p[:,2])<=1.:return False
    xy=p[:,:2]/p[:,2,None]
    if min(np.linalg.norm(xy[0]-xy[15]),np.linalg.norm(xy[0]-xy[240]))<9:return False
    x,y=np.floor(xy).astype(int).T
    h,w=rgb.shape[:2]
    h,w=min(h,mask.shape[0]*SAMPLE_STRIDE),min(w,mask.shape[1]*SAMPLE_STRIDE)
    inside=(x>=0)&(x<w)&(y>=0)&(y<h)
    idx=np.flatnonzero(inside)
    idx=idx[mask[y[idx]//SAMPLE_STRIDE,x[idx]//SAMPLE_STRIDE]]
    if native_mask is not None:
        idx=idx[native_mask[y[idx],x[idx]]]
    if len(idx)<minimum or len(np.unique(y[idx]*w+x[idx]))<20:return False
    patch=rgb[y[idx],x[idx]].astype(np.float32)
    color=patch.mean(0)
    gain=np.clip((atlas.means@color)/np.maximum(1,(atlas.means**2).sum(1)),.30,1.2)
    loss=abs(gain[:,None]*atlas.means-color).mean(1)+3*abs(np.log(gain/.8))
    short=np.argpartition(loss,24)[:24]
    ref=atlas.patterns[short].reshape(24,8,256)
    gray=patch.mean(1)
    def match(indices,selection):
        a=ref[:,:,indices[selection]]
        a=a-a.mean(2,keepdims=True)
        b=gray[selection]-gray[selection].mean()
        return (a@b)/np.maximum(1e-6,np.linalg.norm(a,axis=2)*np.linalg.norm(b))
    groups=((idx//16)//4+(idx%16)//4)%2
    if np.bincount(groups,minlength=2).min()<12 or gray.std()<.4:return False
    scores=np.minimum(match(idx,groups==0),match(idx,groups==1))-loss[short,None]/60
    tile,turn=np.unravel_index(scores.argmax(),scores.shape)
    if scores[tile,turn]<threshold:return False
    same=np.ones(len(idx),bool)
    offsets=[(idx//16)*16+(idx%16+4)%16,(idx+64)%256]
    return match(idx,same)[tile,turn] > max(match(o,same)[tile,turn] for o in offsets)+.12
