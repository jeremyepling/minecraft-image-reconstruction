"""Geometry experiments for single screenshots. No material or camera inputs."""
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from engine.geometry import UNIT, bilinear, project, transform
from engine.propose import estimate_grid


def main_component(mask):
    work=mask.copy(); h,w=mask.shape
    best=[]
    for y,x in zip(*work.nonzero()):
        if not work[y,x]: continue
        group=[]; stack=[(int(y),int(x))]; work[y,x]=False
        while stack:
            y,x=stack.pop(); group.append((y,x))
            for ny,nx in ((y,x-1),(y,x+1),(y-1,x),(y+1,x)):
                if 0<=nx<w and 0<=ny<h and work[ny,nx]:
                    work[ny,nx]=False; stack.append((ny,nx))
        if len(group)>len(best): best=group
    out=np.zeros_like(mask)
    if best: out[tuple(np.array(best).T)]=True
    return out


def foreground(im):
    w,h=im.size; scale=min(1.,256/max(w,h))
    size=max(8,round(w*scale)),max(8,round(h*scale))
    small=im.convert('RGB').resize(size,Image.Resampling.BOX)
    a=np.asarray(small,dtype='f')
    # Learn several backdrop colors independently on each scanline: the sky,
    # terrain and floor seams can coexist on a single source image.
    n=max(3,size[0]//10)
    edges=np.concatenate((a[:,:n],a[:,-n:]),axis=1)
    refs=np.percentile(edges,[5,25,50,75,95],axis=1).astype('f')
    residual=np.full(a.shape[:2],np.inf)
    for ref in refs: residual=np.minimum(residual,np.abs(a-ref[:,None]).mean(2))
    raw=residual>10
    alpha=np.asarray(im.convert('RGBA').resize(size))[:,:,3]
    if alpha.min()<128: raw=alpha>=128
    smooth=Image.fromarray(np.uint8(255*raw)).filter(ImageFilter.MedianFilter(3))
    mask=main_component(np.array(smooth)>0)
    if mask.mean()<.008: mask[:]=True
    return a,mask


def robust_line(x,y):
    m,b=np.polyfit(x,y,1)
    good=abs(y-(m*x+b))<=max(1.5,np.percentile(abs(y-(m*x+b)),70))
    if good.sum()>2: m,b=np.polyfit(x[good],y[good],1)
    return m,b


def plane_fit(mask, region, angles, weights):
    x0,y0,x1,y1=region
    patch=mask[y0:y1,x0:x1]
    py,px=patch.nonzero()
    if len(px)<30: return None
    a,b=int(px.min()),int(px.max())+1
    if b-a<3 or py.max()-py.min()<2: return None
    patch=patch[:,a:b]; xs=np.where(patch.any(0))[0]
    top=patch.argmax(0); bot=patch.shape[0]-patch[::-1].argmax(0)
    u,v=robust_line(xs+.5,top[xs]),robust_line(xs+.5,bot[xs])
    x=np.arange(b-a)+.5
    upper=np.clip(np.polyval(u,x),0,y1-y0); lower=np.clip(np.polyval(v,x),0,y1-y0)
    ys=np.arange(patch.shape[0])[:,None]+.5
    predicted=(ys>=upper)&(ys<=lower)
    w=weights[y0:y1,x0+a:x0+b]*patch
    t=angles[y0:y1,x0+a:x0+b]
    mu=(w*t).sum()/max(1,w.sum())
    # One plane should explain both its silhouette and its horizontal edges.
    cost=np.count_nonzero(predicted^patch)+.7*np.sum(w*(t-mu)**2)
    q=np.array([[a,upper[0]],[b,upper[-1]],[b,lower[-1]],[a,lower[0]]])+[x0,y0]
    if min(q[2,1]-q[1,1],q[3,1]-q[0,1])<2: return None
    return dict(q=q,cost=cost,region=region)


def propose(image):
    a,mask=foreground(image)
    sy,sx=mask.shape; scale=np.array(image.size)/[sx,sy]
    gray=np.asarray(Image.fromarray(a.astype('uint8')).convert('L').filter(ImageFilter.GaussianBlur(1)),float)
    gy,gx=np.gradient(gray)
    t=np.clip(-gx/np.where(abs(gy)>.5,gy,1),-1,1)
    weight=np.minimum(4,np.abs(gy)/5)*(np.abs(gy)>np.abs(gx))
    y,x=mask.nonzero(); start=(x.min(),y.min(),x.max()+1,y.max()+1)
    root=plane_fit(mask,start,t,weight)
    leaves=[root] if root else []
    # Score horizontal and vertical subdivisions of the same parent. Extra
    # boundaries must explain enough area or change of line direction to pay
    # their complexity cost; a colored stripe alone gives little benefit.
    for _ in range(5):
        move=None
        for idx,node in enumerate(leaves):
            x0,y0,x1,y1=node['region']
            for axis,lo,hi in ((0,x0,x1),(1,y0,y1)):
                edges=mask[y0:y1,x0:x1].sum(axis=axis)
                changes=np.abs(np.diff(edges.astype(float)))
                ticks=np.unique(np.r_[np.linspace(lo+3,hi-3,9,dtype=int),
                                      lo+1+changes.argsort()[-8:]])
                for cut in ticks:
                    if min(cut-lo,hi-cut)<3: continue
                    a1=[x0,y0,x1,y1]; a2=a1.copy();a1[2+axis]=cut;a2[axis]=cut
                    left=plane_fit(mask,a1,t,weight);right=plane_fit(mask,a2,t,weight)
                    if left is None or right is None: continue
                    gain=node['cost']-left['cost']-right['cost']-(15+.014*mask.sum())
                    if gain>0 and (move is None or gain>move[0]): move=(gain,idx,left,right)
        if move is None: break
        _,idx,l,r=move;leaves[idx:idx+1]=[l,r]
    return [p['q']*scale for p in leaves],mask,scale


def repeat_grid(rgb,q):
    initial=estimate_grid(rgb,q)
    size=np.linalg.norm(q-np.roll(q,1,axis=0),axis=1)
    w,h=np.maximum(5,np.minimum(360,[size[[1,3]].mean(),size[[0,2]].mean()])).astype(int)
    gx,gy=np.meshgrid((np.arange(w)+.5)/w,(np.arange(h)+.5)/h)
    tile=bilinear(rgb,project(transform(q),np.stack([gx,gy],-1))).mean(2)
    result=[]
    for axis,(prior,_) in enumerate(initial):
        lines=tile if axis==0 else tile.T
        # Normalize the high-pass signal per scanline. A repeating texture
        # supplies a pitch even when its two neighbors have the same material.
        detail=np.diff(lines,axis=1)
        detail=np.clip(detail,-30,30); n=detail.shape[1]
        fft=np.fft.rfft(detail,n=2*n,axis=1)
        corr=np.fft.irfft((fft*np.conjugate(fft)).mean(0),n=2*n)[:n]/np.arange(n,0,-1)
        corr=corr/max(corr[0],.01)
        choices=[]
        for count in range(2,min(100,n//5)+1):
            p=(n+1)/count
            lags=np.array([p*.77,p,p*1.23])
            flank,peak,other=np.interp(lags,np.arange(n),corr)
            specificity=peak-(flank+other)*.5
            score=specificity+.20*peak+.035*(count==prior)
            choices.append((score,count))
        score,count=max(choices,default=(0,prior))
        result.append(count if score>.18 else prior)
    return result


def candidate(path):
    with Image.open(path) as src: image=src.convert('RGB')
    rgb=np.asarray(image)
    quads,mask,scale=propose(image)
    planes=[]; cells=[]
    for i,q in enumerate(quads):
        cols,rows=repeat_grid(rgb,q)
        h=transform(q)@np.diag([1/cols,1/rows,1])
        plane=dict(name=str(i),quad=q.tolist(),columns=cols,rows=rows,grid_to_image=h.tolist())
        planes.append(plane)
        for x in range(cols):
            for y in range(rows):
                quad=project(h,UNIT+[x,y])
                px,py=(quad.mean(0)/scale).astype(int)
                if 0<=px<mask.shape[1] and 0<=py<mask.shape[0] and mask[py,px]:
                    cells.append(dict(plane=str(i),quad=quad.tolist(),ij=[x,y]))
    return planes,cells
