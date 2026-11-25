"""Infer a block pitch from equally spaced texel transitions on a rectified vanilla face."""
import numpy as np
from PIL import Image, ImageFilter
from engine.geometry import project, transform, UNIT, bilinear
from engine.plane_candidates import repeat_grid, propose, main_component


def texel_grid(rgb,q):
    q=np.asarray(q)
    fallback=repeat_grid(rgb,q)
    lengths=np.linalg.norm(q-np.roll(q,1,axis=0),axis=1)
    dims=np.maximum(5,np.ceil([lengths[[1,3]].mean(),lengths[[0,2]].mean()])).astype(int)
    result=[]
    for axis,(length,prior) in enumerate(zip(dims,fallback)):
        # Sample along multiple lines so texture on one row cannot set the
        # block size. At four samples / source pixel edge positions survive
        # rectification, including at subpixel offsets.
        count=min(4096,4*length)
        uv=np.stack(np.meshgrid((np.arange(count)+.5)/count,np.linspace(.08,.92,28)),axis=-1)
        if axis==1: uv=uv[:,:,::-1]
        colors=bilinear(rgb,project(transform(q),uv))
        jumps=np.abs(np.diff(colors,axis=1)).mean(2)
        jumps=np.minimum(jumps,np.percentile(jumps,95,axis=1,keepdims=True))
        # Equal texel spacing concentrates edge energy at sixteen cycles per
        # block. Half a brick course may repeat its color pattern but does
        # not double this pixel frequency. Require independent row support.
        spec=np.abs(np.fft.rfft(jumps,n=count,axis=1))
        dc=np.maximum(spec[:,0],.01)
        candidates=[]
        for n in range(1,min(128,int(length/19),count//32-1)+1):
            f=16*n
            peak=spec[:,f]/dc
            # Compare with spectral leakage and nearby non-texel frequencies.
            nearby=spec[:,[f-3,f-2,f+2,f+3]].mean(1)/dc
            strength=np.quantile(peak-nearby,.65)
            candidates.append((strength,n))
        strength,n=max(candidates,default=(0,prior))
        result.append(n if strength>.045 else prior)
    return result


def structure_parts(image):
    # Isolated walls can belong to the same scene. Keep substantial components
    # individually so a second structure cannot be mistaken for an opening.
    size=np.maximum(8,np.round(np.array(image.size)*min(1,256/max(image.size)))).astype(int)
    a=np.asarray(image.resize(tuple(size),Image.Resampling.BOX),float)
    n=max(3,size[0]//10)
    refs=np.percentile(np.concatenate([a[:,:n],a[:,-n:]],axis=1),[5,25,50,75,95],axis=1)
    residual=np.minimum.reduce([abs(a-r[:,None]).mean(2) for r in refs])
    raw=Image.fromarray(np.uint8((residual>10)*255)).filter(ImageFilter.MedianFilter(3))
    remaining=np.asarray(raw.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3)))>0
    parts=[]; limit=remaining.size*.007
    for _ in range(4):
        mask=main_component(remaining)
        if mask.sum()<limit: break
        limit=max(limit,.15*mask.sum()); remaining=remaining & ~mask
        part=image.convert('RGBA')
        part.putalpha(Image.fromarray(np.uint8(mask*255)).resize(image.size,Image.Resampling.NEAREST))
        parts.append(part)
    return parts or [image]


def candidate(path):
    with Image.open(path) as im: image=im.convert('RGB')
    rgb=np.asarray(image)
    planes=[];cells=[]
    for part in structure_parts(image):
        quads,mask,scale=propose(part)
        for q in quads:
            name=str(len(planes));cols,rows=texel_grid(rgb,q)
            h=transform(q)@np.diag([1/cols,1/rows,1])
            planes.append(dict(name=name,quad=q.tolist(),columns=cols,rows=rows,grid_to_image=h.tolist()))
            for x in range(cols):
                for y in range(rows):
                    cq=project(h,UNIT+[x,y]);px,py=(cq.mean(0)/scale).astype(int)
                    if 0<=px<mask.shape[1] and 0<=py<mask.shape[0] and mask[py,px]:
                        cells.append(dict(plane=name,quad=cq.tolist(),ij=[x,y]))
    return planes,cells
