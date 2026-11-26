"""An explicitly unscaled search volume for a cropped single-view sketch."""
from engine.voxel_space import silhouette_volume

import numpy as np
from PIL import Image


def relative_scene(image, refined_mask, axes, resolution=36, depth_fraction=.35):
    """Clip silhouette rays to a finite slab parallel to the main facade.

    Perspective and orientation come from three fitted Manhattan vanishing
    points. The median visible plane has arbitrary distance one. Its longest
    projected span defines the numerical grid resolution. The depth budget is
    an explicit prior; all coordinates are sketch cells rather than blocks.
    """
    if len(axes) != 3:
        raise ValueError('Relative sketch requires three supported axis directions')
    if not (8 <= resolution <= 80 and .05 <= depth_fraction <= 1.):
        raise ValueError('Relative resolution or depth budget outside supported range')
    w,h = image.size
    v = np.array(axes, float)
    center = np.array([w,h])/2
    xy = v[:,:2]-center*v[:,2,None]
    focal_sq = [-xy[i]@xy[j]/(v[i,2]*v[j,2]) for i,j in ((0,1),(1,2),(2,0))
                if abs(v[i,2]*v[j,2]) > 1.e-8]
    valid = [f for f in focal_sq if (.18*w)**2<f<(3*w)**2]
    if not valid:
        raise ValueError('Cannot infer a finite perspective camera')
    focal = np.sqrt(np.median(valid))
    K = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1.]])
    dirs = np.column_stack((xy/focal,v[:,2]))
    dirs /= np.linalg.norm(dirs,axis=1,keepdims=True)
    vertical = int(np.argmax(abs(dirs[:,1])))
    up = dirs[vertical]*(-1 if dirs[vertical,1]>0 else 1)
    horizontal = np.delete(dirs,vertical,axis=0)
    depth_axis = int(np.argmax(abs(horizontal[:,2])))
    depth = horizontal[depth_axis] * np.sign(horizontal[depth_axis,2])
    right = horizontal[1-depth_axis] * np.sign(horizontal[1-depth_axis,0])
    basis = np.column_stack((right,up,depth))

    # Resolve the range from foreground rays, independently of image boundaries
    # and the position of the horizon. Never extrapolate below the crop.
    size = tuple(max(1,round(n*min(1.,320/max(image.size)))) for n in image.size)
    mask = np.asarray(refined_mask.resize(size,Image.Resampling.NEAREST))>0
    K = np.diag([size[0]/w,size[1]/h,1.])@K
    y,x = np.nonzero(mask)
    uv = np.column_stack((x+.5,y+.5,np.ones(len(x))))
    rays = uv@np.linalg.inv(K).T
    denom = rays@depth
    if len(denom)<32 or (denom>.15).mean()<.99:
        raise ValueError('Foreground does not support a bounded facade sketch')
    rays = rays[denom>.15]/denom[denom>.15,None]
    plane = rays@basis
    low,high = np.quantile(plane[:,:2],[.001,.999],axis=0)
    unit = max(high-low)/resolution
    if unit<=0:
        raise ValueError('Degenerate sketch extent')
    thickness = min(unit*resolution*depth_fraction,1.4)
    z = np.array([1.-thickness/2,1.+thickness/2])
    xy_lo = np.min(low[None]*z[:,None],axis=0)
    xy_hi = np.max(high[None]*z[:,None],axis=0)
    lo = np.r_[xy_lo-unit, z[0]]
    hi = np.r_[xy_hi+unit, z[1]]
    shape = np.ceil((hi-lo)/unit).astype(int)
    if max(shape)>3*resolution or min(shape)<1:
        raise ValueError('Facade is too oblique for the relative depth budget')
    P = K@np.column_stack((basis*unit,basis@lo))
    hull = silhouette_volume(P,tuple(shape),mask)
    return dict(P=P,mask=mask,hull=hull,image=np.asarray(image.convert('RGB').resize(size,Image.Resampling.BOX)),
                metric=np.array(False),source_size=np.array(image.size),
                sketch_resolution=np.array(resolution), depth_fraction=np.array(depth_fraction))
