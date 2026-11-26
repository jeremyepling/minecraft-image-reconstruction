"""Estimate relative shade on image-derived planes from nearby patches with similar color and contrast."""
import numpy as np
from PIL import Image
from engine.geometry import bilinear, project, transform

STEP = 12

def _shift(a, y, x):
    result = np.roll(a, (y, x), axis=(0, 1))
    if y > 0: result[:y] = 0
    if y < 0: result[y:] = 0
    if x > 0: result[:, :x] = 0
    if x < 0: result[:, x:] = 0
    return result

def surface_shadow_map(rgb, proposals, plane_comparison=False):
    h, w = rgb.shape[:2]
    ny, nx = h // STEP, w // STEP
    cells = rgb[:ny*STEP,:nx*STEP].reshape(ny,STEP,nx,STEP,3).transpose(0,2,1,3,4)
    colors = np.median(cells,axis=(2,3)) + 1
    channel_sum = colors.sum(2,keepdims=True)
    chroma = colors/channel_sum
    luma = np.log(colors @ np.array([.213,.715,.072]))
    pixel_light = cells @ np.array([.213,.715,.072])
    lo, hi = np.quantile(pixel_light,[.2,.8],axis=(2,3))
    contrast = (hi-lo)/(hi+lo+2)
    ids = np.zeros((ny,nx),int)
    depths = np.zeros((ny,nx))
    normal_down=np.zeros_like(depths)
    best_coverage=np.zeros_like(depths)
    for index, proposal in enumerate(proposals,1):
        support = proposal['support']
        sw, sh = support['size']
        mask = np.zeros((sh,sw),dtype=bool)
        for y,a,b in support['runs']:
            mask[y,a:b]=True
        # A sample near the boundary often contains both background and surface.
        coverage = np.asarray(Image.fromarray(mask.astype('uint8')*255).resize((nx,ny),Image.Resampling.BOX))
        covered = coverage >= 210
        if plane_comparison:
            covered = (coverage >= 96) & (coverage > best_coverage)
            best_coverage[covered] = coverage[covered]
            normal_down[covered] = (proposal.get('normal_camera') or [0,0,0])[1]
        ids[covered] = index
        depths[covered] = proposal.get('relative_inverse_depth') or 0
    eave = np.zeros_like(ids,dtype=bool)
    for dy in (1,2,3,4):
        upper = _shift(depths,dy,0)
        eave |= (upper > depths+.035) & (_shift(ids,dy,0)!=ids) & (ids>0)
    offsets = [(y,x) for y in (-6,-3,0,3,6) for x in (-6,-3,0,3,6) if x or y]
    gains=[]
    for dy,dx in offsets:
        peer = _shift(ids,dy,dx)
        color_distance = np.linalg.norm(chroma-_shift(chroma,dy,dx),axis=2)
        similar = (ids>0)&(peer==ids)&(color_distance<.028)&(abs(contrast-_shift(contrast,dy,dx))<.055)
        gains.append(np.where(similar,_shift(luma,dy,dx)-luma,np.nan))
    gains=np.stack(gains)
    n=np.isfinite(gains).sum(0)
    # Robust upper quartile of comparable patches, requiring several independent samples.
    ordered=np.sort(np.where(np.isfinite(gains),gains,np.inf),axis=0)
    upper=np.take_along_axis(ordered,(np.maximum(0,n-1)*3//4)[None],axis=0)[0]
    delta=np.where(n>=4,np.clip(upper-.035,0,.40),0)
    confidence=np.minimum(1,n/12)*np.where(eave,.85,.60)
    loggain=delta*confidence
    if plane_comparison:
        near_front=np.zeros_like(eave)
        for dy in (1,2,4,8):
            for dx in (-2,0,2):
                other=_shift(ids,dy,dx)
                near_front |= (ids>0)&(other>0)&(other!=ids)&(_shift(depths,dy,dx)>depths+.025)
        # Camera Y points down. A steep positive Y component suggests a soffit.
        cue=np.where(normal_down>.52,.75,np.where(near_front,.60,0))
        cross=[]
        for dy in (-12,-6,-3,0,3,6,12):
            for dx in (-12,-6,-3,0,3,6,12):
                other=_shift(ids,dy,dx)
                similar=((ids>0)&(other>0)&(other!=ids)&(_shift(normal_down,dy,dx)<.52)
                         &(np.linalg.norm(chroma-_shift(chroma,dy,dx),axis=2)<.018)
                         &(abs(contrast-_shift(contrast,dy,dx))<.035))
                cross.append(np.where(similar,_shift(luma,dy,dx)-luma,np.inf))
        cross=np.sort(np.stack(cross),axis=0)
        count=np.isfinite(cross).sum(0)
        # Median of comparable peers protects against individual bright marks.
        med=np.take_along_axis(cross,(np.maximum(0,count-1)//2)[None],axis=0)[0]
        support=np.minimum(1,count/8)*cue
        boost=np.where(count>=3,np.clip(med-.04,0,.65)*support,0)
        confidence=np.maximum(confidence,np.where(boost>0,support,0))
        loggain=np.maximum(loggain,boost)
        eave=near_front
    return dict(loggain=loggain,confidence=confidence,eave=eave,ids=ids)

def cell_shadow(field, quad, size):
    t=(np.arange(size)+.5)/size
    xy=project(transform(quad),np.stack(np.meshgrid(t,t),axis=-1))/STEP
    center=np.floor(xy.mean(axis=(0,1))).astype(int)
    h,w=field['ids'].shape
    owner=field['ids'][np.clip(center[1],0,h-1),np.clip(center[0],0,w-1)]
    same=(field['ids']==owner)&(owner>0)
    sample=lambda arr: bilinear(arr[...,None],xy)[...,0]
    loggain=sample(field['loggain']*same)
    gain=np.exp(loggain)
    return gain,dict(mean_gain=float(gain.mean()),minimum_gain=float(gain.min()),maximum_gain=float(gain.max()),
                     horizontal_gradient=float(gain[:,-1].mean()-gain[:,0].mean()),
                     vertical_gradient=float(gain[-1].mean()-gain[0].mean()),
                     confidence=float(sample(field['confidence']*same).mean()),nearer_surface_above=float(sample((field['eave']&same).astype(float)).mean()))

def corrected(patch, gains):
    return np.clip(np.rint(patch*gains[...,None]),0,255).astype('uint8')

