"""Measure a common image lattice and rasterize occupied front-elevation cells."""
import numpy as np
from PIL import Image
from engine.geometry import UNIT, project
from engine.local_pitch import estimate_pitch
from engine.texel_grid import structure_parts, texel_grid


def structure_mask(image):
    if image.mode=='RGBA' and image.getchannel('A').getextrema()[0]<128:
        return np.asarray(image.getchannel('A'))>=128
    parts=structure_parts(image.convert('RGB'))
    masks=[np.asarray(p.getchannel('A'))>=128 for p in parts if p.mode=='RGBA']
    return np.logical_or.reduce(masks) if masks else np.ones((image.height,image.width),bool)


def texture_slopes(rgb,mask,spacing):
    """Measure the transverse shift at one block period using image pixels only."""
    gray=rgb.astype(np.float32).mean(axis=2)
    detail=gray-(np.roll(gray,2,0)+np.roll(gray,-2,0)+np.roll(gray,2,1)+np.roll(gray,-2,1))/4
    y,x=np.nonzero(mask[3:-3:3,3:-3:3])
    y,x=y*3+3,x*3+3
    step=max(1,len(x)//6000)
    y,x=y[::step],x[::step]
    slopes=[]
    for axis,p in enumerate(spacing):
        scores=[]
        for offset in range(-round(.46*p),round(.46*p)+1):
            dx,dy=(round(p),offset) if axis==0 else (offset,round(p))
            xx,yy=x+dx,y+dy
            ok=(xx>=0)&(yy>=0)&(xx<mask.shape[1])&(yy<mask.shape[0])
            ids=np.flatnonzero(ok)
            ids=ids[mask[yy[ids],xx[ids]]]
            a=detail[y[ids],x[ids]]
            b=detail[yy[ids],xx[ids]]
            den=np.linalg.norm(a)*np.linalg.norm(b)
            scores.append((float(a@b)/max(1,den),offset))
        strength,offset=max(scores)
        slopes.append(offset/p if strength>.22 else 0.)
    return slopes


def axis_fit(samples,low):
    """Snap phase and refine pitch using long mask boundaries in block-space."""
    if len(samples)==0:
        return 1.,low
    scores=[]
    for period in np.linspace(.94,1.06,25):
        circle=np.exp(2j*np.pi*samples/period).mean()
        strength=abs(circle)
        phase=np.angle(circle)*period/(2*np.pi)
        score=strength-2*abs(np.log(period))
        scores.append((score,period,phase))
    strength,period,phase=max(scores)
    if strength<.20:
        return 1.,low
    origin=phase+np.round((low-phase)/period)*period
    # Retain partial cells crossed by a silhouette edge.
    if origin-low>.38*period:
        origin-=period
    return period,origin


def front_transform(rgb,mask,spacing,fit,calibration):
    """Map front-plane block coordinates to image pixels, increasing downward."""
    if calibration:
        f=calibration['focal_px']
        cy,cx=np.array(mask.shape)/2
        ground=float(np.nonzero(mask)[0].max()+1)
        horizon=calibration['horizon_y']
        pitch=(ground-horizon)*calibration['pitch_slope']
        if pitch>3:
            angle=np.arctan((cy-horizon)/f)
            sine,cosine=np.sin(angle),np.cos(angle)
            depth=f/pitch
            return np.array([[f,cx*sine,cx*depth],
                             [0,cy*sine+f*cosine,ground*depth],
                             [0,sine,depth]])
    px,py=spacing
    sx,sy=texture_slopes(rgb,mask,spacing) if fit else (0.,0.)
    return np.array([[px,sy*py,0],[sx*px,py,0],[0,0,1.]])


def elevation(rgb,mask,spacing,fit=True,calibration=None):
    if not mask.any():
        raise ValueError('The image contains no visible structure')
    basis=front_transform(rgb,mask,spacing,fit,calibration)
    inv=np.linalg.inv(basis)
    # Convert the outline to block units. Points on vertical and horizontal
    # silhouette transitions support the respective lattice phases.
    xedge=np.diff(np.pad(mask.astype(np.int8),((0,0),(1,1))),axis=1)
    yedge=np.diff(np.pad(mask.astype(np.int8),((1,1),(0,0))),axis=0)
    def edges(edge,offset):
        y,x=edge.nonzero()
        return project(inv,np.stack((x,y),axis=1)+offset)
    vx=edges(xedge,[0,.5])
    vy=edges(yedge,[.5,0])
    outline=np.concatenate([vx,vy])
    low,high=outline.min(axis=0),outline.max(axis=0)
    scales,origins=zip(*(axis_fit(s,lo) if fit else (1.,lo)
                         for s,lo in zip((vx[:,0],vy[:,1]),low)))
    counts=np.maximum(1,np.ceil((high-origins)/scales-.12)).astype(int)
    cols,rows=map(int,counts)
    if cols*rows>100000:
        raise ValueError('Estimated lattice exceeds 100,000 visible cells')
    h=basis@np.array([[scales[0],0,origins[0]],[0,scales[1],origins[1]],[0,0,1]])
    u,v=np.meshgrid((np.arange(7)+.5)/7,(np.arange(7)+.5)/7)
    probes=np.stack([u.ravel(),v.ravel()],axis=1)
    cells=[]
    for j in range(rows):
        indices=np.column_stack([np.arange(cols),np.full(cols,j)])
        sample=np.floor(project(h,indices[:,None,:]+probes)).astype(int)
        x,y=sample[:,:,0],sample[:,:,1]
        valid=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])
        occupied=valid & mask[y.clip(0,mask.shape[0]-1),x.clip(0,mask.shape[1]-1)]
        for i in np.flatnonzero(occupied.mean(1)>=.50):
            i=int(i)
            cells.append(dict(plane='elevation',ij=[i,j],xyz=[i,rows-1-j,0],facing='south',
                              quad=project(h,UNIT+[i,j]).tolist()))
    plane=dict(name='elevation',quad=project(h,UNIT*counts).tolist(),grid_to_image=h.tolist(),
               columns=cols,rows=rows,facing='south',origin=[0,0,0],spacing_method='local repetition',
               depth_status='unresolved',location_basis='common projected elevation')
    if calibration:
        plane.update(floor_calibration=calibration,spacing_method='perspective ground grid')
    if any(edge.any() for edge in (mask[0],mask[-1],mask[:,0],mask[:,-1])):
        plane['extent_status']='foreground reaches frame boundary; whole-building size unresolved'
    return [plane],cells


def candidate(image,fit=True,ground=False):
    rgb=np.asarray(image.convert('RGB'))
    mask=structure_mask(image)
    if not mask.any():
        raise ValueError('The image contains no visible structure')
    spacing=estimate_pitch(rgb,mask)
    if spacing is None:
        y,x=mask.nonzero()
        lo=np.array([x.min(),y.min()]); size=np.array([x.max()+1,y.max()+1])-lo
        spacing=size/np.array(texel_grid(rgb,lo+UNIT*size))
    if ground:
        from engine.depth_layers import floor_calibration
        calibration=floor_calibration(rgb)
    else:
        calibration=None
    return elevation(rgb,mask,spacing,fit,calibration)
