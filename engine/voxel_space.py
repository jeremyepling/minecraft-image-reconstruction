"""Image-derived, ground-anchored voxel grid and a forward renderer for block models."""
from itertools import product

import numpy as np
from PIL import Image, ImageDraw

from benchmark.oblique_recess import pavement
from engine.line_outline import cube_outline, silhouette
from engine.structural_lines import detect


def project(P, xyz):
    homogeneous = np.asarray(xyz) @ P[:,:3].T + P[:,3]
    return homogeneous[...,:2] / homogeneous[...,2,None]


def prepare(image, allow_relative=False):
    image = image.convert('RGB')
    rgb = np.asarray(image)
    lines = detect(image)
    refined, evidence = cube_outline(image,lines,silhouette(image,lines))
    mask = np.asarray(refined)>0
    camera = pavement(rgb)
    metric = camera is not None
    if metric:
        try:
            return _ground_scene(image,refined,mask,camera)
        except ValueError:
            if not allow_relative:
                raise
    if allow_relative:
        from engine.voxel_relative import relative_scene
        return relative_scene(image,refined,evidence['axes'])
    raise ValueError('No supported metric ground grid')


def _ground_scene(image,refined,mask,camera):
    K,_,directions,up,_,height,offsets,_ = camera
    xs = np.flatnonzero(mask.any(0))[::3]
    if len(xs)<5:
        raise ValueError('Too little visible foreground')
    ys = mask.shape[0]-mask[::-1,xs].argmax(0)
    rays = np.column_stack((xs,ys,np.ones(len(xs)))) @ np.linalg.inv(K).T
    dot = rays@up
    valid = dot < -.022
    if valid.sum()<5:
        raise ValueError('Ground contacts are outside the visible crop')
    floor = -height*rays[valid]/dot[valid,None]
    # Choose the mostly image-horizontal ground axis as X. Z increases into
    # the picture, X towards image right, and Y up, without compass metadata.
    zk = int(np.argmax([abs(d[2]) for d in directions]))
    xk = 1-zk
    sx = 1 if directions[xk][0]>=0 else -1
    sz = 1 if directions[zk][2]>=0 else -1
    basis = np.stack((sx*directions[xk],up,sz*directions[zk]),axis=1)
    phases = np.array([sx*offsets[xk],sz*offsets[zk]])
    contact = floor@basis[:,[0,2]]
    near = np.quantile(contact[:,1],.12)
    contact = contact[contact[:,1] <= near+min(32,max(6,.63*abs(near)))]
    low,high = np.quantile(contact,[.015,.985],axis=0)
    anchor = np.floor(low-phases+.35)+phases-1
    span = np.ceil(high-anchor+.7).astype(int)
    span[1] = max(span[1]+1,4)
    if max(span)>80 or min(span)<1:
        raise ValueError('No bounded foreground footprint')
    origin = basis@np.array([anchor[0],-height,anchor[1]])
    P = K@np.column_stack((basis,origin))
    # Intersect upper silhouette rays with the vertical middle of the footprint.
    top = mask[:,xs].argmax(0)
    r = np.column_stack((xs,top,np.ones(len(xs)))) @ np.linalg.inv(K).T
    rz = r@basis[:,2]
    good = rz>.1
    if not good.any():
        raise ValueError('Facade direction unresolved')
    y = height+(r[good]@up)*(anchor[1]+.5*span[1])/rz[good]
    ny = int(np.ceil(np.quantile(y,.985)))+3
    if not 2<=ny<=80:
        raise ValueError('No bounded vertical extent')
    shape = (int(span[0]),ny,int(span[1]))
    size = tuple(max(1,round(n*min(1.,320/max(image.size)))) for n in image.size)
    P = np.diag([size[0]/image.width,size[1]/image.height,1.])@P
    small_mask = np.asarray(refined.resize(size,Image.Resampling.NEAREST))>0
    hull = silhouette_volume(P,shape,small_mask)
    return dict(P=P,mask=small_mask,hull=hull,image=np.asarray(image.resize(size,Image.Resampling.BOX)),
                metric=np.array(True),source_size=np.array(image.size))


def silhouette_volume(P, shape, mask):
    ids = np.indices(shape).reshape(3,-1).T
    support = np.zeros(len(ids),int)
    height,width = mask.shape
    for offset in [(.5,.5,.5), *product((.15,.85),repeat=3)]:
        camera = (ids+offset)@P[:,:3].T+P[:,3]
        visible = camera[:,2]>1e-8
        xy = camera[:,:2]/np.where(visible,camera[:,2],1)[:,None]
        x,y = np.floor(xy).astype(int).T
        valid = visible & (x>=0)&(x<width)&(y>=0)&(y<height)
        support[valid] += mask[y[valid],x[valid]]
    hull = (support>=6).reshape(shape)
    if not hull.any():
        raise ValueError('No cubes intersect the detected building')
    return hull


def render(occupied,P,size,shade=False):
    """Rasterize exposed, camera-facing unit faces, ordered far to near."""
    im = Image.new('RGB' if shade else 'L',size,(36,47,60) if shade else 0)
    draw = ImageDraw.Draw(im)
    camera = -np.linalg.solve(P[:,:3],P[:,3])
    faces,tones,distances=[],[],[]
    for axis in range(3):
        u,v = (axis+1)%3,(axis+2)%3
        for sign in (-1,1):
            neighbor = np.roll(occupied,-sign,axis=axis)
            boundary = [slice(None)]*3
            boundary[axis] = -1 if sign>0 else 0
            neighbor[tuple(boundary)] = False
            pts = np.argwhere(occupied&~neighbor).astype(float)
            pts[:,axis] += int(sign>0)
            pts = pts[(camera[axis]-pts[:,axis])*sign>0]
            if not len(pts):
                continue
            quads = np.repeat(pts[:,None,:],4,axis=1)
            quads[:,[1,2],u] += 1
            quads[:,[2,3],v] += 1
            faces.extend(project(P,quads))
            tones.extend([((126,162,175),(191,212,213),(151,184,192))[axis]]*len(pts))
            distances.extend(quads.mean(1)@P[2,:3]+P[2,3])
    for index in np.argsort(distances)[::-1]:
        draw.polygon([tuple(p) for p in faces[index]],fill=tones[index] if shade else 255)
    return im
