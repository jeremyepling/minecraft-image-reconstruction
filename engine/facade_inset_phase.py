"""Find a photographed return and recessed panel from competing wall texture grids."""
import numpy as np
from PIL import ImageFilter

from engine.diagonal_texture_join import texture_fit
from engine.geometry import project
from engine.pillar_seams import orientation
from engine.plane_consensus import with_mask
from engine.surface_perimeter import _native_alpha


def host_grid(image, part, camera):
    patch = _native_alpha(part['support'], image.size)
    if patch is None:
        return None
    alpha, box = patch
    mask = np.asarray(alpha, bool)
    if mask.sum() < 5000:
        return None
    p, rot = orientation(camera, image.size)
    axis = int(np.argmax(abs(rot.T @ part['normal_camera'])))
    if axis == 1:
        return None
    inv = np.linalg.inv(p[:,:3])
    eye = -inv @ p[:,3]
    ray = inv @ [image.width * .5, image.height * .72, 1.]
    if abs(ray[1]) < 1.e-8 or ray[1] * eye[1] >= 0:
        return None
    anchor = round((eye-ray*(eye[1]/ray[1]))[axis])
    fit = texture_fit(np.asarray(image), mask, box, axis, eye, inv, range(anchor-24,anchor+25))
    if not fit or fit['corr'] < .65 or fit['margin'] < .08:
        return None
    y, x = np.nonzero(mask)
    x, y = x+box[0], y+box[1]
    rays = np.stack((x+.5, y+.5, np.ones(len(y))), axis=1) @ inv.T
    if np.min(abs(rays[:,axis])) < 1.e-7:
        return None
    free = [2-axis, 1]
    plane = fit['level']
    ratio = rays[:,free] / rays[:,axis,None]
    uv = eye[free] + (plane-eye[axis])*ratio
    if np.max(np.ptp(uv,axis=0)) > 80:
        return None
    return dict(p=p,rot=rot,axis=axis,eye=eye,fit=fit,free=free,
                x=x,y=y,uv=uv,ratio=ratio,plane=plane,box=box,mask=mask)


def texel_bins(uv):
    bins = np.floor(16*uv).astype(np.int32) % 16
    return 16*bins[:,1]+bins[:,0]


def templates(gray, uv):
    # Median per texel, across different independently observed blocks.
    code = texel_bins(uv)
    table = np.zeros(256)
    groups = np.argsort(code)
    pos = np.searchsorted(code[groups], np.arange(257))
    for k in range(256):
        sel = groups[pos[k]:pos[k+1]]
        if len(sel):
            table[k] = np.median(gray[sel])
    return table


def correlation(a, b):
    if len(a)<12:
        return -1.
    a = a-a.mean()
    b = b-b.mean()
    return float(a@b / max(1.e-8,np.linalg.norm(a)*np.linalg.norm(b)))


def tiles(image, data):
    x,y,uv = (data[k] for k in ('x','y','uv'))
    gray = np.asarray(image.convert('L'),float)[y,x]
    # Sample the center of a projected texel to avoid mip/filter interpolation near its edge.
    sub = 16*uv % 1
    reliable = np.min(np.minimum(sub,1-sub),axis=1) >= .20
    table=templates(gray[reliable],uv[reliable])
    xyz = np.floor(uv).astype(int)
    low = xyz.min(axis=0)
    cols = xyz[:,0].max()-low[0]+1
    code = (xyz[:,1]-low[1])*cols+xyz[:,0]-low[0]
    order = np.argsort(code)
    cuts = np.r_[0,np.flatnonzero(np.diff(code[order]))+1,len(code)]
    rows = {tuple(xyz[order[a]]):order[a:b] for a,b in zip(cuts,cuts[1:]) if b-a >= 64}
    return rows,gray,table


def front_anomalies(image, data, table):
    uv, p = data['uv'], data['p']
    low = np.floor(uv.min(axis=0)).astype(int) - 1
    high = np.ceil(uv.max(axis=0)).astype(int) + 1
    x,y=np.meshgrid(np.arange(low[0],high[0]),np.arange(low[1],high[1]))
    cells=np.stack((x.ravel(),y.ravel()),axis=1)
    # Measure the same registered texels in every candidate front cell.
    # Using the full RGB here lets another existing mask own a return at its edge.
    grid=(np.arange(1,16,2)+.5)/16
    a,b=np.meshgrid(grid,grid)
    offsets=np.stack((a.ravel(),b.ravel()),axis=1)
    uv=cells[:,None,:]+offsets
    xyz=np.full((*uv.shape[:2],3),float(data['plane']))
    xyz[:,:,data['free']]=uv
    xy=np.floor(project(p,xyz)).astype(int)
    inside=((xy[:,:,0]>=0)&(xy[:,:,1]>=0)&(xy[:,:,0]<image.width)&(xy[:,:,1]<image.height)).all(axis=1)
    xy=xy[inside]
    gray=np.asarray(image.convert('L'),float)[xy[:,:,1],xy[:,:,0]]
    templ=table[texel_bins(offsets)]
    mean=gray.mean(axis=1)
    deviations=gray-mean[:,None]
    expected=templ-templ.mean()
    corr=deviations@expected/(np.linalg.norm(deviations,axis=1)*np.linalg.norm(expected)).clip(1.e-8)
    host={tuple(pos) for pos,s in zip(cells[inside],corr) if s>.60}
    weak={tuple(pos) for pos,s,v in zip(cells[inside],corr,mean) if s<.35 and v<.90*templ.mean()}
    return host,weak


def components(points):
    points=set(points)
    while points:
        seed=min(points)
        group={seed}
        queue=[seed]
        points.remove(seed)
        while queue:
            x,y=queue.pop()
            for neighbor in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if neighbor in points:
                    points.remove(neighbor)
                    group.add(neighbor)
                    queue.append(neighbor)
        yield group


def rectangles(host,weak):
    for group in components(weak):
        if len(group)<4 or len(group)>48:
            continue
        lo=np.min(list(group),axis=0)
        hi=np.max(list(group),axis=0)+1
        width,height=hi-lo
        if min(width,height)<2 or max(width,height)>8 or len(group)<.80*width*height:
            continue
        x,y=lo
        r,t=hi
        interior={(i,j) for i in range(x,r) for j in range(y,t)}
        if host & interior:
            continue
        edges=[[(i,y-1) for i in range(x,r)],[(i,t) for i in range(x,r)],
               [(x-1,j) for j in range(y,t)],[(r,j) for j in range(y,t)]]
        if all(sum(k in host for k in edge)>=.75*len(edge) for edge in edges):
            yield interior,lo,hi


def back_surface(data,group,lo,hi,rows,gray,table,observations=None):
    source=[rows[k] for k in sorted(group) if k in rows]
    if len(source)<3:
        return None
    uv=data['uv']
    depth_sign=int(np.sign(data['plane']-data['eye'][data['axis']]))
    ids=np.concatenate(source)
    depth_scores=[]
    # A flush panel and closer relief compete with the three recessed depths.
    for offset in range(-3,4):
        back=uv[ids]+offset*depth_sign*data['ratio'][ids]
        inside=((back>=lo)&(back<hi)).all(axis=1)
        if inside.sum()<500:
            continue
        ii=ids[inside]
        tpl=table[texel_bins(back[inside])]
        score=correlation(gray[ii],tpl)
        depth_scores.append((score,offset,ii))
    depth_scores.sort(key=lambda s:s[0],reverse=True)
    if len(depth_scores)<2:
        return None
    score,depth,selected=depth_scores[0]
    # A strong correlation in each half of the aperture avoids a single patch
    # explaining a larger panel. Split spatially along its wider direction.
    coord=uv[selected,int(np.argmax(hi-lo))]
    split=coord<np.median(coord)
    bins=texel_bins(uv[selected]+depth*depth_sign*data['ratio'][selected])
    half=[correlation(gray[selected[q]],table[bins[q]]) for q in (split,~split)]
    accepted=depth>0 and score>=.60 and score-depth_scores[1][0]>=.22 and min(half)>=.53
    evidence=dict(depth=int(depth),score=round(score,4),runner_up=round(depth_scores[1][0],4),
                  runner_up_depth=int(depth_scores[1][1]), signed_depth_scores={str(k):round(v,4) for v,k,_ in depth_scores},
                  half_scores=[round(v,4) for v in half],host_repeat=round(data['fit']['corr'],4),
                  host_margin=round(data['fit']['margin'],4),samples=len(selected),
                  opening=[lo.tolist(),hi.tolist()],accepted=bool(accepted))
    if observations is not None:
        observations.append(evidence)
    if not accepted:
        return None
    back=uv+depth*depth_sign*data['ratio']
    back_pixels=((uv>=lo)&(uv<hi)&(back>=lo)&(back<hi)).all(axis=1)
    return back_pixels,dict(evidence,pixels=int(back_pixels.sum()))


def refine(image,parts,camera,*,candidate_log=None):
    """Partition only foreground already owned by a strongly registered photographed wall."""
    if not camera or camera.get('P') is None:
        return parts,[]
    try:
        projection=np.asarray(camera['P'],float)
    except (ValueError,TypeError):
        return parts,[]
    if projection.shape!=(3,4) or not np.isfinite(projection).all() or np.linalg.matrix_rank(projection[:,:3])<3:
        return parts,[]
    result,events=[],[]
    smooth=None
    for part in parts:
        if part.get('normal_camera') is None or not part.get('support'):
            result.append(part)
            continue
        data=host_grid(image,part,camera)
        if data is None:
            result.append(part)
            continue
        rows,gray,table=tiles(image,data)
        host,weak=front_anomalies(image,data,table)
        panels=[]
        observed=[] if candidate_log is not None else None
        for group,lo,hi in rectangles(host,weak):
            matched=back_surface(data,group,lo,hi,rows,gray,table,observed)
            if matched:
                panels.append(matched)
        if candidate_log is not None:
            candidate_log.extend(dict(e,parent=part['id'],axis=data['axis'],host_level=int(data['plane'])) for e in observed)
        if not panels:
            result.append(part)
            continue
        if smooth is None:
            smooth=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
        fields={k:v for k,v in part.items() if k not in ('source_crop','context_crop','corner_junctions','diagonal_join')}
        x,y=data['x'],data['y']
        remaining=np.ones(len(x),bool)
        def rewrite(pixels,key):
            mask=np.zeros((image.height,image.width),bool)
            mask[y[pixels],x[pixels]]=True
            return dict(with_mask(fields,mask,image.size,smooth,np.ones(2)),id=key)
        children=[]
        for i,(pixels,evidence) in enumerate(panels):
            key=f"{part['id']}-phase-inset-{i}"
            remaining &= ~pixels
            children.append(rewrite(pixels,key))
            events.append(dict(parent=part['id'],output=key,axis=data['axis'],host_level=int(data['plane']),**evidence))
        result.append(rewrite(remaining,part['id']+'-phase-rim'))
        result.extend(children)
    if not events:
        return parts,[]
    retired={e['parent'] for e in events}
    for i,part in enumerate(result):
        old=part.get('corner_junctions',[])
        live=[j for j in old if j.get('neighbor') not in retired]
        if len(live)!=len(old):
            result[i]=dict(part,corner_junctions=live)
    return result,events
