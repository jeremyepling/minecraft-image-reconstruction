"""Single-image silhouette turns as a candidate metric ruler. No pose or labels."""
import cv2
import numpy as np
from benchmark.camera_grid_joint import mask, patches

cv2.setNumThreads(1)


def outlines(rgb, parts):
    """Refine coarse observed building support against native foreground/background colours."""
    h, w = rgb.shape[:2]
    raw = [mask(p['support'], (w, h)).astype('B') for p in parts if p.get('support')]
    if not raw:
        return [], []
    union = np.maximum.reduce(raw)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    trimap = np.where(cv2.dilate(union, kernel), cv2.GC_PR_BGD, cv2.GC_BGD).astype('B')
    trimap[union > 0] = cv2.GC_PR_FGD
    trimap[cv2.erode(union, kernel) > 0] = cv2.GC_FGD
    if not (trimap == cv2.GC_FGD).any():
        return [], raw
    cv2.setRNGSeed(83)
    cv2.grabCut(rgb, trimap, None, np.zeros((1,65)), np.zeros((1,65)), 2, cv2.GC_INIT_WITH_MASK)
    contours, _ = cv2.findContours((trimap % 2).astype('B'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [cv2.approxPolyDP(c, 1.8, True)[:,0,:].astype(float) + .5
                for c in contours if cv2.contourArea(c) > 220]
    return polygons, raw


def orientation(geo, lines, size):
    """Horizontal azimuth is a fourfold circular consensus after eliminating GeoCalib gravity."""
    fx, fy = geo['focal_xy_px']
    cx, cy = geo['principal_xy_px']
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    up = np.array(geo['up_cam'])
    up /= np.linalg.norm(up)
    x0 = np.array([1.,0,0]) - up * up[0]
    x0 /= np.linalg.norm(x0)
    z0 = np.cross(x0,up)
    votes = []
    for line in lines:
        xy = np.asarray(line).reshape(2,2)
        rays = np.column_stack((xy,np.ones(2))) @ np.linalg.inv(K).T
        plane = np.cross(*rays)
        if abs(plane @ up) < .18*np.linalg.norm(plane):
            continue
        direction = np.cross(up,plane)
        votes.append((np.arctan2(direction @ z0,direction @ x0), np.linalg.norm(xy[0]-xy[1])))
    if len(votes)<4:
        return None
    angles,weights = np.asarray(votes).T
    seed = np.angle(np.sum(weights*np.exp(4j*angles)))/4
    keep = np.cos(4*(angles-seed)) > np.cos(np.deg2rad(28))
    if keep.sum()<3:
        return None
    angle = np.angle(np.sum(weights[keep]*np.exp(4j*angles[keep])))/4
    x = np.cos(angle)*x0+np.sin(angle)*z0
    return K @ np.column_stack((x,up,np.cross(x,up)))


def chart(xy, linear, plane_axis):
    rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(linear).T
    if min(abs(rays[:,plane_axis]))<.045 or np.ptp(np.sign(rays[:,plane_axis]))>0:
        return None
    return rays/abs(rays[:,[plane_axis]])


def runs(polys, linear):
    traces = []
    for poly in polys:
        if len(poly)<5:
            continue
        for start in range(len(poly)):
            xy = poly[(np.arange(5)+start)%len(poly)]
            if min(np.linalg.norm(np.diff(xy,axis=0),axis=1))<4.5:
                continue
            for normal in (0,2):
                flat = chart(xy,linear,normal)
                if flat is None:
                    continue
                delta = np.diff(flat,axis=0)[:,[2-normal,1]]
                axes = np.argmax(abs(delta),axis=1)
                if list(axes) not in ([0,1,0,1],[1,0,1,0]):
                    continue
                major = abs(delta).max(1)
                if np.max(abs(delta).min(1)/major)>.17:
                    continue
                # Repeated heights or tread widths. Compare each alternating dimension
                # independently so a 2x1 stair can remain an explicit competitor.
                ratios = major[:2]/major[2:]
                if np.any((ratios<.78)|(ratios>1.28)):
                    continue
                ystep = np.median(major[axes==1])
                traces.append(dict(xy=xy.tolist(),normal=normal,unit_depth=float(1/ystep),
                                   width_over_rise=float(np.median(major[axes==0])/ystep)))
    return traces


def evaluate(traces, polys, parts, masks, linear, anchors, field, geo, rgb):
    """A distinct complete sprite on the same observed wall resolves integer dilation."""
    h,w = masks[0].shape if masks else (0,0)
    union = np.maximum.reduce(masks) if masks else None
    metrics=[]
    K=np.array([[geo['focal_xy_px'][0],0,geo['principal_xy_px'][0]],
                [0,geo['focal_xy_px'][1],geo['principal_xy_px'][1]],[0,0,1.]])
    # Only source orientations produced with these intrinsics enter this gate.
    R=np.linalg.solve(K,linear)
    for i,t in enumerate(traces):
        xy=np.array(t['xy'])
        normal=t['normal']
        for part, m in zip(parts,masks):
            if part.get('normal_camera') is None or abs(np.array(part['normal_camera'])@R[:,normal])<.80:
                continue
            near=cv2.dilate(m,np.ones((15,15),'B'))
            pts=np.concatenate([np.linspace(a,b,9) for a,b in zip(xy,xy[1:])]).astype(int)
            if near[pts[:,1],pts[:,0]].mean()<.80:
                continue
            for rise in (1,2,3,4):
                depth=rise*t['unit_depth']
                flat=chart(xy,linear,normal)*depth
                widths=t['width_over_rise']*rise
                if abs(widths-round(widths))>.18 or round(widths)<1:
                    continue
                verified=[]
                for a in anchors:
                    if a['surface']!=part['id']:
                        continue
                    quad=np.asarray(a['quad'])
                    if np.min(np.linalg.norm(quad[:,None]-xy[None,:],axis=2))<12:
                        continue
                    q=chart(quad,linear,normal)
                    if q is None:
                        continue
                    lengths=np.linalg.norm(np.roll(q,1,axis=0)-q,axis=1)*depth
                    if np.max(abs(lengths-1))<.14:
                        verified.append(quad)
                # Check scale-conditioned unit cells on a disjoint interior portion.
                # A projected quad can vote only if the entire cell is on the same
                # observed mask. The turn itself fixes lattice phase before matching.
                direct=[]
                along=2-normal
                ys,xs=np.nonzero(m[::12,::12])
                if len(xs):
                    samples=chart(np.column_stack((xs*12+.5,ys*12+.5)),linear,normal)
                    if samples is not None:
                        cells=np.unique(np.floor((samples*depth-flat[0])[:,[along,1]]).astype(int),axis=0)
                        from benchmark.full_block_lattice import verify_sprite
                        for cell in cells[np.linspace(0,len(cells)-1,min(20,len(cells)),dtype=int)]:
                            xyz=np.tile(flat[0],(4,1))
                            xyz[:,[along,1]] += cell+np.array([[0,0],[1,0],[1,1],[0,1]])
                            pix=xyz@linear.T
                            q=pix[:,:2]/pix[:,2:]
                            pix=q.astype(int)
                            if np.any(pix<0) or np.any(pix>=[w,h]) or np.min(np.linalg.norm(np.roll(q,1,axis=0)-q,axis=1))<8:
                                continue
                            if np.min(np.linalg.norm(q[:,None]-xy[None,:],axis=2))<20:
                                continue
                            lo=pix.min(0)
                            hi=pix.max(0)
                            patchmask=np.zeros(tuple((hi-lo+1)[::-1]),'B')
                            cv2.fillConvexPoly(patchmask,pix-lo,1)
                            if not m[lo[1]:hi[1]+1,lo[0]:hi[0]+1][patchmask.astype(bool)].all():
                                continue
                            match=verify_sprite(rgb,q)
                            if match and match[-1]>=.72:
                                if all(np.min(np.linalg.norm(q[:,None]-old[None,:],axis=2))>4 for old in direct):
                                    direct.append(q)
                if len(direct)>=2:
                    verified.extend(direct)
                metric=dict(direct_cells=len(direct),plane_depth=depth,linear=linear.tolist(),trace=i,rise=rise,part=part['id'],sprite_count=len(verified),contact=False,held=False)
                metrics.append(metric)

                # A visible base must meet a source-estimated upward floor, at consistent
                # scaled Y, across multiple columns of this same observed facade.
                cols=np.flatnonzero(m.any(0))[::6]
                yy=h-1-np.argmax(m[::-1,cols],axis=0)
                fh,fw=field['mask'].shape
                xf=np.clip((cols+.5)*fw/w,0,fw-1).astype(int)
                yf=np.clip((yy+4.5)*fh/h,0,fh-1).astype(int)
                yt=np.clip((yy-3.5)*fh/h,0,fh-1).astype(int)
                z=field['points'][...,2]+float(field['z_shift'])
                outside=(yy+5<h)&(union[np.minimum(yy+5,h-1),cols]==0)
                floor=(field['normal'][yf,xf]@R[:,1])>.85
                valid=outside & floor & field['mask'][yf,xf] & field['mask'][yt,xf] & (abs(z[yf,xf]/np.maximum(z[yt,xf],.001)-1)<.05)
                ground=chart(np.column_stack((cols[valid]+.5,yy[valid]+1)),linear,normal) if valid.sum()>=4 and np.ptp(cols[valid])>=30 else None
                if ground is None:
                    continue
                height=-np.median(ground[:,1])*depth
                if height<=.15 or np.quantile(abs(ground[:,1]*depth+height),.8)>.18:
                    continue
                metric['contact']=True
                eye=-flat[0].copy()
                eye[1]=height
                metric['ground_P']=np.column_stack((linear,-linear@eye)).tolist()
                if not verified:
                    continue
                if np.max(abs(flat[:,1]+height-np.rint(flat[:,1]+height)))>.18:
                    continue
                for poly in polys:
                    for j,point in enumerate(poly):
                        xx,yy=point.astype(int)
                        neighborhood=poly[(np.arange(3)+j-1)%len(poly)]
                        if np.min(np.linalg.norm(point-xy,axis=1))<32 or np.any(near[neighborhood[:,1].astype(int),neighborhood[:,0].astype(int)]==0):
                            continue
                        if any(np.min(np.linalg.norm(q-point,axis=1))<20 for q in verified):
                            continue
                        p=chart(neighborhood,linear,normal)
                        if p is None:
                            continue
                        ds=np.diff(p,axis=0)[:,[2-normal,1]]
                        if len(set(abs(ds).argmax(axis=1)))<2 or max(abs(ds).min(1)/np.maximum(abs(ds).max(1),1.e-9))>.17:
                            continue
                        pos=p[1]*depth+eye
                        if abs(pos-np.rint(pos)).max()<.13:
                            metric.update(held=True,P=np.column_stack((linear,-linear@eye)).tolist())
                            break
                    if metric['held']:
                        break
    return metrics
