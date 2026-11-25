"""Use a visible unit pavement grid to measure front-on architectural returns."""
import numpy as np
from PIL import Image, ImageFilter


def floor_calibration(rgb):
    """Infer a horizon and focal scale from unit floor seams in both image margins.

    This model needs ground lines parallel to the image rows, a perspective
    ground plane, and a roughly centered camera. Decline other camera views.
    """
    h,w = rgb.shape[:2]
    if min(h,w)<120:
        return None
    gray = rgb[:,:,:3].mean(axis=2)
    margins = [np.median(gray[:,k:k+w//6],axis=1) for k in (0,5*w//6)]
    signal = np.maximum(*margins)
    seams = []
    for i in range(int(.68*h),h-5):
        if signal[i]!=min(signal[i-2:i+3]):
            continue
        shoulder = max(signal[i-4:i-2].mean(),signal[i+3:i+5].mean())
        if shoulder-signal[i]<14:
            continue
        if seams and i-seams[-1]<max(4,h/160):
            if signal[i]<signal[seams[-1]]:
                seams[-1]=i
        else:
            seams.append(i)
    if len(seams)<6:
        return None
    y = (np.array(seams[:-1])+seams[1:])/2
    periods=[]
    nfft = 1 << int(np.ceil(np.log2(6*w)))
    for row in np.rint(y).astype(int):
        line=gray[row]
        a=np.clip(line-np.median(line),-35,10)
        spectra=[abs(np.fft.rfft(z*np.hanning(len(z)),nfft)) for z in (a[:w//4],a[-w//4:])]
        power=np.minimum(*spectra)
        peaks=np.where((power[1:-1]>power[:-2])&(power[1:-1]>power[2:]))[0]+1
        peaks=peaks[(peaks>nfft/(w*.12))&(peaks<nfft/5)]
        if not len(peaks):
            return None
        peaks=peaks[power[peaks]>.5*max(power[peaks])]
        periods.append(nfft/min(peaks))
    # One noisy or harmonic row must not drive the ground vanishing line.
    periods=np.array(periods)
    slopes=[(periods[j]-periods[i])/(y[j]-y[i]) for i in range(len(y)) for j in range(i+1,len(y))]
    slope=float(np.median(slopes))
    if slope<.015 or slope>.7:
        return None
    horizon=float(np.median(y-periods/slope))
    if not .05*h<horizon<.68*h:
        return None
    scale=slope*(y-horizon)
    if np.median(abs(periods/scale-1))>.035:
        return None
    # Consecutive unit lines have a constant inverse-depth step. This also
    # rejects lawns, decorative horizontal stripes, and most oblique views.
    deltas=-np.diff(1/(np.array(seams)-horizon))
    step=np.median(deltas)
    if np.quantile(abs(deltas/step-1),.75)>.14:
        return None
    focal=float(slope/step)
    if not .4*h < focal < 3*h:
        return None
    return dict(horizon_y=horizon,focal_px=focal,pitch_slope=slope,seams=len(seams))

def ledges(gray, mask, vp, pitch, focal):
    """Find a ledge only at a bilateral outward step in the building silhouette."""
    yy,xx=np.where(mask)
    lo,hi=yy.min(),yy.max()
    width=np.zeros(gray.shape[0]); mids=np.zeros_like(width)
    for y in range(lo,hi+1):
        xs=np.flatnonzero(mask[y])
        if len(xs):
            width[y]=xs[-1]-xs[0]+1
            mids[y]=(xs[-1]+xs[0])/2
    step=max(4,round(.7*pitch))
    hits=[]
    for y in range(lo+step,hi-step):
        gain=width[y+step]-width[y-step]
        shift=abs(mids[y+step]-mids[y-step])
        if gain>1.2*pitch and shift<gain*.24:
            hits.append(y)
    clusters=np.split(hits,np.flatnonzero(np.diff(hits)>1)+1) if hits else []
    result=[]
    for cluster in clusters:
        a=max(lo,int(cluster[0])-step)
        b=min(hi,int(cluster[-1])+step)
        pitch_window=max(4,int(pitch*.7))
        upper=np.median(width[max(lo,a-pitch_window):a+1])
        lower=np.median(width[b:min(hi,b+pitch_window)+1])
        if upper<pitch or lower<upper+1.2*pitch:
            continue
        mid=int(np.median(mids[a:b+1]))
        rows=np.arange(max(3,a),min(gray.shape[0]-3,b+1))
        # Front lip extends onto both lateral ledges; the recessed upper
        # front edge terminates at its narrower corners.
        center=np.arange(int(mid-.38*upper),int(mid+.38*upper))
        wing=np.r_[np.arange(int(mid-.45*lower),int(mid-.52*upper)),
                   np.arange(int(mid+.52*upper),int(mid+.45*lower))].astype(int)
        if len(wing)<6:
            continue
        grad=gray[rows+2]-gray[rows-2]
        w=grad[:,wing].mean(1)
        c=grad[:,center].mean(1)
        lip=int(np.argmax(abs(w)))
        if abs(w[lip])<6:
            continue
        yl=rows[lip]
        if yl<=vp[1]+.7*pitch:
            continue
        candidates=[]
        for d in range(1,9):
            yback=vp[1]+(yl-vp[1])/(1+d*pitch/focal)
            j=np.argmin(abs(rows-yback))
            if abs(rows[j]-yback)>1.1 or yl-rows[j]<4:
                continue
            # The back edge must terminate on the narrower face, and have
            # opposing contrast to the front lip across the ledge.
            probe=np.arange(max(0,j-2),min(len(rows),j+3))
            local=-np.sign(w[lip])*c[probe]-.7*abs(w[probe])-.8*abs(rows[probe]-yback)
            j=probe[np.argmax(local)]
            score=max(local)
            candidates.append((score,d,rows[j]))
        candidates.sort(reverse=True)
        if candidates and candidates[0][0]>6 and (len(candidates)==1 or candidates[0][0]>1.3*candidates[1][0]):
            strength,d,upper_y=candidates[0]
            result.append(dict(front_y=int(yl),back_y=int(upper_y),depth=int(d),
                               width_px=float(upper),center_x=mid,edge_strength=round(float(strength),2)))
    return result

def front_lattice(gray, mask, cal):
    from engine.geometry import project, transform
    yy,xx=np.where(mask)
    h,w=gray.shape
    y0,y1=int(yy.min()),int(yy.max())
    if y1<=cal['horizon_y'] or y1-y0<24 or cal['horizon_y']>=h/2-1:
        return None
    pitch=(y1-cal['horizon_y'])*cal['pitch_slope']
    if pitch<8:
        return None
    # Calibrate with straight silhouette edges; bail out for stepped exteriors.
    rows=np.linspace(y0+.1*(y1-y0),y1-.1*(y1-y0),25).astype(int)
    ends=[np.flatnonzero(mask[y]) for y in rows]
    if any(not len(e) for e in ends):
        return None
    widths=np.array([e[-1]-e[0] for e in ends])
    if widths.min()<.88*widths.max():
        return None
    cx=w/2; cy=h/2
    if cy-cal['horizon_y']<1:
        return None
    vp=cy+cal['focal_px']**2/(cy-cal['horizon_y'])
    side_center=np.median(rows)
    sides=[np.median(cx+(np.array([e[j] for e in ends])-cx)*(vp-side_center)/(vp-rows)) for j in (0,-1)]
    cols=np.arange(max(0,int(sides[0]+.1*widths.mean())),min(w,int(sides[1]-.1*widths.mean())))
    if not len(cols):
        return None
    def horizontal(seed):
        ys=np.arange(max(3,round(seed-.4*pitch)),min(h-3,round(seed+.4*pitch)))
        if not len(ys):
            return None
        scores=abs((gray[ys+2][:,cols]-gray[ys-2][:,cols]).mean(1))
        return ys[scores.argmax()]
    top,base=horizontal(y0),horizontal(y1)
    if top is None or base is None or base<=max(top,cal['horizon_y']):
        return None
    pitch=(base-cal['horizon_y'])*cal['pitch_slope']
    sides2=[]
    for side in sides:
        shifts=np.arange(-int(.3*pitch),int(.3*pitch)+1)
        x=np.rint(cx+(side+shifts[:,None]-cx)*(vp-rows)/(vp-side_center)).astype(int)
        x=np.clip(x,3,w-4)
        strength=abs((gray[rows,x+2]-gray[rows,x-2]).mean(axis=1))
        sides2.append(side+shifts[strength.argmax()])
    left,right=sides2
    quad=np.array([[cx+(s-cx)*(vp-y)/(vp-side_center),y] for s,y in
                   ((left,top),(right,top),(right,base),(left,base))])
    count_x=round((quad[2,0]-quad[3,0])/pitch)
    sin_t=(cy-cal['horizon_y'])/np.hypot(cy-cal['horizon_y'],cal['focal_px'])
    count_y=round((1-(vp-base)/(vp-top))*cal['focal_px']/(pitch*sin_t))
    if min(count_x,count_y)<3 or max(count_x,count_y)>100:
        return None
    grid=transform(quad)@np.diag([1/count_x,1/count_y,1.])
    return grid,count_x,count_y,pitch


def visible_recesses(rgb, gray, mask, cal, lattice=None):
    from engine.geometry import UNIT,bilinear,project
    if lattice is None:
        lattice=front_lattice(gray,mask,cal)
    if not lattice:
        return []
    grid,nx,ny,pitch=lattice
    vp=np.array([rgb.shape[1]/2,cal['horizon_y']])
    # Rectify to one regular lattice. Identify connected dark cells; only a
    # fitted angular return plus two back-rim edges can move any block.
    probes=np.stack(np.meshgrid(np.arange(nx)+.5,np.arange(ny)+.5),axis=-1)
    cell_mean=np.mean([bilinear(rgb,project(grid,probes+[dx,dy])).mean(2)
                      for dx in (-.22,0,.22) for dy in (-.22,0,.22)],axis=0)
    seeds=cell_mean < np.quantile(cell_mean,.75,axis=1,keepdims=True)*.91
    if np.median(cell_mean[-1])<.85*np.median(cell_mean[-2]):
        seeds[-1]=seeds[-2]
    boxes=[]
    for yi,xi in zip(*np.where(seeds)):
        if not seeds[yi,xi]:
            continue
        seeds[yi,xi]=False
        pts=[]
        stack=[(yi,xi)]
        while stack:
            y,x=stack.pop();pts.append((x,y))
            for j,i in ((y,x-1),(y,x+1),(y-1,x),(y+1,x)):
                if 0<=i<nx and 0<=j<ny and seeds[j,i]:
                    seeds[j,i]=False;stack.append((j,i))
        pts=np.array(pts)
        a,b=pts.min(0),pts.max(0)+1
        if a[0]==0 or b[0]==nx or a[1]==0 or min(b-a)<2 or len(pts)<.75*np.prod(b-a):
            continue
        boxes.append((a,b,project(grid,a+UNIT*(b-a))))
    results=[]
    for a,b,q in boxes:
        fingerprint=depth_fingerprints(rgb,grid,nx,ny,a,b,vp,pitch,cal['focal_px'])
        out=fit_reveal(gray,q,vp,pitch,cal['focal_px'],fingerprint)
        results.append(dict(rim=q.round(2).tolist(),**out))
    return results


def fit_reveal(gray,q,vp,pitch,focal,fingerprint):
    from engine.geometry import bilinear
    field=gray[:,:,None]
    def line_score(p,r):
        v=r-p
        d=np.linalg.norm(v)
        if d<3:
            return 0
        normal=np.array([-v[1],v[0]])/d
        pts=p+np.linspace(.18,.82,23)[:,None]*v
        return abs(float((bilinear(field,pts+2*normal)-bilinear(field,pts-2*normal)).mean()))
    # Side toward the vanishing point is hidden. The opposite rule applies
    # to the opening: back edges visible inside the aperture bound its reveals.
    left=vp[0]>q[:,0].mean()
    top=vp[1]>q[:,1].mean()
    side=(0,3) if left else (1,2)
    sill=(0,1) if top else (3,2)
    corner=next(iter(set(side)&set(sill)))
    scores=[]
    for d in range(1,9):
        back=vp+(q-vp)/(1+d*pitch/focal)
        delta=back[corner]-q[corner]
        denom=np.linalg.norm(delta)
        if min(abs(delta))<2 or denom<5:
            continue
        # Check edge along the actual diagonal, not just darkness inside a
        # candidate window. Reject shoulders parallel to a row or column.
        diag=line_score(q[corner],back[corner])
        axis=[line_score(back[j],back[k]) for j,k in (side,sill)]
        if max(axis)<10 or diag<9:
            continue
        score=max(axis)*min(1,diag/14)*(1+.012*min(axis))
        if fingerprint[d]>.5 and fingerprint[d]>fingerprint[0]+.15:
            score*=1+.12*fingerprint[d]
        if fingerprint[0]>.55 and fingerprint[0]>fingerprint[d]+.15:
            continue
        scores.append((score,d,back,axis,diag))
    scores.sort(key=lambda entry:-entry[0])
    if not scores or scores[0][0]<6 or (len(scores)>1 and scores[1][0]>.9*scores[0][0]):
        return dict(status='unresolved',depth=None)
    strength,d,back,edges,diagonal=scores[0]
    return dict(status='measured',depth=d,back_rim=back.round(2).tolist(),
                reveal_segment=[q[corner].round(2).tolist(),back[corner].round(2).tolist()],
                rim_strength=[round(s,2) for s in edges],diagonal_strength=round(diagonal,2),
                aligned_texture_similarity=[round(float(f),3) for f in fingerprint])

def depth_fingerprints(rgb,grid,nx,ny,a,b,vp,pitch,focal):
    """Cross-check a recess by reprojecting front-face texture onto the back."""
    from engine.geometry import project,bilinear
    uv=np.stack(np.meshgrid(np.linspace(.17,.83,9),np.linspace(.17,.83,9)),axis=-1)
    def patterns(coords):
        pixels=bilinear(rgb,coords).mean(-1).reshape(len(coords),-1)
        pixels-=pixels.mean(1,keepdims=True)
        return pixels/np.maximum(np.linalg.norm(pixels,axis=1,keepdims=True),10)
    ref=np.array([[x,y] for x in range(nx) for y in range(ny)
                  if (x<a[0] or x>=b[0] or y<a[1] or y>=b[1]) and min(abs(x-a[0]),abs(y-a[1]))<4])
    pos=np.array([[x,y] for x in range(a[0],b[0]) for y in range(a[1],b[1])])
    if not len(ref) or not len(pos): return [0]*9
    front=project(grid,ref[:,None,None,:]+uv)
    bank=patterns(front)
    points=project(grid,pos[:,None,None,:]+uv)
    inv=np.linalg.inv(grid)
    result=[]
    for depth in range(9):
        on_image=vp+(points-vp)/(1+depth*pitch/focal)
        lattice=project(inv,on_image)
        inside=((lattice>=a)&(lattice<b)).all(axis=-1).mean(axis=(1,2))>.85
        strength=patterns(on_image[inside])@bank.T if inside.any() else np.zeros((1,1))
        result.append(float(np.median(strength.max(axis=1))))
    return result

def inspect_depth(rgb):
    """Measure only evidence visible in the screenshot; ground truth stays in scoring."""
    from engine.texel_grid import structure_parts
    cal=floor_calibration(rgb)
    audit=dict(calibration=cal,recesses=[],ledges=[])
    if not cal:
        audit['reason']='No reliable front-facing unit pavement grid'
        return audit
    image=Image.fromarray(rgb[:,:,:3].astype('uint8'),'RGB')
    parts=structure_parts(image)
    masks=[np.asarray(part.getchannel('A'))>0 for part in parts if 'A' in part.getbands()]
    all_y=[np.where(mask)[0].max() for mask in masks if mask.any()]
    if not all_y:
        audit['reason']='No separated foreground to measure'
        return audit
    for mask in masks:
        if not mask.any():
            continue
        ground=int(np.where(mask)[0].max())
        pitch=(ground-cal['horizon_y'])*cal['pitch_slope']
        if pitch<8:
            continue
        gray=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(float(.04*pitch))),float)
        vp=np.array([image.width/2,cal['horizon_y']])
        for trace in ledges(gray,mask,vp,pitch,cal['focal_px']):
            trace['pitch_px']=float(pitch)
            audit['ledges'].append(trace)
        lattice=front_lattice(gray,mask,cal)
        if lattice is None:
            continue
        traces=visible_recesses(rgb,gray,mask,cal,lattice)
        audit['recesses'].extend(traces)
        if len(masks)==1 and not audit['ledges'] and any(r['status']=='measured' for r in traces):
            # The same calibrated grid must drive the aperture measurements
            # and their final placement. Preserve disconnected air holes.
            from engine.geometry import project
            grid,nx,ny,_=lattice
            indices=np.stack(np.meshgrid(np.arange(nx),np.arange(ny)),axis=-1)
            offsets=np.stack(np.meshgrid((np.arange(5)+.5)/5,(np.arange(5)+.5)/5),axis=-1)
            points=project(grid,indices[:,:,None,None,:]+offsets).astype(int)
            inside=(points[:,:,:,:,0]>=0)&(points[:,:,:,:,0]<mask.shape[1])&(points[:,:,:,:,1]>=0)&(points[:,:,:,:,1]<mask.shape[0])
            occupied=inside & mask[points[...,1].clip(0,mask.shape[0]-1),points[...,0].clip(0,mask.shape[1]-1)]
            audit['front_lattice']=dict(grid_to_image=grid.tolist(),columns=nx,rows=ny,
                                        omitted_ij=indices[occupied.mean(axis=(2,3))<.5].tolist())
    return audit


def refine_depth(rgb,planes,cells):
    """Assign supported setbacks; sample back walls and visible return faces independently."""
    from engine.geometry import UNIT,transform,project
    planes=[dict(p) for p in planes]
    cells=[dict(c,xyz=list(c['xyz'])) for c in cells]
    if not planes or not cells:
        return planes,cells
    audit=inspect_depth(rgb)
    planes[0]['depth_evidence']=audit
    measured=[r for r in audit['recesses'] if r['status']=='measured']
    if not measured and not audit['ledges']:
        return planes,cells
    if 'front_lattice' in audit and len(planes)==1:
        fitted=audit['front_lattice']
        grid=np.array(fitted['grid_to_image'])
        nx,ny=fitted['columns'],fitted['rows']
        plane=planes[0]
        plane.update(columns=nx,rows=ny,grid_to_image=grid.tolist(),origin=[0,0,0],
                     quad=project(grid,UNIT*[nx,ny]).tolist(),spacing_method='calibrated facade rim')
        omitted=set(map(tuple,fitted['omitted_ij']))
        cells=[dict(plane=plane['name'],ij=[x,j],xyz=[x,ny-1-j,0],facing='south',
                    quad=project(grid,UNIT+[x,j]).tolist())
               for j in range(ny) for x in range(nx) if (x,j) not in omitted]
    centers=np.array([np.mean(c['quad'],axis=0) for c in cells])
    occupied={tuple(c['xyz']) for c in cells}
    facade={tuple(c['xyz']):dict(c) for c in cells}
    additions=[]
    def surface(pos,quad,direction,parent,inferred=False):
        key=tuple(map(int,pos))
        if key in occupied or key[1]<min_y:
            return
        occupied.add(key)
        cell=dict(parent,xyz=list(key),quad=np.round(quad,4).tolist(),
                  facing=direction,depth_surface='visible return')
        if inferred:
            cell.update(depth_surface='inferred return',sample_source='adjacent facade',
                        sample_source_xyz=parent['xyz'])
        additions.append(cell)
    min_y=min(c['xyz'][1] for c in cells)
    vp=np.array([rgb.shape[1]/2,audit['calibration']['horizon_y']])
    for ledge in sorted(audit['ledges'],key=lambda r:-r['front_y']):
        # A shared ledge must belong to the same projected structure.
        width=ledge['width_px']
        middle=ledge['center_x']
        inside=np.abs(centers[:,0]-middle)<width*.53
        upper=inside & (centers[:,1]<=ledge['back_y'])
        base=inside & (centers[:,1]>=ledge['front_y'])
        if not upper.any() or not base.any():
            continue
        near=sorted(np.where(base)[0],key=lambda i:centers[i,1])
        front={}
        for i in near:
            c=cells[i]
            front.setdefault(c['xyz'][0],c)
        for i in np.flatnonzero(upper):
            c=cells[i]
            occupied.discard(tuple(c['xyz']))
            c['xyz'][2]-=ledge['depth']
            occupied.add(tuple(c['xyz']))
        for c in front.values():
            x,y,z=c['xyz']
            q=np.array(c['quad'],float)[:2];q[:,1]=ledge['front_y']
            for k in range(1,ledge['depth']):
                near_q=vp+(q-vp)/(1+k*ledge['pitch_px']/audit['calibration']['focal_px'])
                far_q=vp+(q-vp)/(1+(k+1)*ledge['pitch_px']/audit['calibration']['focal_px'])
                surface([x,y,z-k],np.vstack([far_q,near_q[::-1]]),'up',c)
    for trace in measured:
        q=np.array(trace['rim'])
        back=np.array(trace['back_rim'])
        inv=np.linalg.inv(transform(q))
        uv=project(inv,centers)
        indices=np.flatnonzero(((uv>=0)&(uv<1)).all(axis=1))
        if not len(indices):
            continue
        # Use the calibrated outer-to-inner scale for every depth slice.
        scale=np.linalg.norm(back[2]-back[0])/np.linalg.norm(q[2]-q[0])
        d=trace['depth']
        def retreat(points,k):
            return vp+(points-vp)/(1+(1/scale-1)*k/d)
        locations={tuple(cells[i]['xyz'][:2]) for i in indices}
        sx=-1 if vp[0]>q[:,0].mean() else 1
        sy=1 if vp[1]>q[:,1].mean() else -1
        for i in indices:
            c=cells[i];x,y,z=c['xyz'];cq=np.array(c['quad'])
            occupied.discard((x,y,z))
            c['xyz'][2]-=d
            occupied.add(tuple(c['xyz']))
            c['quad']=retreat(cq,d).round(4).tolist()
            c['depth_surface']='recess back'
            edges=[(x-1,y,cq[[0,3]],'east',sx==-1),(x+1,y,cq[[1,2]],'west',sx==1),
                   (x,y+1,cq[[0,1]],'down',sy==1),(x,y-1,cq[[3,2]],'up',sy==-1)]
            for px,py,edge,direction,visible in edges:
                # A door at ground level has no lower sill. Require a
                # neighboring facade voxel before extending any jamb.
                proxy=facade.get((px,py,z))
                if (px,py) in locations or proxy is None:
                    continue
                for k in range(1,d):
                    if visible:
                        near=retreat(edge,k); far=retreat(edge,k+1)
                        surface([px,py,z-k],np.vstack([near,far[::-1]]),direction,c)
                    else:
                        surface([px,py,z-k],proxy['quad'],direction,proxy,inferred=True)
    planes[0]['depth_status']='visible returns measured; hidden extent unresolved'
    return planes,cells+additions
