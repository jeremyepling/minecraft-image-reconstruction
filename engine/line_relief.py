"""Infer qualitative relief from faces and junctions in a whole-image line map."""
import numpy as np
from PIL import Image, ImageFilter

from engine.geometry import UNIT, bilinear, project, transform


def _cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def _faces(lines, mask,field):
    """Pair opposite jambs and intersect them with supported lintel and sill lines."""
    height, width = mask.shape
    if len(lines) < 4:
        return []
    ends = np.asarray([[s['p'], s['q']] for s in lines], float)
    mids = ends.mean(1)
    vectors = ends[:, 1]-ends[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    axes = vectors/np.maximum(lengths[:, None], .001)
    x, y = np.rint(mids).astype(int).T
    inside = mask[y.clip(0, height-1), x.clip(0, width-1)]
    substantial = lengths > max(10, .012*max(width, height))
    vertical = np.flatnonzero(inside & substantial & (abs(axes[:, 1]) > .84))
    horizontal = np.flatnonzero(inside & substantial & (abs(axes[:, 0]) > .78))
    if not len(horizontal):
        return []
    h_unit, h_center = axes[horizontal], mids[horizontal]
    meet = {}
    for i in vertical:
        a = axes[i]
        # Both segments must reach the intersection or terminate near it.
        denom = _cross(a, h_unit)
        delta = h_center-mids[i]
        t = _cross(delta, h_unit)/denom
        u = _cross(delta, a)/denom
        pad = np.clip(.27*np.minimum(lengths[i], lengths[horizontal]), 5., 23.)
        ok = ((abs(t) < lengths[i]/2+pad) & (abs(u) < lengths[horizontal]/2+pad))
        meet[i] = {int(j): mids[i]+s*a for j, s in zip(horizontal[ok], t[ok])}
    found = []
    for ix, i in enumerate(vertical):
        for j in vertical[ix+1:]:
            if abs(axes[i] @ axes[j]) < .98:
                continue
            left, right = (i, j) if mids[i, 0] < mids[j, 0] else (j, i)
            sep = mids[right, 0]-mids[left, 0]
            if not .18*min(lengths[i], lengths[j]) < sep < 4.5*min(lengths[i], lengths[j]):
                continue
            common = sorted(meet[i].keys() & meet[j].keys(), key=lambda h: meet[i][h][1])
            for n, top in enumerate(common):
                for bottom in common[n+1:]:
                    if abs(axes[top] @ axes[bottom]) < .984:
                        continue
                    quad = np.array([meet[left][top], meet[right][top],
                                     meet[right][bottom], meet[left][bottom]])
                    edges=np.roll(quad,-1,axis=0)-quad
                    if np.min(_cross(edges,np.roll(edges,-1,axis=0)))<=1.:
                        continue
                    sides = np.linalg.norm(edges,axis=1)
                    shortest, longest = min(sides), max(sides)
                    if shortest < max(10., longest*.19) or longest > .58*max(width, height):
                        continue
                    ids = [top, right, bottom, left]
                    fractions = np.minimum(sides, lengths[ids])/np.maximum(sides, lengths[ids])
                    if min(fractions) < .18 or np.mean(fractions) < .54:
                        continue
                    mid = quad.mean(0).round().astype(int)
                    if not mask[mid[1].clip(0,height-1),mid[0].clip(0,width-1)]:
                        continue
                    score = (np.mean(fractions)**2 * np.sqrt(shortest)
                             * np.mean([np.log1p(lines[k]['strength']) for k in ids]))
                    found.append((score, quad))
    # Partial contours are common on equal-material relief. Extrapolate a
    # corner with two terminating arms, then seek its opposite edges in the
    # image, requiring sustained contrast at each recovered side.
    elbows = []
    for i, crossed in meet.items():
        for j,corner in crossed.items():
            lengths_here = lengths[[i,j]]
            if min(lengths_here)<.28*max(lengths_here) or max(lengths_here)>.32*max(width,height):
                continue
            vectors_here = ends[[i,j]]-corner
            d = np.linalg.norm(vectors_here,axis=2)
            if (d.min(1)>.23*lengths_here).any():
                continue
            v,h = vectors_here[np.arange(2),d.argmax(1)]
            prior = np.sqrt(min(lengths_here))*np.log1p(min(lines[i]['strength'],lines[j]['strength']))
            elbows.append((prior,corner,v,h))
    for prior,corner,v,h in sorted(elbows,key=lambda e:-e[0])[:180]:
        shifts=np.linspace(.86,1.28,13)
        shifts_v=shifts[:,None]*v
        shifts_h=shifts[:,None]*h
        vsteps=_stripe(field,corner+shifts_v,corner+shifts_v+h,2.)
        hsteps=_stripe(field,corner+shifts_h,corner+shifts_h+v,2.)
        vx,hx=(int(np.argmax(response-.11*abs(shifts-1))) for response in (vsteps,hsteps))
        weakest=min(vsteps[vx],hsteps[hx])
        if weakest<.024 or max(vsteps[vx],hsteps[hx])<.055:
            continue
        v,h=shifts_v[vx],shifts_h[hx]
        origin=corner
        if v[1]<0:
            origin=origin+v;v=-v
        if h[0]<0:
            origin=origin+h;h=-h
        quad=origin+UNIT[:,0,None]*h+UNIT[:,1,None]*v
        x,y=quad.mean(0).round().astype(int)
        if 0<=x<width and 0<=y<height and mask[y,x]:
            found.append((.76*prior*min(1.,weakest/.07),quad))
    kept = []
    for score, quad in sorted(found, key=lambda f:-f[0]):
        low,high=quad.min(0),quad.max(0)
        area=np.prod(high-low)
        def overlap(other):
            a,b=other.min(0),other.max(0)
            shared=np.prod(np.maximum(0.,np.minimum(b,high)-np.maximum(a,low)))
            return shared/(area+np.prod(b-a)-shared)
        if any(overlap(other)>.51 for _,other in kept):
            continue
        kept.append((score, quad))
        if len(kept) == 90:
            break
    return kept


def _stripe(field, start, end, spread, steps=13):
    """A signed contrast step persistent along the segment, excluding the junctions."""
    vector = end-start
    normal = np.stack((-vector[..., 1], vector[..., 0]), axis=-1)
    normal /= np.maximum(.001, np.linalg.norm(normal, axis=-1, keepdims=True))
    sample = start[...,None,:]+vector[...,None,:]*np.linspace(.13,.87,steps)[:,None]
    delta = normal[...,None,:]*spread
    def at(offset):
        return bilinear(field, sample+offset*delta)[...,0]
    # A linear illumination ramp gives the same step on both flanks. Remove
    # it, leaving a change concentrated at the hypothesized border.
    a, b = at(-1), at(1)
    across = (b-a) - .5*((a-at(-3))+(at(3)-b))
    signed = np.median(across, axis=-1)
    consistent = np.mean(np.sign(signed[...,None])*across > 0., axis=-1)
    return abs(signed)*np.maximum(0.,2*consistent-1.)


def _vanishing_directions(lines, size):
    """Find two horizontal-world line families independently of facade proposals."""
    w,h=size
    ends=np.array([[s['p'],s['q']] for s in lines],float)
    if len(ends)<4:
        return []
    mid=ends.mean(1); axes=ends[:,1]-ends[:,0]
    lengths=np.linalg.norm(axes,axis=1)
    use=(lengths>.031*max(w,h))&(abs(axes[:,1])<.89*lengths)
    mid, axes, lengths=mid[use],axes[use],lengths[use]
    axes=axes/lengths[:,None]
    index=np.argsort(-lengths)[:95]
    i,j=np.triu_indices(len(index),1)
    i,j=index[i],index[j]
    denom=_cross(axes[i],axes[j])
    good=abs(denom)>.045
    i,j,denom=i[good],j[good],denom[good]
    vp=mid[i]+axes[i]*(_cross(mid[j]-mid[i],axes[j])/denom)[:,None]
    vp=vp[(abs(vp[:,0]-w/2)<8*w)&(vp[:,1]>-.5*h)&(vp[:,1]<h*.95)]
    chosen=[]
    for _ in range(2):
        if not len(vp) or len(axes)<3:
            break
        winners=[]
        for points in np.array_split(vp,max(1,len(vp)//180)):
            rays=points[:,None]-mid
            distance=np.linalg.norm(rays,axis=-1)
            error=abs(_cross(rays,axes))/np.maximum(1.,distance)
            votes=np.maximum(0.,1-error/.032)*(distance>lengths*.6)
            scores=votes@np.minimum(lengths,w*.3)
            k=int(scores.argmax())
            winners.append((scores[k],points[k]))
        score,p=max(winners,key=lambda e:e[0])
        if score<w*.36:
            break
        chosen.append(p)
        rays=p-mid
        keep=abs(_cross(rays,axes))>.065*np.linalg.norm(rays,axis=1)
        mid,axes,lengths=mid[keep],axes[keep],lengths[keep]
        vp=vp[np.linalg.norm(vp-p,axis=1)>.20*w]
    return chosen


def _junctions(field, quad, vps, third_axis=None):
    edges = np.roll(quad,-1,axis=0)-quad
    lengths = np.linalg.norm(edges, axis=1)
    sides = edges/lengths[:,None]
    span = min(lengths)
    spread = np.clip(span*.025,1.4,3.0)
    # Search both signs of the image depth direction. An oblique return must
    # visibly depart from both projected facade axes.
    base = np.arctan2(edges[0,1],edges[0,0])
    vanishing=None
    if len(vps)==2:
        # The facade-width family follows both lintel and sill. The other
        # horizontal-world vanishing point predicts a signed local depth ray.
        errors=[]
        for vp in vps:
            rays=vp-quad[[0,2]]
            errors.append(np.mean(abs(_cross(rays,sides[[0,2]]))/np.maximum(1.,np.linalg.norm(rays,axis=1))))
        if max(errors)-min(errors)>.025:
            vanishing=vps[int(np.argmax(errors))]
    offsets = np.arange(max(9., .13*span), min(100.,.82*span)+2, max(2.,.045*span))
    if not len(offsets):
        return []
    trials = []
    for i,corner in enumerate(quad):
        if third_axis is not None:
            vp = np.asarray(third_axis, dtype=float)
            ray = vp[:2] - corner*vp[2]
            if np.linalg.norm(ray)<1.e-8:
                continue
            if vp[2]<0:
                ray=-ray
            # An endpoint can be displaced a few pixels by sampling or blur.
            # Search nearby directions, then apply a pixel-space tolerance that
            # tightens in angular units as the connector grows longer.
            angles=np.arctan2(ray[1],ray[0])+np.deg2rad(np.arange(-8.,9.,2.))
            if abs(vp[2])<1.e-10:
                angles=np.r_[angles,angles+np.pi]
        elif vanishing is None:
            angles=base+np.deg2rad(np.arange(-56,57,4))
            angles=np.r_[angles,angles+np.pi]
        else:
            diff=vanishing-corner
            angles=np.arctan2(diff[1],diff[0])+np.deg2rad(np.arange(-6,7,2))
        direction=np.stack((np.cos(angles),np.sin(angles)),axis=-1)
        shift=(direction[:,None,:]*offsets[None,:,None]).reshape(-1,2)
        away1,away2 = sides[i],-sides[(i-1)%4]
        # Positive local coordinates point into the quadrilateral.
        coeff = shift@np.linalg.inv(np.stack((away1,away2)))
        valid = np.min(np.abs(coeff), axis=1) > max(2.,spread*1.35)
        if third_axis is not None:
            miss=abs(_cross(shift,ray))/np.linalg.norm(ray)
            valid &= miss < 2.+.05*np.linalg.norm(shift,axis=1)
        ray = shift[valid]
        coeff = coeff[valid]
        if not len(ray):
            continue
        end = corner+ray
        normal = ray/np.linalg.norm(ray,axis=1,keepdims=True)
        after = end+normal*np.clip(np.linalg.norm(ray,axis=1)*.45,5,13)[:,None]
        height,width = field.shape[:2]
        visible = ((end>=4)&(end<[width-4,height-4])&(after>=4)&(after<[width-4,height-4])).all(1)
        diagonal = _stripe(field,corner,end,spread)
        cont = _stripe(field,end,after,spread,7)
        arm = np.clip(span*.32,9.,28.)
        first = _stripe(field,end,end+arm*away1,spread)
        second = _stripe(field,end,end+arm*away2,spread)
        # Favor two returning edges meeting at a common endpoint. The diagonal
        # should stop there. A single ongoing stripe is a weak junction.
        weight = np.minimum(diagonal, .7*np.maximum(first,second)+.3*np.minimum(first,second))
        weight *= np.clip(1.20-.85*cont/(diagonal+.012),.12,1)
        weight *= np.clip(np.minimum(first,second)/.065,.18,1.)
        weight = np.where(visible,weight,0)
        for label,valid_sign in [('recess',(coeff>0).all(1)),('protrusion',(coeff<0).any(1))]:
            if not valid_sign.any():
                continue
            ix = np.argmax(np.where(valid_sign,weight,-1))
            trials.append(dict(label=label,corner=i,weight=float(weight[ix]),
                ray=normal[ix],line=[corner.tolist(),end[ix].tolist()],
                cross_step=round(float(diagonal[ix]),4),after=round(float(cont[ix]),4),
                arm_steps=np.round([first[ix],second[ix]],4).tolist()))
    return trials


def _flat_repeat(field,quad):
    # Test for common narrow peaks in the wall and panel spacing on both axes.
    # A smooth, featureless panel cannot provide this affirmative evidence.
    h = transform(quad)
    axis = np.arange(-16,65)/48
    u,v = np.meshgrid(axis,axis)
    xyz = project(h,np.stack((u,v),axis=-1))
    valid = ((xyz>=1)&(xyz<[field.shape[1]-1,field.shape[0]-1])).all(-1)
    interior = (u>.07)&(u<.93)&(v>.07)&(v<.93)
    exterior = (u<-.04)|(u>1.04)|(v<-.04)|(v>1.04)
    samples = bilinear(field,xyz)[...,0]
    detail = samples-.25*(np.roll(samples,2,0)+np.roll(samples,-2,0)+
                           np.roll(samples,2,1)+np.roll(samples,-2,1))
    for dim in (0,1):
        spectra = []
        for mask in (interior,exterior):
            scores = []
            for lag in range(4,25):
                a,b = np.moveaxis(detail,dim,0)[:-lag],np.moveaxis(detail,dim,0)[lag:]
                ok = np.moveaxis(mask&valid,dim,0)
                used = ok[:-lag]&ok[lag:]
                x,y = a[used],b[used]
                if len(x)<60 or min(x.std(),y.std())<.014:
                    scores.append(0.)
                else:
                    x,y = x-x.mean(),y-y.mean()
                    scores.append(float(x@y / (np.linalg.norm(x)*np.linalg.norm(y))))
            spectra.append(np.array(scores))
        p,q=spectra
        shared=np.minimum(p,q)
        k=int(shared.argmax())
        if k in (0,len(p)-1) or shared[k]<.55 or min(p[k]-np.median(p),q[k]-np.median(q))<.34:
            return False
    return True


def _best_junctions(trials):
    best = {}
    for t in trials:
        score=t['weight']
        if t['label']=='protrusion':
            # The cap has visible connectors departing multiple corners
            # along the same direction; a lone outward stripe is weak.
            other=max((v['weight'] for v in trials if v['label']==t['label']
                and v['corner']!=t['corner'] and t['ray']@v['ray']>.95), default=0.)
            score=min(score,1.5*other)
        if score>best.get(t['label'],(0.,None))[0]:
            best[t['label']]=(score,t)
    return tuple(best.get(k,(0.,None)) for k in ('recess','protrusion'))


def _aligned_connector(t,axis):
    start,end=np.asarray(t['line'])
    ray=axis[:2]-start*axis[2]
    delta=end-start
    miss=abs(_cross(delta,ray))/max(1.e-12,np.linalg.norm(ray))
    return miss<=2.+.05*np.linalg.norm(delta)


def analyze(image,lines,outline_mask,cube_axes=None):
    """Run after whole-image tracing. Inputs and returned quadrilaterals use source pixels."""
    mask = np.asarray(outline_mask.convert('L'))>0
    if mask.shape != (image.height,image.width):
        raise ValueError('Outline mask must use source image coordinates')
    gray = image.convert('L').filter(ImageFilter.GaussianBlur(.9))
    field = np.log(np.asarray(gray,dtype=float)+8.)[:,:,None]
    directions = _vanishing_directions(lines,image.size)
    bbox=outline_mask.getbbox()
    if bbox is None:
        return []
    from engine.structural_lines import detect
    rgb=image.convert('RGB')
    region=Image.composite(rgb,rgb.resize((1,1)).resize(rgb.size),outline_mask.convert('L'))
    crop=region.crop(bbox).filter(ImageFilter.GaussianBlur(max(.9,.0016*max(image.size))))
    local=detect(crop)
    for segment in local:
        for key in ('p','q'):
            x,y=segment[key]
            segment[key]=[x+bbox[0],y+bbox[1]]
    results = []
    for face_score,quad in _faces([*lines,*local],mask,field):
        third_axis=None
        evidence={}
        measured_quad=quad
        if cube_axes:
            from engine.cube_faces import fit_face
            fitted=fit_face(quad,cube_axes)
            if fitted is None:
                results.append(dict(quad=quad.round(2).tolist(),label='unknown',
                    face_score=round(float(face_score),3),cube_turn=dict(status='unsupported')))
                continue
            quad,turn=fitted
            third_axis=cube_axes[turn['third_axis']]
            evidence['cube_turn']=turn
        observed = _junctions(field,measured_quad,directions)
        inward,outer = _best_junctions(observed)
        flat_possible = max(inward[0],outer[0]) < .075
        if third_axis is not None:
            compatible=[t for t in observed if _aligned_connector(t,np.asarray(third_axis))]
            compatible.extend(_junctions(field,measured_quad,directions,third_axis))
            inward,outer=_best_junctions(compatible)
        winner,runner=(inward,outer) if inward[0]>=outer[0] else (outer,inward)
        row=dict(quad=quad.round(2).tolist(),label='unknown',face_score=round(float(face_score),3),
                 inward_score=round(inward[0],4),outward_score=round(outer[0],4),**evidence)
        if winner[0]>.09 and winner[0]>1.45*runner[0]+.035:
            t=winner[1]
            row.update(label=t['label'],connector=np.round(t['line'],2).tolist(),
                       cross_gradient=t['cross_step'],continuation=t['after'],end_arms=t['arm_steps'])
        elif flat_possible and winner[0]<.075 and _flat_repeat(field,measured_quad):
            row.update(label='flat',evidence='matching exterior and interior texture periods on both axes')
        results.append(row)
    return results
