"""Experimental oblique recess measurement using the visible unit pavement grid."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.recess_review import known_openings
from engine.depth_layers import fit_reveal, depth_fingerprints
from engine.geometry import bilinear, project, UNIT
from engine.remap import structure_mask
from engine.rim_spacing import texture_detail,local_spacing


def pavement(rgb):
    h,w=rgb.shape[:2]
    if min(w,h)<128:
        return None
    sky=np.median(np.concatenate((rgb[:,:w//12],rgb[:,-w//12:]),axis=1),axis=1)
    chroma=sky[:,2].astype(float)-sky[:,0]
    ys=np.arange(h//7,2*h//3)
    horizon=int(ys[np.argmax(chroma[ys-3]-chroma[ys+3])])
    if chroma[horizon-4]-chroma[horizon+4]<12:
        return None
    ys=np.linspace(.81*h,.985*h,26).astype(int)
    yref=float(ys.mean())
    xref=np.arange(.14*w,.86*w,1.)
    stretch=(ys[:,None]-horizon)/(yref-horizon)
    gray=rgb.mean(2)
    residual=(np.roll(gray,6,1)+np.roll(gray,-6,1))/2-gray
    def profile(v):
        xx=v+(xref[None,:]-v)*stretch
        i=np.rint(xx).astype(int)
        ok=(i>=7)&(i<w-7)
        return (residual[ys[:,None],np.clip(i,0,w-1)]*ok).sum(0)/np.maximum(1,ok.sum(0))
    def strength(v):
        p=profile(v)
        return np.mean(np.clip(p,0,80)**2)
    vps=[]; gaps=[]
    for side in (-1,1):
        grid=w/2+side*w*np.linspace(.18,3,95)
        coarse=grid[np.argmax([strength(v) for v in grid])]
        fine=np.linspace(coarse-.04*w,coarse+.04*w,50)
        v=fine[np.argmax([strength(n) for n in fine])]
        vps.append(v)
    # A finite render distance displaces the visible sky-ground boundary.
    # Recover the geometric horizon where BOTH floor line families converge.
    initial_y=horizon
    def refine(candidate_heights,start):
        nonlocal horizon,stretch
        trials=[]
        for yy in candidate_heights:
            horizon=yy
            stretch=(ys[:,None]-yy)/(yref-yy)
            predicted=w/2+(np.asarray(start)-w/2)*(yref-yy)/(yref-initial_y)
            pair=[];total=0.
            for guess in predicted:
                vv=guess+np.linspace(-.014*w,.014*w,15)
                ss=[strength(z) for z in vv]
                j=int(np.argmax(ss));pair.append(vv[j]);total+=np.log(max(.01,ss[j]))
            trials.append((total,yy,pair))
        return max(trials,key=lambda t:t[0])
    best=refine(np.arange(max(.04*h,horizon-.17*h),horizon+.025*h,6),vps)
    _,horizon,vps=refine(np.arange(best[1]-4,best[1]+5),vps)
    stretch=(ys[:,None]-horizon)/(yref-horizon)
    for v in vps:
        p=profile(v)
        detail=p-p.mean()
        ac=np.correlate(detail,detail,mode='full')[len(p)-1:]/np.arange(len(p),0,-1)
        lags=np.arange(12,min(230,len(p)//4))
        peaks=lags[(ac[lags]>ac[lags-1])&(ac[lags]>=ac[lags+1])]
        if not len(peaks):return None
        accepted=peaks[ac[peaks]>.72*ac[peaks].max()]
        if not len(accepted) or ac[accepted[0]]<.32*ac[0]:return None
        period=accepted[0]
        # Refine the fundamental using the phase coherence of strong line centers.
        periods=np.linspace(period-1.5,period+1.5,43)
        weights=np.maximum(p-np.quantile(p,.65),0)**2
        phases=np.exp(2j*np.pi*xref[None,:]/periods[:,None])@weights
        idx=abs(phases).argmax();period=periods[idx]
        shift=np.angle(phases[idx])*period/(2*np.pi)
        centers=np.arange(shift,1.1*w,period)
        gaps.append((centers,yref))
    focal_sq=-(vps[0]-w/2)*(vps[1]-w/2)-(horizon-h/2)**2
    if not (.4*h)**2<focal_sq<(2.5*h)**2: return None
    f=focal_sq**.5
    K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
    kinv=np.linalg.inv(K)
    directions=[kinv@[v,horizon,1] for v in vps]
    directions=[v/np.linalg.norm(v) for v in directions]
    up=np.cross(*directions); up*=1 if up[1]<0 else -1
    def ground(xy):
        ray=np.c_[xy,np.ones(len(xy))]@kinv.T
        return -ray/(ray@up)[:,None]
    phases=[];heights=[]
    for k,(x,y) in enumerate(gaps):
        xyz=ground(np.c_[x,np.full(len(x),y)])@directions[1-k]
        heights.append(1/np.median(abs(np.diff(xyz))))
        phases.append(xyz)
    if min(heights)/max(heights)<.90: return None
    height=float(np.mean(heights))
    offsets=[np.angle(np.exp(2j*np.pi*xyz*height).mean())/(2*np.pi) for xyz in reversed(phases)]
    return K,kinv,directions,up,ground,height,offsets,dict(horizon=horizon,vps=vps,focal=f,height=height)


def dark_boxes(seeds):
    boxes=[]
    ny,nx=seeds.shape
    for j,i in zip(*seeds.nonzero()):
        if not seeds[j,i]:continue
        seeds[j,i]=False; cells=[]; todo=[(j,i)]
        while todo:
            y,x=todo.pop();cells.append((x,y))
            for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=yy<ny and 0<=xx<nx and seeds[yy,xx]:
                    seeds[yy,xx]=False;todo.append((yy,xx))
        lo=np.min(cells,axis=0);hi=np.max(cells,axis=0)+1
        if min(hi-lo)>=2 and len(cells)>=.7*np.prod(hi-lo):
            boxes.append((lo,hi))
    return boxes


def internal_repeat(field,lo,hi,vp,distance,depth,pitch,spacing=(1.,1.)):
    """Compare adjacent motifs *inside* the apparent back face at two candidate pitches.

    Both measurements sample the same visible rectangle after excluding the
    reveals. Neither needs a reference patch outside the opening or its ID.
    """
    fraction = 1/(1+depth/distance)
    lower=np.maximum(lo,vp+(lo-vp)*fraction)+.07*pitch
    upper=np.minimum(hi,vp+(hi-vp)*fraction)-.07*pitch
    results=[[],[]]
    axes=[]
    for axis in (0,1):
        step=np.eye(2)[axis]*pitch*spacing[axis]
        limits=upper-step
        if np.any(limits-lower<.30*pitch):
            continue
        x,y=np.meshgrid(np.arange(lower[0],limits[0],1.75),np.arange(lower[1],limits[1],1.75))
        origin=np.stack((x.ravel(),y.ravel()),axis=1)
        if len(origin)<48:continue
        a=bilinear(field,origin).ravel()
        a-=a.mean()
        for j,k in enumerate((1.,fraction)):
            b=bilinear(field,origin+k*step).ravel()
            b-=b.mean()
            results[j].append(float(a@b/max(1.,np.linalg.norm(a)*np.linalg.norm(b))))
        axes.append('xy'[axis])
    # A strongly conserved native pattern at zero depth is positive evidence
    # for a flat insert. Weak or absent fingerprints provide no veto.
    zero=max(results[0],default=0.)
    inset=max(results[1],default=0.)
    return dict(axes=axes,plane=results[0],inset=results[1],
                veto=zero>.50 and zero>inset+.05)


def infer(image):
    rgb=np.array(image.convert('RGB')); mask=structure_mask(image)
    cal=pavement(rgb)
    if not cal:return dict(reason='no stable pavement calibration',candidates=[])
    K,kinv,directions,up,ground,height,offsets,info=cal
    xs=np.flatnonzero(mask.any(0))[::3]
    if len(xs)<5:return dict(reason='insufficient facade silhouette',candidates=[])
    ys=mask.shape[0]-mask[::-1,xs].argmax(0)
    pts=ground(np.c_[xs,ys])*height
    # The most common integer-depth ground contact defines the front facade.
    fits=[]
    for axis,d in enumerate(directions):
        units=pts@d-offsets[axis]
        line=np.rint(units).astype(int)
        u,count=np.unique(line,return_counts=True)
        i=count.argmax(); fits.append((count[i]/len(line),axis,u[i]+offsets[axis]))
    vote,k,z=max(fits)
    if vote<.40:return dict(reason='non-planar base',calibration=info,candidates=[])
    forward=directions[k];side=directions[1-k]
    base=np.abs(pts@forward-z)<.5
    xx=pts[base]@side-offsets[1-k]
    low,high=np.rint(np.quantile(xx,[.005,.995])).astype(int)
    nx=high-low
    if not 3<=nx<=100:return dict(reason='width unresolved',candidates=[])
    ox=low+offsets[1-k]
    # Derive facade height by intersecting the top silhouette with the wall.
    top=mask[:,xs].argmax(0)
    rays=np.c_[xs,top,np.ones(len(xs))]@kinv.T
    units=(rays*(z/(rays@forward))[:,None])@up+height
    ny=int(np.rint(np.median(units[base])))
    if not 3<=ny<=60:return dict(reason='height unresolved',candidates=[])
    H=K@np.stack([side,-up,side*ox+forward*z+(ny-height)*up],axis=1)
    scale=36
    full=np.stack(np.meshgrid(np.arange(nx*scale)+.5,np.arange(ny*scale)+.5),axis=-1)
    source_xy=project(H,full/scale)
    rect=bilinear(rgb,source_xy).astype(np.uint8)
    mean=rect.reshape(ny,scale,nx,scale,3)[:,6:-6,:,6:-6].mean(axis=(1,3,4))
    seeds=mean<np.quantile(mean,.75,axis=1,keepdims=True)*.91
    G=np.diag([scale,scale,1.]); vp_grid=np.linalg.solve(H,K@forward)
    vp=vp_grid[:2]/vp_grid[2]*scale
    # Rectification makes all back planes shrink toward one finite epipole.
    ratio=abs(z); out=[]
    boxes=dark_boxes(seeds.copy())
    front=(bilinear(mask[:,:,None],source_xy)[:,:,0]>.99)
    front &= ~seeds.repeat(scale,axis=0).repeat(scale,axis=1)
    for a,b in boxes:
        low=np.maximum(0,a*scale-4); high=b*scale+4
        front[low[1]:high[1],low[0]:high[0]]=False
    field=texture_detail(rect)
    for a,b in boxes:
        q=(a+UNIT*(b-a))*scale
        fingerprints=depth_fingerprints(rect,G,nx,ny,a,b,vp,1.,ratio)
        r=fit_reveal(rect.mean(2),q,vp,1.,ratio,fingerprints)
        base_depth=r['depth']
        test=(internal_repeat(field,a*scale,b*scale,vp,ratio,base_depth,scale)
              if base_depth else None)
        calibration=local_spacing(field,front,a*scale,b*scale,scale)
        spacing=[c['ratio'] if c else 1. for c in calibration]
        corrected=internal_repeat(field,a*scale,b*scale,vp,ratio,base_depth,scale,spacing) if base_depth and all(calibration) else test
        out.append(dict(depth=None if corrected and corrected['veto'] else base_depth,
                        prior_depth=None if test and test['veto'] else base_depth,
                        local_spacing=calibration,corrected_repeat=corrected,
                        initial_depth=base_depth,internal_repeat=test,
                        rim_size=(b-a).tolist(),
                        rim=project(H,q/scale).tolist()))
    return dict(calibration=info,grid=[int(nx),int(ny)],candidates=out)


def run(source,controls,out,split=None,positive=None):
    records={}
    for root in ([source,controls,positive] if positive else [source,controls]):
        if root!=source and split=='dev':
            continue
        scenes={b['id']:b for b in json.loads((root/'buildings.json').read_text())}
        for view in json.loads((root/'views.json').read_text()):
            if view['angle'] not in ('left','right','oblique-left','oblique-right') or (root==source and
                    (view['kind']!='recess' or view['split']=='train' or
                     (split and view['split']!=split))):continue
            path=root/'captures'/(view['id']+'.png')
            with Image.open(path) as im: result=infer(im)
            known=known_openings(scenes[view['building_id']],json.loads(path.with_suffix('.json').read_text())) if root!=controls else {}
            for candidate in result['candidates']:
                xy=np.mean(candidate['rim'],axis=0)
                hits=[(name,t) for name,(inv,t) in known.items() if ((project(inv,xy)>=0)&(project(inv,xy)<=1)).all()]
                candidate['target']=hits[0] if hits else None
            result['truth']={name:t for name,(_,t) in known.items()}
            records[view['id']]=result
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(records,indent=2)+'\n')
    for name,r in records.items():
        print(name,r.get('grid',r.get('reason')),[(c['target'],c['depth']) for c in r['candidates']])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--source',type=Path)
    mode.add_argument('--image',type=Path,help='Inspect an individual screenshot without fixture metadata')
    parser.add_argument('--controls',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=['dev','test'])
    parser.add_argument('--positive',type=Path,help='Independent live positive control containing genuine recesses')
    a=parser.parse_args()
    if a.image:
        with Image.open(a.image) as im:
            result=infer(im)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    elif a.controls:
        run(a.source,a.controls,a.out,a.split,a.positive)
    else:
        parser.error('--source needs --controls')
