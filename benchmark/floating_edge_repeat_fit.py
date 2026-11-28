"""Local subcell size/phase refinement constrained by repeated native atlas evidence."""
import itertools
import numpy as np

from benchmark.expanded_wall_source_resample import TEXELS,atlas
from benchmark.floating_peer_ruler_refit import harmonic_margin,profiles,project
from benchmark.floating_verified_wall_expansion import full_face


def repeat_fit(h,material,mask,rgb):
    library=atlas()
    indices=[i for i,ids in enumerate(library.ids) if material in ids]
    if not indices:
        return h,dict(accepted=False,reason="no calibrated texture")
    patterns=library.patterns[indices].reshape(-1,256).astype("f")
    xy=np.column_stack(np.nonzero(mask)[::-1])+.5
    split=np.quantile(xy[:,0],[.4,.6])
    uv=np.unique(np.floor(project(np.linalg.inv(h),xy[(xy[:,0]>=split[0])&(xy[:,0]<=split[1])])).astype(int),axis=0)
    mid=project(h,uv+.5)[:,0]
    uv=uv[(mid>=split[0])&(mid<=split[1])&(np.max(abs(uv),axis=1)>=2)]
    tiles=np.vstack(([0,0],uv))
    def corr(ruler):
        points=project(ruler,tiles[:,None,None,:]+TEXELS)
        xx,yy=np.floor(points).astype(int).transpose(3,0,1,2)
        valid=(xx>=0)&(yy>=0)&(xx<mask.shape[1])&(yy<mask.shape[0])
        xx=xx.clip(0,mask.shape[1]-1)
        yy=yy.clip(0,mask.shape[0]-1)
        complete=(valid&mask[yy,xx]).mean((1,2))>=.985
        center=points.mean((1,2))[1:,0]
        complete[1:] &= (center>=split[0])&(center<=split[1])
        patches=rgb[yy,xx].mean(-1).reshape(len(tiles),-1).astype("f")
        patches-=patches.mean(1,keepdims=True)
        similarity=(patches@patterns.T).max(1)/(256*np.maximum(1,patches.std(1)))
        similarity[~complete]=-1
        return similarity
    choices=[]
    for sx,sy,dx,dy in itertools.product((.92,.96,1.,1.04,1.08),(.92,.96,1.,1.04,1.08),(-.0625,0,.0625),(-.0625,0,.0625)):
        trans=np.array([[sx,0,dx+.5*(1-sx)],[0,sy,dy+.5*(1-sy)],[0,0,1.]])
        trial=h@trans
        corrs=corr(trial)
        repeated=sorted(corrs[1:],reverse=True)[:2]
        score=corrs[0]+.35*sum(max(0.,v-.45) for v in repeated)-.03*(abs(sx-1)+abs(sy-1)+abs(dx)+abs(dy))
        choices.append((float(score),trial,dict(scale=[sx,sy],offset=[dx,dy],near=float(corrs[0]),middle=list(map(float,repeated)))))
    _,winner,params=max(choices,key=lambda row:row[0])
    near=full_face(winner,np.zeros(2),mask,rgb)
    alluv=np.unique(np.floor(project(np.linalg.inv(winner),xy)).astype(int),axis=0)
    x=project(winner,alluv+.5)[:,0]
    # Enumerate every distinct candidate tile in the withheld right region; keep
    # positions and materials to distinguish a distant repeat from a different material.
    far=[dict(uv=p.tolist(),**full_face(winner,p,mask,rgb)) for p in alluv[x>split[1]]]
    triangles=[harmonic_margin(p,1.,0.) for p in profiles(winner,mask,rgb,split)]
    repeat=any(s['ok'] and s['material']==material for s in far)
    accepted=bool(near["ok"] and near["material"]==material and repeat and
                  all(a["validation"]>=.06 and a["margin"]>=.015 for a in triangles))
    return winner,dict(accepted=accepted,fit=params,near=near,far=far,axes=triangles,
                       middle_candidates=len(uv),far_repeat=repeat,homography=winner.tolist())
def validate_repeats(h,record,mask,rgb):
    """Challenge each repeated sprite at half/double pitch while remaining on the same pixels."""
    near=record.get("near",{})
    if not near.get("ok"):
        return dict(accepted=False,reason="unverified origin")
    label=near["material"]
    candidates=[dict(uv=[0,0],**near)]+[p for p in record.get("far",[]) if p["ok"] and p["material"]==label]
    audits=[]
    for p in candidates:
        uv=np.array(p["uv"],float)
        alternatives=[]
        for sx,sy in ((.5,1.),(1.,.5),(.5,.5),(2.,1.),(1.,2.),(2.,2.)):
            # Compare full central alternative and both half-size halves where applicable.
            choices=lambda s: (0.,.25,.5) if s==.5 else ((-.5,) if s==2. else (0.,))
            for dx,dy in itertools.product(choices(sx),choices(sy)):
                trans=np.array([[sx,0.,uv[0]+dx],[0.,sy,uv[1]+dy],[0.,0.,1.]])
                alternate=full_face(h@trans,np.zeros(2),mask,rgb)
                if alternate.get("ok"):
                    alternatives.append(float(alternate["correlation"]))
        advantage=p["correlation"]-max(alternatives,default=0.)
        audits.append(dict(uv=p["uv"],corr=p["correlation"],best_competing=max(alternatives,default=None),
                           margin=advantage,ok=advantage>=.05))
    passing=[v["uv"] for v in audits[1:] if v["ok"]]
    spread=max((max(abs(a-b) for a,b in zip(u,v)) for u in passing for v in passing),default=0)
    across_axes=any(min(abs(u),abs(v))>=2 for u,v in passing)
    accept=bool(audits[0]["ok"] and spread>=3 and across_axes)
    return dict(accepted=accept,spread=spread,across_axes=across_axes,tests=audits)
