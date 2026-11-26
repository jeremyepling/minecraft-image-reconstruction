"""Trial native-ray and vanilla texture evidence for edge-on horizontal faces."""
from collections import defaultdict
import numpy as np

from engine.visible_edge_faces import native_support
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE


def match(colors, uv, atlas):
    if len(colors)<14:return (0.,0.,-1)
    tex=np.floor((uv%1)*16).astype(int).clip(0,15)
    if len(np.unique(tex,axis=0))<10:return (0.,0.,-1)
    gray=colors.astype(np.float32).mean(1)
    if gray.std()<.5:return (0.,0.,-1)
    mean=colors.astype(float).mean(0)
    gain=np.clip((atlas.means@mean)/np.maximum(1.,(atlas.means**2).sum(1)),.3,1.2)
    loss=abs(atlas.means*gain[:,None]-mean).mean(1)+3*abs(np.log(gain/.80))
    ids=np.argpartition(loss,16)[:16]
    patterns=atlas.patterns[ids].reshape(-1,16,16)
    best=(0.,0.,-1)
    for mip in (1,2,4):
        size=16//mip
        ref=patterns.reshape(-1,size,mip,size,mip).mean((2,4))
        x,y=(tex//mip).T
        # Separate repeated screen observations by texture location before verifying.
        half=((tex[:,0]//4+tex[:,1]//4)%2).astype(bool)
        if min(half.sum(),(~half).sum())<5:continue
        def corr(i,j,sel):
            a=ref[:,j[sel],i[sel]];a=a-a.mean(1,keepdims=True)
            b=gray[sel]-gray[sel].mean()
            return (a@b)/np.maximum(1e-6,np.linalg.norm(a,axis=1)*np.linalg.norm(b))
        score=np.minimum(corr(x,y,half),corr(x,y,~half))-np.repeat(loss[ids],8)/65
        k=score.argmax()
        whole=np.ones(len(half),bool)
        wrong=max(corr((x+max(1,size//4))%size,y,whole)[k],corr(x,(y+max(1,size//4))%size,whole)[k])
        candidate=(float(score[k]),float(corr(x,y,whole)[k]-wrong),int(ids[k//8]))
        if candidate[0]>best[0]:best=candidate
    return best


def proposals(image, parts, P, cells, offsets, atlas, *, union=False):
    if atlas is None or not offsets or not cells:return {}
    P=np.asarray(P,float)
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    inverse=np.linalg.inv(P[:,:3])
    rgb=np.asarray(image.convert('RGB'))
    mask_by_id={p['id']:native_support(p['support'],image.size) for p in parts if p.get('support')}
    if not mask_by_id:return {}
    native_P=P.copy();native_P[:2]*=SAMPLE_STRIDE
    owner=_frontmost_cells(sorted(cells),native_P,eye,rgb.shape[:2])
    union_mask=np.logical_or.reduce(list(mask_by_id.values()))
    by_level=defaultdict(list)
    for r in offsets:
        if r['axis']==1 and r['id'] in mask_by_id:
            for level in (r['level']-1,r['level'],r['level']+1):
                if level>0 and .04<abs(level-eye[1])<2.6:
                    by_level[level].append(mask_by_id[r['id']])
    if union:
        # Only seek very thin visible tops/undersides at heights of observed side blocks.
        for _,y,_ in cells:
            for level in (y,y+1):
                if level>0 and .04<abs(level-eye[1])<1.6:
                    by_level.setdefault(level,[])
    out={}
    for level,masks in by_level.items():
        area=union_mask if union else np.logical_or.reduce(masks)
        y,x=np.nonzero(area & (owner<0))
        rays=np.column_stack((x+.5,y+.5,np.ones(len(x))))@inverse.T
        depth=np.divide(level-eye[1],rays[:,1],out=np.zeros(len(x)),where=abs(rays[:,1])>1.e-8)
        good=(depth>2)&(depth<65)&(abs(rays[:,1])<.14)
        if good.sum()<14:continue
        y,x=y[good],x[good]
        uv=eye[[0,2]]+rays[good][:,[0,2]]*depth[good,None]
        xy,inv,counts=np.unique(np.floor(uv).astype(int),axis=0,return_inverse=True,return_counts=True)
        order=np.argsort(inv.ravel(),kind='stable');ends=counts.cumsum()
        height=level-int(eye[1]>level)
        # The photograph must contain a face witness on this height near the new cell.
        anchors=np.asarray([(a,c) for a,b,c in cells if b==height],int)
        if not len(anchors):continue
        for i in np.flatnonzero(counts>=14):
            a,c=xy[i];cell=(int(a),int(height),int(c))
            if cell in cells or abs(anchors-xy[i]).sum(1).min()>8:continue
            group=order[ends[i]-counts[i]:ends[i]]
            score,phase,tile=match(rgb[y[group],x[group]],uv[group],atlas)
            out[cell]=dict(score=score,phase=phase,tile=tile,n=len(group),level=level,
                           source_pixels=y[group]*rgb.shape[1]+x[group],
                           mask_pixels=sum(int(m[y[group],x[group]].sum()) for m in masks))
    return out


def verified(image,parts,P,cells,offsets,atlas,*,union=False):
    rows=proposals(image,parts,P,cells,offsets,atlas,union=union)
    strong={p:r for p,r in rows.items() if r['score']>.74 and r['phase']>.16}
    if not strong:return set()
    # Require a second independently read cell of the same texture close to the candidate.
    corroborated={p for p,r in strong.items() if any(p!=q and p[1]==q[1] and
        abs(p[0]-q[0])+abs(p[2]-q[2])<=2 and rr['tile']==r['tile'] for q,rr in strong.items())}
    if not corroborated:return set()
    points=sorted(cells|corroborated)
    K=np.asarray(P,float).copy()
    eye=np.linalg.solve(K[:,:3],-K[:,3]);K[:2]*=SAMPLE_STRIDE
    owner,axes=_frontmost_cells(points,K,eye,image.size[::-1],face_axes=True)
    owners=owner.ravel()
    directions=axes.ravel()
    accepted=set()
    for i,q in enumerate(points):
        if q in corroborated:
            observed=strong[q]['source_pixels']
            if np.count_nonzero((owners[observed]==i)&(directions[observed]==1))>=14:
                accepted.add(q)
    return accepted
