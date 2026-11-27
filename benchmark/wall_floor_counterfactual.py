"""Compare projected foreground with the ground texture phase learned outside the build."""
import numpy as np

from benchmark.wall_source_projection import STEP,Scene


def table(uv,colors,near):
    tile=np.zeros((256,3))
    enough=np.zeros(256,bool)
    s=uv[near]
    rgb=colors[near]
    order=np.argsort(s,kind='stable')
    if not len(order):
        return tile,enough
    for indices in np.split(order,np.flatnonzero(np.diff(s[order]))+1):
        i=s[indices[0]]
        if len(indices)>=8:
            tile[i]=np.median(rgb[indices],axis=0)
            enough[i]=True
    return tile,enough


def evaluate(image,parts,camera,prior):
    if not prior or not parts or camera.get('P') is None:
        return {}
    scene=Scene(image,parts,camera,prior)
    h,w=scene.shape
    yy,xx=np.indices((h,w))
    rays=np.column_stack((xx.ravel()*STEP+.5,yy.ravel()*STEP+.5,np.ones(h*w)))@scene.inv.T
    with np.errstate(divide='ignore',invalid='ignore'):
        depth=-scene.eye[1]/rays[:,1]
    good=np.isfinite(depth)&(depth>2)&(depth<120)
    uv=np.zeros(h*w,int)
    xz=scene.eye[[0,2]]+rays[good][:,[0,2]]*depth[good,None]
    texel=np.floor(xz*16).astype(int)%16
    uv[good]=texel[:,0]+16*texel[:,1]
    outside=good&(scene.owner<0)&~scene.union
    colors=scene.rgb.reshape(-1,3).astype(float)
    # Two disjoint spatial halves guard against fitting a different object as ground.
    half=((xx//12+yy//12)%2).ravel().astype(bool)
    levels=np.floor(np.log2(np.clip(depth,2,120))*2)
    bins={}
    for level in np.unique(levels[good&(scene.owner>=0)]):
        near=outside&(abs(levels-level)<=1)
        if near.sum()<2048:
            continue
        maps=[]
        for fold in (half,~half):
            tile,enough=table(uv,colors,near&fold)
            if enough.mean()<.90:
                break
            maps.append(tile)
        if len(maps)!=2:
            continue
        agree=np.median(abs(maps[0]-maps[1]))
        tile=np.mean(maps,axis=0)
        if agree<=2 and tile.mean(1).std()>=2:
            bins[level]=tile
    output={}
    for n,idx in scene.owned.items():
        sel=idx[good[idx]]
        if len(sel)<16 or len(sel)<.98*len(idx):
            continue
        level=np.median(levels[sel])
        if not bins:
            continue
        b=min(bins,key=lambda a:abs(a-level))
        if abs(b-level)>1:
            continue
        tile=bins[b]
        tx=uv[sel]
        valid_bins=np.unique(tx)
        if len(valid_bins)<12:
            continue
        expected=tile[tx]
        actual=colors[sel]
        # Same lighting as the neighboring pavement or a mild shadow.
        gain=np.clip(np.median((actual.mean(1)+1)/(expected.mean(1)+1)),.72,1.05)
        err=abs(actual-gain*expected).mean(1)
        intensity=actual.mean(1)
        intensity-=intensity.mean()
        def correlation(ids):
            gray=tile[ids].mean(1)
            gray-=gray.mean()
            return (gray@intensity)/max(1,np.linalg.norm(gray)*np.linalg.norm(intensity))
        corr=correlation(tx)
        rivals=[correlation(16*(tx//16)+(tx+x)%16) for x in (2,4,8)]
        rivals += [correlation((tx+16*y)%256) for y in (2,4,8)]
        wall=scene.cells[n]
        output[wall]=dict(pixels=len(sel),distinct=len(valid_bins),corr=round(float(corr),4),
                          margin=round(float(corr-max(rivals)),4),mae=round(float(err.mean()),4),
                          high_error=round(float((err>8).mean()),4))
    return output


def recover(image,parts,counts,camera,prior):
    evidence=evaluate(image,parts,camera,prior)
    removed={c for c,v in evidence.items() if v['corr']>=.70 and v['margin']>=.12 and
             v['mae']<=4 and v['high_error']<=.06 and v['distinct']>=16}
    return prior-removed,dict(verified_floor_pixels=[dict(cell=c,**evidence[c]) for c in sorted(removed)])
