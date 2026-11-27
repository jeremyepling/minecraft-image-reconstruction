"""Find unassigned pixels in a connected patch of the same near-uniform material."""
import numpy as np

from engine.camera_lines import _runs
from engine.surface_perimeter import _native_alpha


def proposals(image, parts, *, exterior_veto=False, row_veto=False, patch_veto=False):
    natives = [_native_alpha(p['support'],image.size) for p in parts]
    boxes = [item[1] for item in natives if item is not None]
    if not boxes:
        return []
    # A connected area reaching this outside margin provides no closed visible extent.
    left,top = np.maximum(0,np.min(boxes,axis=0)[:2]-12)
    right,bottom = np.minimum(image.size,np.max(boxes,axis=0)[2:]+12)
    full = np.asarray(image.convert('RGB'),np.int16)
    rgb = full[top:bottom,left:right]
    exterior = np.concatenate((full[::4,:8].reshape(-1,3),full[::4,-8:].reshape(-1,3),
                               full[:8,::4].reshape(-1,3),full[-8:,::4].reshape(-1,3)))
    exterior = np.unique(exterior,axis=0)
    exterior_chroma = 128*exterior.astype(float)/np.maximum(1,exterior.mean(axis=1,keepdims=True))
    owners = np.zeros(rgb.shape[:2],np.int32)
    colors = {}
    for i,native in enumerate(natives,1):
        if native is None:
            continue
        alpha,(x,y,r,b) = native
        mask=np.asarray(alpha,bool)
        box=np.s_[y-top:b-top,x-left:r-left]
        window=owners[box]
        window[mask]=np.where(window[mask]==0,i,-1)
        inside=rgb[box][::3,::3][mask[::3,::3]]
        if len(inside)<80:
            continue
        palette,n=np.unique(inside,axis=0,return_counts=True)
        for ix in np.argsort(-n)[:4]:
            if n[ix]>=16:
                colors.setdefault(tuple(map(int,palette[ix])),set()).add(i)
    pending=np.zeros_like(owners)
    for color,sponsors in colors.items():
        tone = np.asarray(color,dtype=float)
        if exterior_veto and np.any(np.max(abs(exterior_chroma-128*tone/max(1,tone.mean())),axis=1)<=2.5):
            continue
        similar=abs(rgb-np.array(color,np.int16)).max(axis=2)<=3
        for runs in _runs(similar):
            n=sum(b-a for y,a,b in runs)
            if n<500 or any(y==0 or y+1==similar.shape[0] or a==0 or b==similar.shape[1] for y,a,b in runs):
                continue
            votes=np.bincount(np.concatenate([owners[y,a:b] for y,a,b in runs])+1,minlength=len(parts)+2)
            host=int(votes[2:].argmax())+1
            if host not in sponsors or votes[host+1]<max(320,.20*n) or votes[1]<64:
                continue
            if votes[host+1]<.985*(n-votes[1]):
                continue
            if row_veto:
                y0=top+min(y for y,_,_ in runs)
                y1=top+max(y for y,_,_ in runs)+1
                sides=full[y0:y1:2,::4]
                margins=np.concatenate((sides[:,:8],sides[:,-8:]),axis=1).reshape(-1,3)
                palette,frequency=np.unique(margins,axis=0,return_counts=True)
                palette=palette[frequency>=max(4,.005*len(margins))].astype(float)
                chroma=128*palette/np.maximum(1,palette.mean(axis=1,keepdims=True))
                if np.any(abs(chroma-128*tone/max(1,tone.mean())).max(axis=1)<=2.5):
                    continue
            for y,a,b in runs:
                available=owners[y,a:b]==0
                old=pending[y,a:b]
                old[available]=np.where((old[available]==0)|(old[available]==host),host,-1)
    events=[]
    for host in np.unique(pending):
        if host<=0:
            continue
        for runs in _runs(pending==host):
            area=sum(b-a for y,a,b in runs)
            if area<64:
                continue
            spans=[[int(y+top),int(a+left),int(b+left)] for y,a,b in runs]
            events.append(dict(surface=parts[host-1]['id'],pixels=area,
                               box=[min(a for y,a,b in spans),min(y for y,a,b in spans),
                                    max(b for y,a,b in spans),1+max(y for y,a,b in spans)],runs=spans))
    return [e for e in events if not _resembles_exterior(full,e)] if patch_veto else events


def _resembles_exterior(rgb, event):
    # Compare the *unassigned pixels* with common colors at that height.
    # Distant pavement changes color rapidly at the horizon. A shadow can
    # darken the same pavement without materially changing its chromaticity.
    matching=0
    for y,a,b in event['runs']:
        strips=rgb[max(0,y-1):y+2]
        margin=np.concatenate((strips[:,:32:2],strips[:,-32::2]),axis=1).reshape(-1,3)
        patch=np.median(rgb[y,a:b],axis=0)
        gain=patch.mean()/np.maximum(1,margin.mean(axis=1))
        same=(abs(margin-patch).max(axis=1)<=3)
        shade=(gain>=.48)&(gain<=.90)&(abs(margin*gain[:,None]-patch).max(axis=1)<=1.8)
        if (same|shade).mean()>=.12:
            matching+=b-a
    return matching>=.35*event['pixels']
