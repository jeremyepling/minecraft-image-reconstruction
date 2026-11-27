"""Condition the anchor and neighboring subface matcher on the projected native pixel footprint."""
from collections import Counter,defaultdict
from functools import lru_cache
from unittest.mock import patch
import numpy as np
from benchmark import full_block_lattice as grid
from benchmark.view_conditioned_atlas import draw_templates,footprint_uv
from engine.geometry import face,project,transform
from engine.atlas_count import _shortlist
from engine.thin_atlas_faces import _corr


_BASE_KNOWN=grid.known_anchors
_BASE_UNKNOWN=grid.sprite_candidates
_BASE_GROUP=grid.occupancy
_LAST={}


@lru_cache(maxsize=1)
def gray_tiles():
    return grid.atlas().patterns.reshape(-1,16,16)


def face_frame(linear,xyz,axis):
    # Origin at the lower unit-cell corner. The optical center can lie on either side.
    origin=xyz.min(0).copy()
    if xyz[0,axis]<0:
        origin[axis]-=1
    return np.column_stack((linear,linear@origin))


def pixels(quad,part,shape):
    h,w=shape[:2]
    lo=np.maximum(np.floor(quad.min(0)).astype(int),0)
    hi=np.minimum(np.ceil(quad.max(0)).astype(int),[w,h])
    if np.any(hi<=lo):
        return None
    n=np.prod(hi-lo)
    stride=max(1,int(np.sqrt(n/460)))
    x,y=np.meshgrid(np.arange(lo[0],hi[0],stride),np.arange(lo[1],hi[1],stride))
    xy=np.column_stack((x.ravel(),y.ravel()))+.5
    uv=project(np.linalg.inv(transform(quad)),xy)
    keep=((uv>.06)&(uv<.94)).all(1) & part['mask'][y.ravel()//4,x.ravel()//4]
    if keep.sum()<18:
        return None
    xy=xy[keep]
    if len(xy)>180:
        xy=xy[np.linspace(0,len(xy)-1,180,dtype=int)]
    return xy


def projected(P,xy,cell,axis,rgb,choices):
    coords=xy.astype(int)
    gray=rgb[coords[:,1],coords[:,0]].astype('f').mean(1)
    if gray.std()<1.8 or len(gray)<12:
        return None
    uv=footprint_uv(P,xy,cell,axis,samples=2)
    patterns=draw_templates(gray_tiles()[choices],uv)
    scores=_corr(patterns,gray)
    i=int(scores.argmax())
    best=scores[i]
    one=patterns[i]
    # Interleaved sampling and spatial halves must retain local appearance.
    split=np.arange(len(gray))%2==0
    halves=[float(_corr(one[s],gray[s])) for s in (split,~split)]
    for coord in (0,1):
        s=xy[:,coord]<np.median(xy[:,coord])
        if min(s.sum(),(~s).sum())>=6:
            halves.extend(float(_corr(one[m],gray[m])) for m in (s,~s))
    point=draw_templates(gray_tiles()[choices[i:i+1]],footprint_uv(P,xy,cell,axis,samples=1))[0]
    return dict(corr=float(best),weakest=min(halves),key=int(choices[i]),point=float(_corr(point,gray)))


def coarse_choices(rgb,quad):
    raw=face(rgb,quad,16)
    short,prior=_shortlist(raw,np.ones((16,16),bool),grid.atlas())
    values=raw.astype('f').mean(2)
    scores=_corr(grid.atlas().patterns[short].reshape(-1,256),values.ravel()).reshape(-1,8)
    scores-=prior[short,None]/60
    order=np.argsort(-scores.ravel(),kind='stable')[:8]
    return short[order//8]*8+order%8


def refine_anchor(a,part,rgb,linear,adjust):
    xyz=np.array(a['xyz'])
    axis=a['axis']
    free=[i for i in range(3) if i!=axis]
    center=xyz.mean(0)
    offset=xyz-center
    choices=coarse_choices(rgb,np.array(a['quad']))
    proposals=[(1.,0.,0.)]
    if adjust:
        proposals.extend([(t,0.,0.) for t in (.96,1.04)])
        proposals.extend([(1.,x,y) for x,y in ((.0625,0),(-.0625,0),(0,.0625),(0,-.0625))])
    winner=None
    for scale,x,y in proposals:
        guess=center*scale+offset
        guess[:,free]+=np.array([x,y])
        quad=project(np.column_stack((linear,np.zeros(3))),guess)
        xy=pixels(quad,part,rgb.shape)
        if xy is None:
            continue
        P=face_frame(linear,guess,axis)
        m=projected(P,xy,[0,0,0],axis,rgb,choices)
        if m is None or m['corr']<.59 or m['weakest']<.37:
            continue
        strength=m['corr']+.12*m['weakest']-.008*int(scale!=1 or x!=0 or y!=0)
        if winner is None or strength>winner[0]:
            tile=m['key']//8
            winner=strength,dict(a,quad=quad.tolist(),xyz=guess.tolist(),corr=round(m['corr'],5),
                texture=grid.atlas().labels[tile],block=grid.atlas().ids[tile][0].removeprefix('minecraft:'),
                turn=m['key']%8,point_corr=round(m['point'],5),weakest=round(m['weakest'],5),
                adjusted=scale!=1 or x!=0 or y!=0,range_scale=scale,shift_uv=[x,y])
    return winner[1] if winner else None


def raw_verified(rgb,quad):
    return 'unresolved','unresolved',0.


def known(part,rgb,P,eye,levels):
    with patch.object(grid,'verify_sprite',raw_verified):
        raw=_BASE_KNOWN(part,rgb,P,eye,levels)
    return [new for a in raw if (new:=refine_anchor(a,part,rgb,P[:,:3],False))]


def unknown(part,rgb,linear):
    with patch.object(grid,'verify_sprite',raw_verified):
        raw=_BASE_UNKNOWN(part,rgb,linear)
    return [new for a in raw if (new:=refine_anchor(a,part,rgb,linear,True))]


def cut_trial(level,axis,eye,linear,xy,colors,choices):
    # Keep world UV origin fixed. Translate only the plane normal coordinate so the
    # sampler's boundary coincides with a measured eighth-, half- or whole-block level.
    cell=np.zeros(3,dtype=int)
    cell[axis]=int(np.floor(level-1e-8 if eye[axis]>level else level+1e-8))
    edge=cell[axis]+int(eye[axis]>level)
    origin=-eye.copy()
    origin[axis]+=level-edge
    P=np.column_stack((linear,linear@origin))
    uv=footprint_uv(P,xy,cell,axis,samples=2)
    templates=draw_templates(gray_tiles()[choices],uv)
    scores=_corr(templates,colors)
    best=int(scores.argmax())
    strength=float(scores[best])
    split=np.arange(len(xy))%2==0
    halves=[float(_corr(templates[best][m],colors[m])) for m in (split,~split)]
    return strength,min(halves),int(choices[best])


def group(parts,anchors,eye,linear,*,observe=None):
    records=[]
    def capture(surface,cell,axis,level,y,x):
        records.append((surface,cell,axis,level,y,x))
    base,shared,junctions=_BASE_GROUP(parts,anchors,eye,linear,observe=capture)
    if not records:
        return base,shared,junctions
    anchor_ids={}
    for a in anchors:
        c=np.array(a['xyz']).mean(0)+eye
        ax=a['axis']
        c[ax]-=np.sign(eye[ax]-c[ax])*.02
        anchor_ids[(a['surface'],tuple(np.floor(c+1e-7).astype(int)))]=a['block']
    selected={}
    owners=defaultdict(set)
    used=Counter()
    trace=[]
    def put(surface,cell,axis,level,y,x,name):
        selected.setdefault(cell,name)
        if (surface,cell) in anchor_ids:
            selected[cell]=anchor_ids[(surface,cell)]
        owners[cell].add(surface)
        if observe is not None:
            observe(surface,cell,axis,level,y,x)
    rgb=_LAST['rgb']
    cat=grid.atlas()
    for surface,cell,axis,level,y,x in records:
        if (surface,cell) in anchor_ids or len(x)<12:
            put(surface,cell,axis,level,y,x,base[cell])
            continue
        # Texture+pitch is compared on the exact same observed source fragment for every depth.
        choice=np.linspace(0,len(x)-1,min(72,len(x)),dtype=int)
        xy=(np.column_stack((x[choice],y[choice]))+.5)*4
        coords=xy.astype(int)
        colors=rgb[coords[:,1],coords[:,0]]
        gray=colors.astype('f').mean(1)
        if gray.std()<2:
            put(surface,cell,axis,level,y,x,base[cell])
            continue
        short,_=_shortlist(colors[:,None],np.ones((len(colors),1),bool),cat)
        options=np.unique(np.r_[level,round(level),round(level*2)/2,level-.25,level+.25])
        variants=(short[:5,None]*8+np.arange(8)).ravel()
        scores={cut:cut_trial(cut,axis,eye,linear,xy,gray,variants) for cut in options}
        best=max(options,key=lambda c:scores[c][0]-.025*abs(c-level))
        corr,weak,key=scores[best]
        used['tested']+=1
        if best==level or corr<.68 or weak<.5 or corr-scores[level][0]<.09:
            put(surface,cell,axis,level,y,x,base[cell])
            continue
        xy_all=(np.column_stack((x,y))+.5)*4
        r=grid.rays(linear,xy_all)
        hit=eye+r*((best-eye[axis])/r[:,[axis]])
        hit[:,axis]-=.01*np.sign(eye[axis]-best)
        target,votes=np.unique(np.floor(hit).astype(int),axis=0,return_counts=True)
        # Small uncertainty at a border cannot split one visible fragment into unsupported pieces.
        j=int(votes.argmax())
        if votes[j]<.85*len(hit) or max(abs(target[j]-cell))>1:
            put(surface,cell,axis,level,y,x,base[cell])
            continue
        dest=tuple(target[j].tolist())
        put(surface,dest,axis,float(best),y,x,cat.ids[key//8][0].removeprefix('minecraft:'))
        trace.append(dict(surface=surface,old=list(cell),new=list(dest),old_level=level,new_level=float(best),
                          source_corr=round(corr,4),gain=round(corr-scores[level][0],4)))
        used['changed_level']+=1
        used['changed_owner']+=dest!=cell
    _LAST.update(cuts=dict(used),edits=trace)
    return selected,sorted([list(p) for p in selected if len(owners[p])>=2]),junctions


def infer(image,parts,camera,offsets,lines,baseline):
    _LAST.clear()
    _LAST['rgb']=np.asarray(image.convert('RGB'))
    with patch.object(grid,'known_anchors',known),patch.object(grid,'sprite_candidates',unknown),patch.object(grid,'occupancy',group):
        cells,P,evidence=grid.infer(image,parts,camera,offsets,lines,baseline)
    evidence.update(cuts=_LAST.get('cuts',{}),edits=_LAST.get('edits',[]))
    return cells,P,evidence
