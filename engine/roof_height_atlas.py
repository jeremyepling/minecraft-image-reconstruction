"""Locate an observed horizontal patch on the integer height grid before assigning pixels to blocks."""
import numpy as np

from engine.atlas_phase import ATLAS_SHA256
from engine.camera_atlas_floor import _atlas
from engine.surface_model import intrinsic
from engine.surface_perimeter import _native_alpha
from engine.thin_atlas_faces import _corr, _match, _mips
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import SAMPLE_STRIDE


def observed_tops(parts,size,rotation,P,eye,prior):
    """Locate upper ends of vertical runs visible on a photographed support."""
    cells=sorted(prior)
    owner,axes=_frontmost_cells(cells,P,eye,((size[1]+3)//4,(size[0]+3)//4),face_axes=True)
    tips={}
    for part in parts:
        normal=part.get('normal_camera')
        if normal is None or not part.get('support'):
            continue
        axis=int(np.argmax(abs(rotation.T @ normal)))
        if axis==1:
            continue
        support=part['support']
        width,height=support['size']
        mask=np.zeros((height,width),bool)
        for yy,left,right in support['runs']:
            mask[yy,left:right]=True
        ys=((4*np.arange(owner.shape[0])+2)*height//size[1]).clip(max=height-1)
        xs=((4*np.arange(owner.shape[1])+2)*width//size[0]).clip(max=width-1)
        hits=owner[(owner>=0)&(axes==axis)&mask[np.ix_(ys,xs)]]
        visible={cells[i] for i in np.flatnonzero(np.bincount(hits,minlength=len(cells))>=4)}
        for x,y,z in visible:
            if (x,y-1,z) in visible and (x,y+1,z) not in visible:
                tips.setdefault(y+1,set()).add((x,z))
    return tips


def support_span(rays,eye,h,tips):
    """A nearby post must touch this surface in the inferred horizontal grid."""
    sample=rays[np.linspace(0,len(rays)-1,min(900,len(rays)),dtype=int)]
    footprint=set(map(tuple,np.floor(coords(eye,sample,h)).astype(int).tolist()))
    block_y=h-int(eye[1]>h)
    # A top course touches the wall below it; an exposed wall can also terminate
    # flush with the top surface of a taller build.
    sites=tips.get(block_y,set()) | tips.get(h,set())
    contacts=[(x,z) for x,z in sites if any((x+dx,z+dz) in footprint
              for dx,dz in ((0,0),(1,0),(-1,0),(0,1),(0,-1)))]
    if len(contacts)<2:
        return 0
    return int(np.ptp(np.asarray(contacts),axis=0).max())


def horizontal_pixels(part, size, rotation):
    normal=part.get('normal_camera')
    if not part.get('support') or normal is None or np.argmax(abs(rotation.T @ normal))!=1:
        return None
    region=_native_alpha(part['support'],size)
    if region is None:
        return None
    mask, (left,top,_,_)=region
    y,x=np.nonzero(mask)
    if len(x)<48:
        return None
    return x+left,y+top


def coords(eye,rays,h):
    return eye[[0,2]] + (h-eye[1])*rays[:,[0,2]]/rays[:,[1]]


def indices(xz):
    bins=np.floor(16*xz).astype(np.int32)%16
    return 16*bins[:,1]+bins[:,0]


def rank_planes(x,y,rays,colors,eye,heights,atlas,maps):
    # Identical sampling and alternative heights for both selection and validation.
    sample=np.random.default_rng(173).choice(len(x),min(2400,len(x)),replace=False)
    sx,sy=x[sample],y[sample]
    rays=rays[sample]
    colors=colors[sample]
    gray=colors.mean(1)
    order=np.argsort(sx if np.ptp(sx)>np.ptp(sy) else sy)
    halves=np.array_split(order,2)
    if gray.std()<1:
        return []
    rows=[]
    for h in heights:
        depth=(h-eye[1])/rays[:,1]
        if np.percentile(depth,2)<1 or np.percentile(depth,98)>180:
            continue
        xz=coords(eye,rays,h)
        match=_match(np.floor(xz*16).astype(int),colors,atlas,maps)
        if match is None:
            continue
        template=maps[match['sprite'],match['mip'],match['turn']]
        signal=template[indices(xz)]
        corrs=[float(_corr(signal[t],gray[t])) for t in halves]
        pixel_corr=float(_corr(signal,gray))
        value=.65*pixel_corr+.35*min(corrs)-match['color']/125.
        rows.append(dict(plane_y=int(h),value=round(value,4),pixel_corr=round(pixel_corr,4),
                         halves=[round(c,4) for c in corrs],**match))
    return sorted(rows,key=lambda r:r['value'],reverse=True)


def placement_pixels(x,y,rays,colors,eye,decision,maps,strict):
    h=decision['plane_y']
    uv=coords(eye,rays,h)
    xz=np.floor(uv).astype(np.int32)
    groups,ids,counts=np.unique(xz,axis=0,return_inverse=True,return_counts=True)
    signal=maps[decision['sprite'],decision['mip'],decision['turn']][indices(uv)]
    gray=colors.mean(1)
    order=np.argsort(ids,kind='stable')
    cells={}
    for i,pixels in enumerate(np.split(order,np.cumsum(counts)[:-1])):
        if len(pixels)<(12 if strict else 6):
            continue
        model, observed=signal[pixels],gray[pixels]
        if min(model.std(),observed.std())<.5:
            continue
        quality=float(_corr(model,observed))
        if quality<(.67 if strict else .47):
            continue
        c=int(groups[i,0]),decision['block_y'],int(groups[i,1])
        cells[c]=(x[pixels],y[pixels])
    return cells


def apply_planes(size,P,eye,prior,witnesses,union,strict):
    # owner indices address sorted geometry only. Every proposed cube must ultimately
    # expose its own measured roof pixels after competing cubes are rendered together.
    proposed=set(witnesses)-prior
    if not proposed:
        return prior
    native=P.copy()
    native[:2]*=SAMPLE_STRIDE
    shape=size[::-1]
    cells=sorted(prior)
    old=_frontmost_cells(cells,native,eye,shape)
    old_seen=np.bincount(old[old>=0],minlength=len(cells))
    replaced=np.zeros(shape,bool)
    for cell,(x,y) in witnesses.items():
        hits=old[y,x]
        different=np.ones(len(hits),bool)
        if cell in prior:
            different=hits!=cells.index(cell)
        replaced[y[different],x[different]]=True
    ids=old[replaced & (old>=0)]
    hits=np.bincount(ids,minlength=len(cells))
    removals={cells[i] for i in np.flatnonzero((hits>=8)&(hits>=.97*old_seen)) if cells[i] not in witnesses}
    # Slight discrepancies at the boundary of the learned silhouette are tolerated.
    padded=np.pad(union,1)
    height,width=shape
    margin=np.logical_or.reduce([padded[j:j+height,i:i+width] for i in range(3) for j in range(3)])
    kept=proposed.copy()
    while kept:
        scene=sorted((prior-removals)|kept)
        front,axes=_frontmost_cells(scene,native,eye,shape,face_axes=True)
        visible=front>=0
        seen=np.bincount(front[visible],minlength=len(scene))
        spill=np.bincount(front[visible & ~margin & (old<0)],minlength=len(scene))
        number={cell:i for i,cell in enumerate(scene)}
        verified=np.zeros(shape,bool)
        valid=set()
        for cell in kept|set(witnesses).intersection(scene):
            x,y=witnesses[cell]
            i=number[cell]
            ok=(front[y,x]==i)&(axes[y,x]==1)
            n=int(ok.sum())
            if n >= (9 if strict else 4) and n>=.06*seen[i] and spill[i]==0:
                valid.add(cell)
                verified[y[ok],x[ok]]=True
        # Revert a proposed deletion unless its original native pixels actually
        # resolve to a measured roof face of a different block in the new scene.
        witnessed=old[verified & (old>=0)]
        votes=np.bincount(witnessed,minlength=len(cells))
        confirmed={c for i,c in enumerate(cells) if c in removals and votes[i]>=.95*old_seen[i]}
        lost={c for i,c in enumerate(cells) if c not in removals and old_seen[i]>0 and seen[number[c]]==0}
        for cell in lost:
            i=cells.index(cell)
            if votes[i]>=.98*old_seen[i] and cell not in witnesses:
                confirmed.add(cell)
            else:
                blockers=np.unique(front[old==i])
                valid.difference_update(scene[j] for j in blockers if j>=0)
        next_kept=valid & kept
        if next_kept==kept and confirmed==removals:
            return (prior-removals)|kept
        if next_kept==kept and confirmed>removals:
            # All additional deletions are fully occluded in the already rerendered scene.
            return (prior-confirmed)|kept
        # Never re-enable a rejected removal. A deleted block with no prior
        # pixels carries no vote and cannot be removed by this pass.
        removals &= confirmed
        kept=next_kept
    return prior


def recover(image,parts,counts,camera,prior,*,atlas_path=None,mode='loose',course_override=None,place_cells=True):
    status=dict(height_decisions=[],primary_height=None)
    if not prior or camera.get('P') is None or not parts:
        return prior,status
    atlas=_atlas('26.3',atlas_path)
    if atlas is None or atlas.provenance.get('atlas_sha256')!=ATLAS_SHA256:
        return prior,status
    P=np.asarray(camera['P'],float)
    P=P/np.linalg.norm(P[2,:3])
    inverse=np.linalg.inv(P[:,:3])
    eye=-inverse@P[:,3]
    focal=np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
    rotation=np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    maps=None
    tips=None
    heights=range(1,min(64,max(p[1] for p in prior)+6)+1)
    rgb=np.asarray(image.convert('RGB'))
    witnesses={}
    strict=mode=='strict'
    for part in parts:
        pixels=horizontal_pixels(part,image.size,rotation)
        if pixels is None:
            continue
        x,y=pixels
        rays=np.column_stack((x+.5,y+.5,np.ones(len(x))))@inverse.T
        if np.min(abs(rays[:,1]))<1e-6:
            continue
        if maps is None:
            maps=_mips(atlas)
        colors=rgb[y,x]
        ranked=rank_planes(x,y,rays,colors,eye,heights,atlas,maps)
        if not ranked:
            continue
        best=ranked[0]
        h=best['plane_y']
        gap=best['value']-ranked[1]['value'] if len(ranked)>1 else best['value']
        accepted=(best['value']>=(.49 if strict else .38) and gap>=(.13 if strict else .08)
                  and best['phase']>.06 and best['color']<14 and best['texels']>=16)
        span=0
        fallback=False
        if not accepted and mode in ('supported','supported_wide'):
            wide=mode=='supported_wide'
            if (best['value']>=(.16 if wide else .25) and gap>=(.05 if wide else .08)
                    and best['phase']>.04 and best['color']<16 and best['texels']>=16):
                if tips is None:
                    tips=observed_tops(parts,image.size,rotation,P,eye,prior)
                span=support_span(rays,eye,h,tips)
                fallback=span>=3
                accepted=fallback
        if course_override is not None:
            # Keep the experimental patch-admission gate identical. Only the
            # physical plane used to index its native texels and emit cells changes.
            forced=course_override(part,eye,prior)
            selected=next((b for b in ranked if b['plane_y']==forced),None)
            if selected is None:
                accepted=False
            else:
                best=selected
            h=int(forced)
        row=dict(part=part['id'],plane_y=h,block_y=h-int(eye[1]>h),normal_y=1 if eye[1]>h else -1,
                 area=len(x),accepted=bool(accepted),margin=round(gap,4),pixel_corr=best['pixel_corr'],
                 hypotheses=[[b['plane_y'],b['value']] for b in ranked])
        if mode in ('supported','supported_wide'):
            row.update(support_fallback=bool(fallback),support_span=span)
        status['height_decisions'].append(row)
        if accepted and place_cells:
            # A block observed by two independent patches can use both witnesses.
            for c,coords2d in placement_pixels(x,y,rays,colors,eye,dict(best,block_y=row['block_y']),maps,strict).items():
                if c in witnesses:
                    coords2d=tuple(np.concatenate(pair) for pair in zip(witnesses[c],coords2d))
                witnesses[c]=coords2d
    accepted=[r for r in status['height_decisions'] if r['accepted']]
    if accepted:
        status['primary_height']=max(accepted,key=lambda r:(r['area']*max(0,r['pixel_corr']),r['margin']))
    if not (witnesses.keys()-prior):
        return prior,status
    union=np.zeros(image.size[::-1],bool)
    for part in parts:
        if part.get('support'):
            patch=_native_alpha(part['support'],image.size)
            if patch:
                mask,(x,y,x1,y1)=patch
                union[y:y1,x:x1]|=np.asarray(mask)>0
    return apply_planes(image.size,P,eye,prior,witnesses,union,strict),status


def strict(image,parts,counts,camera,prior):
    return recover(image,parts,counts,camera,prior,mode='strict')


def tallest(image,parts,counts,camera,prior):
    def course(part,eye,existing):
        level=max(p[1] for p in existing)
        return level+int(eye[1]>level+1)
    return recover(image,parts,counts,camera,prior,course_override=course)


def supported(image,parts,counts,camera,prior):
    return recover(image,parts,counts,camera,prior,mode='supported_wide')
