"""Compare rival texture coordinates on native photo pixels before replacing a prior block."""
from collections import defaultdict
import numpy as np

from engine import roof_span_texture as strips
from engine.thin_atlas_faces import _corr, _match
from engine.visible_placement import _frontmost_cells


def texels(ctx, x, y, coordinates, axis):
    rays=ctx._rays(x,y)
    free=[i for i in range(3) if i!=axis]
    plane=coordinates + (ctx.eye[axis]>coordinates+1)
    uv=ctx.eye[free] + (plane-ctx.eye[axis])[:,None]*rays[:,free]/rays[:,[axis]]
    return np.floor(uv*16).astype(int)


def supported(ctx, prior, proposed, rows, quality=.78, advantage=.18):
    if not proposed:
        return prior
    before=sorted(prior)
    original=np.asarray(before)
    old,old_axes=_frontmost_cells(before,ctx.native,ctx.eye,ctx.shape,face_axes=True)
    original_ids=set(np.unique(old)) - {-1}
    while proposed:
        after=sorted(prior|proposed)
        owner,axes,ids=ctx.joint(proposed)
        seen={after[i] for i in np.unique(owner) if i>=0}
        lost={i for i in original_ids if before[i] not in seen}
        if not lost:
            return prior|proposed
        # Every ray that used to see a disappearing block gets a competing
        # photo prediction from the old and the proposed projection.
        y,x=np.nonzero(np.isin(old,list(lost)))
        owners=old[y,x]
        explained=np.zeros(len(y),bool)
        candidates=owner[y,x]
        groups=defaultdict(list)
        for j,(i,a,b) in enumerate(zip(candidates,axes[y,x],old_axes[y,x])):
            if i>=0 and (after[i],a) in rows:
                groups[i,int(a),int(b)].append(j)
        for (i,a,b),idx in groups.items():
            if len(idx)<16 or after[i] not in proposed:
                continue
            idx=np.asarray(idx)
            if len(idx)>350:
                fitted=idx[np.linspace(0,len(idx)-1,350,dtype=int)]
            else:
                fitted=idx
            u,v=x[fitted],y[fitted]
            colors=ctx.rgb[v,u]
            gray=colors.mean(1)
            coords=texels(ctx,u,v,np.full(len(u),after[i][a]),a)%16
            bins=16*coords[:,1]+coords[:,0]
            if len(np.unique(bins))<10 or gray.std()<1:
                continue
            new=rows[after[i],a]
            template=ctx.maps[new['sprite'],new['mip'],new['turn']]
            # The new sprite and its orientation were selected on the entire
            # face. Give the prior geometry a fresh fit on the contested rays.
            pred=template[bins]
            new_score=float(_corr(pred,gray))
            if new_score<quality:
                continue
            old_bins=texels(ctx,u,v,original[owners[fitted],b],b)
            match=_match(old_bins,colors,ctx.atlas,ctx.maps,list(ctx.anchors))
            if match:
                previous=match['corr']
            else:
                # Too little color variance or too few projected texels to
                # identify an old sprite. Give each old texel a free intensity.
                oid=(old_bins[:,1]%16)*16+(old_bins[:,0]%16)
                _,group,n=np.unique(oid,return_inverse=True,return_counts=True)
                fitted_gray=(np.bincount(group,weights=gray)/n)[group]
                previous=float(_corr(fitted_gray,gray))
            if new_score-previous<advantage:
                continue
            # Match on the *individual* native pixels as well. Protect an old
            # block if even one of its original rays has no winning evidence.
            u,v=x[idx],y[idx]
            allcoords=texels(ctx,u,v,np.full(len(u),after[i][a]),a)%16
            expected=template[16*allcoords[:,1]+allcoords[:,0]]
            gain=np.std(gray)/max(.1,np.std(pred))
            residual=np.abs(ctx.rgb[v,u].mean(1)-(gray.mean()+gain*(expected-pred.mean())))
            explained[idx] = residual <= max(4,1.5*gray.std())
        unresolved=set(owners[~explained])
        if not unresolved:
            return (prior-{before[i] for i in lost})|proposed
        blockers={after[i] for i in candidates[np.isin(owners,list(unresolved))] if i>=0}
        retained=proposed-blockers
        if retained==proposed:
            return prior
        proposed=retained
    return prior


def recover(image,parts,counts,camera,prior,*,atlas_path=None):
    ctx, rows=strips.observations(image,parts,camera,prior,atlas_path)
    selected=strips.screened(ctx,rows,prior,'conservative')
    known=supported(ctx,prior,selected,rows)
    return known,dict(replaced=len(prior-known),added=len(known-prior))
