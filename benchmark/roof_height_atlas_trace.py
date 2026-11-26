"""Explain losses between selected height, local texture and rendered visibility on measured canopy roofs."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
from benchmark import framed_depth_eval as ev
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_canopy_spacing import shift
from benchmark.texture_camera_eval import checked, read, save
from engine import roof_height_atlas as model


def trace(split,f,base,prior,scoring):
    image,parts,counts,camera=ev.photo(split,f,base)
    truth,occupied,T,_=ev.references('old',f,ev.NATIVE_INPUTS[split],ev.TRAINING_DATA,ev.DEPTH,
                                    dict(old=ev.transfer.BUILDS))
    top=ev.roof_specs()[f['id']]['roof_y']
    roof={q for q in truth if q[1]==top}
    delta=scoring['roof']['spacing']['offset']
    assert delta is not None
    def matches(local):
        return len(shift(registered(local,T),delta)&roof)
    pools={'integer_plane':set(),'local_texture':set(),'unoccluded':set(),'silhouette':set()}
    conflicts=[]
    materialize=model.placement_pixels
    apply=model.apply_planes
    def measure(x,y,rays,colors,eye,decision,maps,strict):
        uv=model.coords(eye,rays,decision['plane_y'])
        xz,n=np.unique(np.floor(uv).astype(int),axis=0,return_counts=True)
        pools['integer_plane'].update((int(g[0]),decision['block_y'],int(g[1])) for g in xz[n>=6])
        measured=materialize(x,y,rays,colors,eye,decision,maps,strict)
        pools['local_texture'].update(measured)
        return measured
    def joint(size,P,eye,cells,witnesses,union,strict):
        result=apply(size,P,eye,cells,witnesses,union,strict)
        native=P.copy()
        native[:2]*=model.SAMPLE_STRIDE
        scene=sorted(cells|set(witnesses))
        front,axes=model._frontmost_cells(scene,native,eye,size[::-1],face_axes=True)
        old=model._frontmost_cells(sorted(cells),native,eye,size[::-1])
        padded=np.pad(union,1)
        height,width=size[::-1]
        margin=np.logical_or.reduce([padded[j:j+height,i:i+width] for i in range(3) for j in range(3)])
        for c,(x,y) in witnesses.items():
            i=scene.index(c)
            good=(front[y,x]==i)&(axes[y,x]==1)
            pixels=front==i
            if good.sum()>=4 and good.sum()>=.06*pixels.sum():
                pools['unoccluded'].add(c)
                if not (pixels & ~margin & (old<0)).any():
                    pools['silhouette'].add(c)
        # Scoring-only counterfactual: render all individually viable additions,
        # allow them to hide older cells, and inspect what those old pixels saw.
        before=sorted(cells)
        viable=cells | pools['silhouette']
        after=sorted(viable)
        front,axes=model._frontmost_cells(after,native,eye,size[::-1],face_axes=True)
        before_seen=set(before[i] for i in np.unique(old) if i>=0)
        after_seen=set(after[i] for i in np.unique(front) if i>=0)
        for cell in sorted(before_seen-after_seen):
            region=old==before.index(cell)
            target=next(iter(shift(registered({cell},T),delta)))
            incoming={after[i] for i in np.unique(front[region]) if i>=0} - cells
            conflicts.append(dict(local=cell,correct_at_post_alignment=target in truth,
                prior_pixels=int(region.sum()),new_horizontal=int((axes[region]==1).sum()),
                new_vertical=int(np.isin(axes[region],(0,2)).sum()),
                roof_additions_landing_on_surface=matches(incoming)))
        return result
    with patch.object(model,'placement_pixels',measure),patch.object(model,'apply_planes',joint):
        after,status=model.supported(image,parts,counts,camera,prior)
    return dict(id=f['id'],split=split,anchor=delta,target=len(roof),original=matches(prior),
                correct_previously_missing={k:matches(v-prior) for k,v in pools.items()},
                gained_at_prior_anchor=matches(after-prior),lost_at_prior_anchor=matches(prior-after),
                decisions=status['height_decisions'],disappearing_original_blocks=conflicts)


def run(prior_path,out):
    lookup={(r['split'],r['id']):r for r in read(prior_path/'cases.json')}
    rows=[]
    for split,f,base in ev.cases(('old','fresh','extra')):
        if f['id'] not in ev.roof_specs():
            continue
        r=lookup[split,f['id']]
        prior=set(decoded_pack(checked(prior_path/split/f['id']/'reconstruction.zip',r['pack_sha256'])))
        rows.append(trace(split,f,base,prior,r))
    save(out,dict(rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.prior,a.out)
