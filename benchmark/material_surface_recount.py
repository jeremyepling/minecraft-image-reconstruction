"""Recount and re-export every seam accepted by the guarded source-only feedback rule."""
import json
from pathlib import Path

import numpy as np

from PIL import ImageFilter

from benchmark import concrete_shadow_boundaries as seams, reference_surface_feedback as replay, typed_structures as typed
from benchmark.relative_geometry import best_translation
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.thin_face_placement_run import ref_inputs
from engine import eave_pinholes, placed_materials, smooth_atlas_outline
from engine.plane_consensus import with_mask
from engine.surface_perimeter import with_outline

OUT = Path(__file__).resolve().parent/'reviews/material-feedback-surfaces-01'


def exported(im,parts,camera,transform,truth,optional):
    measured = eave_pinholes.measurement_parts(smooth_atlas_outline.measurement_parts(parts))
    counts = seams.viewer.recount(im, measured, None)
    cells = replay.downstream(im,parts,counts,camera)
    ids,evidence = placed_materials.assign(im,camera,cells,seams.viewer.calibrated_atlas())
    _,decoded = typed.typed_pack(ids)
    world = typed.aligned_sites(list(decoded),transform,(0,0,0))
    offset = best_translation(world,truth)[0] if world else (0,0,0)
    world = typed.aligned_sites(list(decoded),transform,offset)
    placed = dict(zip(world,decoded.values(),strict=True))
    assigned = {p for p,c in zip(world,decoded,strict=True) if evidence[c]!='fallback'}
    return dict(geometry=typed.geometry_score(placed,truth,optional),typed=typed.score(placed,truth,optional,assigned),
                complete_grids=sum(None not in r['dimensions']['blocks'] for r in counts['surfaces']))


def main():
    records={r['id']:r for r in typed.read(OUT/'cases.json') if r['guarded_plane_and_id']['joins']}
    jobs=seams.suite.jobs([*seams.suite.framed.SIZES,'shelter'])
    release,expected=replay.releases()
    result=[]
    for job in jobs:
        key=job[1]['id']
        if key not in records:
            continue
        im,parts,_,camera=seams.viewer.structure_stages(job,expected[key])
        gray=np.asarray(im.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
        lookup={p['id']:p for p in parts}
        revised=list(parts)
        # Pair identities are from the source-only rule. The third field in the
        # comparison record labels the result and never participates in this edit.
        for left,right,_ in records[key]['guarded_plane_and_id']['joins']:
            a,b=lookup[left],lookup[right]
            masks=[seams.native_mask(p,im.size) for p in (a,b)]
            donor=(a,b)[int(masks[1].sum()>masks[0].sum())]
            donor={k:v for k,v in donor.items() if not k.startswith('measured_')}
            union=with_outline(with_mask(donor,masks[0]|masks[1],im.size,gray,np.ones(2)))
            revised=[union if p['id']==donor['id'] else p for p in revised if p['id']==donor['id'] or p['id'] not in (left,right)]
        labels,rays=seams.visible_faces(job[0],job[1])
        truth={tuple(b['xyz_world']):b['block_id'] for b in labels['blocks']}
        _,_,transform,_=ref_inputs(job)
        optional=horizontal_slivers(labels,rays)
        result.append(dict(id=key, archived_geometry=release[key]['geometry'],source_hash=records[key]['rgb_sha256'],
                           before=exported(im,parts,camera,transform,truth,optional),
                           after=exported(im,revised,camera,transform,truth,optional)))
    assert len(result)==len(records)
    typed.compact(OUT/'closed-loop.json',result)


if __name__=='__main__':
    main()
