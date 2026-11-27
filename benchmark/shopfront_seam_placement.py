"""Recount and replay affected seam unions against verified ordinary placements."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import numpy as np

from benchmark import facade_inset_round as suite, pavement_opening_eval as placement
from benchmark.edge_vote_outline_eval import baseline_pack
from benchmark.mask_blocks import registered
from benchmark.relative_geometry import placement_scores
from benchmark.shopfront_seam_continuity import predict
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import read, save
from benchmark.thin_face_placement_run import ref_inputs
from engine import pavement_opening
from engine.pavement_obstructions import window_obstructions
from engine.pillar_seams import native_mask


OUT=Path('benchmark/reviews/shopfront-seam-continuity-01')


def build(image,parts,saved_counts,camera):
    cells,revised,dims=placement.build(image,parts,saved_counts,camera)
    matched=set(pavement_opening.detect(image,parts,camera))
    if matched:
        cells-=window_obstructions(image.size,camera,[p for p in parts if p['id'] in matched],cells)
    return cells,dims


def grade(task):
    job,expected,phase=task
    s,f,_,_=job
    key=f['id']
    image,base,counts,camera=suite.ordinary_inputs(*job)
    changed,pairs=predict(image,base,camera,protect_junctions=True,phase_only=phase)
    selected=[p for p in pairs if p['accepted']]
    assert [p['inputs'] for p in selected]==[p['inputs'] for p in expected if p['accepted']],key
    pack,prior=baseline_pack(job)
    assert 'colonnade-01' not in key, 'Use current pavement archive for this case'
    orig,old_dims=build(image,base,counts,camera)
    assert orig==prior,(key,len(orig-prior),len(prior-orig))
    cells,new_dims=build(image,changed,counts,camera)
    truth,occupied,transform,_=ref_inputs(job)
    labels,rays=placement.visible_faces(s,f)
    optional=horizontal_slivers(labels,rays)
    before=registered(prior,transform); after=registered(cells,transform)
    scores=placement_scores(after,before,dict.fromkeys(truth),occupied)
    strict=scores['relative_geometry']
    raw=placement_scores(before,before,dict.fromkeys(truth),occupied)['relative_geometry']
    new=suite.framed.spacing.shift(after,scores['translation'])
    old=suite.framed.spacing.shift(before,scores['baseline_translation'])
    measured={}
    if phase:
        # Reference envelopes for each old/new mask. Include a block face when
        # at least half its visible native pixels are inside that mask.
        areas=np.bincount(rays.ravel())
        for part in [p for p in base if any(p['id'] in r['inputs'] for r in selected)]+[p for p in changed if p['id'].endswith('-seam-union')]:
            mask=native_mask(part,image.size)
            hits=np.bincount(rays[mask],minlength=len(areas))
            sid_weights={}
            for face in labels['faces']:
                sid=face['surface_id']
                sid_weights[sid]=sid_weights.get(sid,0)+int(hits[face['id']])
            sid=max(sid_weights,key=sid_weights.get)
            faces=[f for f in labels['faces'] if f['surface_id']==sid and hits[f['id']] and 2*hits[f['id']]>=areas[f['id']]]
            if faces:
                coords=np.array([f['xyz_world'] for f in faces])
                axis=int(np.argmax(np.abs(faces[0]['normal_world'])))
                extents=np.ptp(coords,axis=0)+1
                measured[part['id']]=dict(visible_faces=len(faces),world_axis=axis,xyz_span=extents.tolist())
    return dict(id=key,split=s,baseline_pack_sha256=pack,baseline_reproduced=True,strict={k:strict[k] for k in placement.FIELDS},
                strict_prior={k:raw[k] for k in placement.FIELDS},facade=placement.priority(new,truth,optional),
                facade_prior=placement.priority(old,truth,optional),same_coordinates=cells==prior,
                input_dimensions={k:old_dims[k] for p in selected for k in p['inputs']},
                joined_dimensions={k:v for k,v in new_dims.items() if k.endswith('-seam-union')},
                majority_visible_span=measured)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-only',action='store_true')
    args=parser.parse_args()
    guarded=read(OUT/('phase-only.json' if args.phase_only else 'guarded.json'))
    changed={r['id']:r['pairs'] for r in guarded['rows'] if any(p['accepted'] for p in r['pairs'])}
    tasks=[(j,changed[j[1]['id']],args.phase_only) for j in suite.jobs([*suite.framed.SIZES,'shelter']) if j[1]['id'] in changed]
    assert len(tasks)==len(changed)
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(grade,tasks))
    save(OUT/('phase-placement.json' if args.phase_only else 'placement.json'),dict(baseline='a95816d',rows=rows))
    for row in rows:
        print(row['id'],row['strict_prior'],row['strict'],row['facade'])


if __name__=='__main__':
    main()
