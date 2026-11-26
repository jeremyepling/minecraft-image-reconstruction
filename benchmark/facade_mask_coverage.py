"""Trace native wall pixels of missed blocks to the existing Surface Finder outputs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import framed_depth_eval as framed
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked, read, save
from engine.surface_perimeter import _native_alpha


def coverage(job):
    split, frame, previous, scored, run = job
    image, parts, _, _, _, labels, ids, _, _, placed, _ = read_view(run,split,frame,previous,scored)
    faces, blocks = labels['faces'],labels['blocks']
    # Reserve zero for background and for all horizontal faces.
    block_lut = np.zeros(len(faces)+1, int)
    axis_lut = np.zeros_like(block_lut)
    for face in faces:
        normal = face['normal_world']
        if normal[1] == 0:
            block_lut[face['id']] = face['block']
            axis_lut[face['id']] = int(np.argmax(np.abs(normal)))
    counts = np.bincount(block_lut[ids.ravel()],minlength=len(blocks)+1)
    wanted = np.zeros(len(blocks)+1,bool)
    for b in blocks:
        i = b['id']
        wanted[i] = counts[i]>=64 and tuple(b['xyz_world']) not in placed
    yy,xx = np.nonzero(wanted[block_lut[ids]])
    face_id = ids[yy,xx]
    owner = block_lut[face_id]
    axis = axis_lut[face_id]
    # The recorded orientation is used only to grade a predicted normal after inference.
    rotation = np.asarray(labels['camera']['world_to_camera_rotation'])
    unions = {k:np.zeros(len(xx),bool) for k in ('any_surface','any_vertical','matching_axis')}
    best = {k:np.zeros(len(blocks)+1,int) for k in ('single_any','single_matching_axis','single_within_20deg')}
    for part in parts:
        if not part.get('support'):
            continue
        native = _native_alpha(part['support'],image.size)
        if native is None:
            continue
        alpha,(l,t,r,b) = native
        indices = np.flatnonzero((xx>=l)&(xx<r)&(yy>=t)&(yy<b))
        covered = indices[np.asarray(alpha)[yy[indices]-t,xx[indices]-l] != 0]
        unions['any_surface'][covered] = True
        hits = np.bincount(owner[covered],minlength=len(blocks)+1)
        np.maximum(best['single_any'],hits,out=best['single_any'])
        if part.get('normal_camera') is None:
            continue
        n = rotation.T @ np.asarray(part['normal_camera'])
        length = np.linalg.norm(n)
        if not length or not np.isfinite(length):
            continue
        a = int(np.argmax(np.abs(n)))
        if a == 1:
            continue
        unions['any_vertical'][covered] = True
        matched = covered[axis[covered] == a]
        unions['matching_axis'][matched] = True
        hits = np.bincount(owner[matched],minlength=len(blocks)+1)
        np.maximum(best['single_matching_axis'],hits,out=best['single_matching_axis'])
        if abs(n[a])/length >= np.cos(np.deg2rad(20)):
            np.maximum(best['single_within_20deg'],hits,out=best['single_within_20deg'])
    measures = {k:np.bincount(owner[mask],minlength=len(blocks)+1) for k,mask in unions.items()}
    measures.update(best)
    return dict(id=frame['id'],split=split,rgb_sha256=scored['rgb_sha256'],pack_sha256=scored['pack_sha256'],
                support_hashes=scored['input_hashes'],eligible=int(wanted.sum()),
                half_covered={k:int((2*v[wanted]>=counts[wanted]).sum()) for k,v in measures.items()},
                three_quarters_covered={k:int((4*v[wanted]>=3*counts[wanted]).sum()) for k,v in measures.items()})


def main(args):
    meta=read(args.run/'summary.json')
    rows=read(checked(args.run/'cases.json',meta['cases_sha256']))
    index={(r['split'],r['id']):r for r in rows}
    jobs=[(s,f,p,index[s,f['id']],args.run) for s,f,p in framed.cases(framed.EVALUATION)]
    assert len(jobs)==104
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        result=list(pool.map(coverage,jobs))
    totals=dict(eligible=sum(r['eligible'] for r in result))
    for column in ('half_covered','three_quarters_covered'):
        totals[column]=dict(sum((Counter(r[column]) for r in result),Counter()))
    proof=read(framed.REVIEW/'roof-height-01/verification.json')
    if proof['cases_sha256']==meta['cases_sha256']:
        assert totals['eligible']==1466
    elif args.out.resolve()==(framed.REVIEW/'facade-visibility-01/wall-masks.json').resolve():
        raise ValueError('Supply a separate candidate output file.')
    save(args.out,dict(cases_sha256=meta['cases_sha256'],views=len(result),totals=totals,frames=result))
    print(totals)
    print(next(r for r in result if r['id']=='legacy-light_gray_concrete-dev-shopfront-plaster-low-left'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('../roof-height-round/runs/production'))
    p.add_argument('--out',type=Path,default=Path('benchmark/reviews/facade-visibility-01/wall-masks.json'))
    p.add_argument('--workers',type=int,default=4)
    main(p.parse_args())
