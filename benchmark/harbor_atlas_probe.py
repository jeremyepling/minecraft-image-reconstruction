"""Measure calibrated atlas evidence for neighboring levels of one actual wall carrier."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from benchmark.harbor_carrier_audit import CASES, REVIEW, SOURCE, build_dir, frames, native_face_planes, read, save
from benchmark.harbor_plane_replay import sampled_ids
from engine.atlas_plane_depth import _coefficients, _patch_scores
from engine.camera_atlas_floor import _atlas
from engine.secondary_plane_depth import coordinates, supported_face
from engine.visible_raster import SAMPLE_STRIDE, observed_support


def audit(entry, case, tag):
    folder = build_dir(entry, case)
    frame = Path(entry['frame'])
    image = Image.open(frame/'rgb.png').convert('RGB')
    rgb = np.asarray(image)
    labels = read(frame/'labels.json')
    ids = sampled_ids(labels, 0)
    planes = native_face_planes(labels, np.asarray(case['registration']['transform']))
    parts = read(folder/'surfaces.json')['planes']
    part = next(p for p in parts if p['id'] == tag)
    status = read(folder/'build-manifest.json')['placement']
    chosen = next(d for d in status['plane_offsets'] if d['id']==tag)
    axis, primary = chosen['axis'], chosen['level']
    P = np.asarray(status['P'])
    P = P/np.linalg.norm(P[2,:3])
    eye = np.linalg.solve(P[:,:3], -P[:,3])
    mask = observed_support(part['support'], ids.shape, image.size)
    y,x = np.nonzero(mask)
    rays = np.stack([(x+.5)*SAMPLE_STRIDE,(y+.5)*SAMPLE_STRIDE,np.ones(len(x))],axis=1) @ np.linalg.inv(P[:,:3]).T
    free = [a for a in range(3) if a != axis]
    coeff = _coefficients(rays,axis,free)
    atlas = _atlas('26.3')
    levels = {}
    reference = np.array([(-1,0)] + [planes[i] for i in range(1,len(planes)+1)])[ids[mask]]
    for level in [primary]+[primary + i for i in range(-6,7) if i]:
        uv = eye[free]+coeff*(level-eye[axis])
        keys,inv,n = np.unique(np.floor(uv).astype(int),axis=0,return_inverse=True,return_counts=True)
        inv = inv.ravel()
        scores = dict(_patch_scores(mask, keys, axis, free, level, P,rgb,atlas,by_cell=True))
        q = np.array([scores.get(tuple(k), 0.) for k in keys])
        if level == primary:
            main = q[inv]
        good = {}
        for idx in np.flatnonzero((q>=.70)&(n>=8)):
            if level==primary or np.mean(main[inv==idx]+.20 < q[idx])>=.8:
                cell=coordinates(keys[idx:idx+1],axis,level,eye)[0]
                if supported_face(cell,axis,level,eye,P,mask):
                    good[tuple(keys[idx])] = idx
        qualifying = [i for (a,b),i in good.items() if any(k in good for k in ((a+1,b),(a-1,b),(a,b+1),(a,b-1)))]
        pixel = np.isin(inv,qualifying)
        truth = (reference[:,0]==axis) & (reference[:,1]==level)
        levels[level]=dict(scored=len(scores),above_65=int(np.sum(q>.65)),above_70=int(np.sum(q>=.70)),
            max_q=float(q.max()), qualifying=len(qualifying), qualifying_samples=int(pixel.sum()),
            correct_qualifying_samples=int((pixel&truth).sum()), truth_samples=int(truth.sum()),
            qualifying_cells_correct_plane_majority=sum(int(np.mean(truth[inv==idx])>.5) for idx in qualifying))
    assert len(levels)==13
    return dict(id=entry['id'], tag=tag, primary=primary, axis=axis, radius=6,
                primary_gate_would_pass=levels[primary]['above_65']>=2, levels=levels)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('ids',nargs='+')
    args=p.parse_args()
    cases={r['id']:r for r in read(CASES)}
    for e in frames(SOURCE,['visible-builds-v3']):
        if e['id'] in args.ids:
            tag=read(REVIEW/(e['id']+'.json'))['carriers'][0]['id']
            result=audit(e,cases[e['id']],tag)
            save(REVIEW/(e['id']+'-atlas-probe.json'),result)
            print(e['id'],result['primary_gate_would_pass'],result['primary'],
                  [(level,r['above_65'],r['qualifying'],r['qualifying_cells_correct_plane_majority'])
                   for level,r in result['levels'].items()],flush=True)


if __name__=='__main__':
    main()
