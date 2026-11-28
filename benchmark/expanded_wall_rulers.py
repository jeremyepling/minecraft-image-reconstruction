"""Measure local pitch and phase on predicted wall charts in the 38 exact-mask test views."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import expanded_scores, floating_cloud_replay as paths, relative_face_round as assembly
from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save
from benchmark.full_block_lattice import atlas, verify_sprite
from benchmark.shared_corner_rows import sample, unpack
from engine.geometry import UNIT, project

NATIVE = WORKTREES / 'crop-ceiling-expanded/runs/native-crop-ceiling/native'
OUT = ROOT / 'runs/wall-rulers'
ARMS = ('noop', 'spacing', 'origin')
FRACTIONS = (np.arange(7) + .5) / 7
SAMPLES = np.stack(np.meshgrid(FRACTIONS, FRACTIONS), axis=-1)


def reference_chart(faces, blocks, camera):
    """Local horizontal-world-coordinate and height basis; validate against all visible face corners."""
    first = faces[0]
    normal = first['normal_world']
    if normal[1] or any(f['normal_world'] != normal or blocks[f['block']] != 'full_cube' for f in faces):
        return
    axis = 0 if normal[2] else 2
    origin = np.asarray(first['corners_world'][0])
    tangent = np.eye(3)[:, [axis, 1]]
    embed = np.column_stack((tangent, origin))
    H = camera @ np.vstack((embed, [0., 0., 1.]))
    world = np.asarray([f['corners_world'] for f in faces])
    uv = (world - origin) @ tangent
    assert np.allclose(world - origin, uv @ tangent.T)
    error = np.linalg.norm(project(H, uv) - [f['corners_px'] for f in faces], axis=-1).max()
    if error > .01:
        raise ValueError(f'Corner projection differs by {error} pixels')
    return H, float(error)


def jacobian(H, xy):
    point = H @ np.r_[xy, 1.]
    return (H[:2,:2] - np.outer(point[:2] / point[2], H[2,:2])) / point[2]


def local_rulers(chart, reference):
    """Preserve source vanishing directions. Change two derivative lengths at one common anchor."""
    H = np.asarray(chart['H'])
    uv = np.array([(c['u'], c['v']) for c in chart['cells']])
    middle = np.median(uv, axis=0)
    anchor = uv[np.argmin(np.linalg.norm(uv-middle, axis=1))]
    point = project(H, anchor)
    true_uv = project(np.linalg.inv(reference), point)
    old = jacobian(H, anchor)
    true = jacobian(reference, true_uv)
    predicted_pitch = np.linalg.norm(old, axis=0)
    true_pitch = np.linalg.norm(true, axis=0)
    cosine = np.abs((old / predicted_pitch).T @ (true / true_pitch))
    order = [0, 1] if np.trace(cosine) >= np.trace(cosine[:,::-1]) else [1, 0]
    exact_pitch = true_pitch[order]
    factor = exact_pitch / predicted_pitch
    relative = np.eye(3)
    relative[:2,:2] = np.diag(factor)
    relative[:2,2] = anchor * (1-factor)
    spacing = H @ relative
    np.testing.assert_allclose(project(spacing, anchor), point, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(jacobian(spacing,anchor),axis=0), exact_pitch, rtol=1e-7)
    snapped = project(reference, np.rint(true_uv))
    shift = np.eye(3)
    shift[:2,2] = project(np.linalg.inv(spacing), snapped) - anchor
    origin = spacing @ shift
    np.testing.assert_allclose(project(origin,anchor), snapped, atol=1e-6)
    return [H, spacing, origin], dict(anchor_uv=anchor.tolist(), pixel=point.tolist(),
        predicted_pitch=predicted_pitch.tolist(), exact_pitch=exact_pitch.tolist(),
        pitch_factors=factor.tolist(), phase_offset_blocks=shift[:2,2].tolist(),
        snapped_pixel=snapped.tolist(), axis_cosines=cosine[np.arange(2),order].tolist())


def resample(chart, H, mask, rgb, points):
    """Apply shared_corner_rows.chart_cells' occupancy and texture rules on the native mask.

    Enumerating observed pixels avoids walking a bounding rectangle across a vanishing line.
    No reference cell mask, extent, block coordinate or ID reaches this sampler.
    """
    uv = project(np.linalg.inv(H), points)
    bins = np.unique(np.floor(uv).astype(np.int64),axis=0)
    centers = project(H, bins+.5)
    bins = bins[sample(mask, centers, 1)]
    cells = []
    for start in range(0,len(bins),512):
        chunk = bins[start:start+512]
        quads = project(H, chunk[:,None,:]+UNIT)
        length = np.linalg.norm(quads-np.roll(quads,1,axis=1),axis=-1).min(1)
        support = sample(mask,project(H,chunk[:,None,None,:]+SAMPLES),1)
        coverage = support.mean((1,2))
        for i in np.flatnonzero((length>=3.5) & support[:,3,3] & (coverage>=.78)):
            quad = quads[i]
            hit = verify_sprite(rgb,quad[[3,2,1,0]])
            cells.append(dict(u=int(chunk[i,0]),v=int(chunk[i,1]),block_id=hit[0] if hit else 'smooth_stone',
                confidence=hit[2] if hit else 0.,fallback=hit is None,mask_coverage=float(coverage[i])))
    return dict(chart,H=H.tolist(),cells=cells,confidence=float(np.mean([c['confidence'] for c in cells])) if cells else 0.)


def cases():
    for dataset in ('visible-builds-v2', 'visible-builds-v3', 'decorative-builds-v1'):
        base = SOURCE/dataset
        for rec in read(base/'manifest.json')['frames']:
            if rec['split']=='test':
                frame = base/'data/test/frames'/rec['id']
                yield dict(rec,dataset=dataset,frame=str(frame),
                    variants={'sky':dict(image=str(frame/'rgb.png'),source_sha256=rec['rgb_sha256'])})


def run(case):
    cv2.setNumThreads(1)
    name = case['id']
    folder = Path(case['frame'])
    src = folder/'rgb.png'
    assert digest(src)==case['rgb_sha256']
    handoff = NATIVE/'sky'/name/'source-handoff'
    native_file = NATIVE/'atlas/connected/sky'/name/'charts.json'
    packet = read(native_file)
    paths.check_handoff(packet,handoff,case['rgb_sha256'])
    rgb = np.array(Image.open(src).convert('RGB'))
    labels = read(folder/'labels.json')
    meta = read(folder/'frame.json')
    for f in ('camera.json','labels.npz','labels.json'):
        assert digest(folder/f)==meta['files'][f]
    faces = defaultdict(list)
    for face in labels['faces']:
        faces[face['surface_id']].append(face)
    blocks={b['id']:b.get('geometry_kind','full_cube') for b in labels['blocks']}
    camera=np.asarray(labels['camera']['world_to_pixel_homogeneous'])
    parts={p['id']:p for p in read(handoff/'surfaces.json')['planes']}
    output = {arm:[] for arm in ARMS}
    evidence = []
    for chart in packet['charts']:
        sid = int(chart['id'].removeprefix('plane-'))
        truth = reference_chart(faces[sid],blocks,camera)
        if truth is None or not chart['cells']:
            for arm in ARMS:
                output[arm].append(chart)
            continue
        Href,error=truth
        mask=unpack(parts[chart['id']])
        assert tuple(mask.shape)==rgb.shape[:2]
        points=np.column_stack(np.nonzero(mask)[::-1])+.5
        warps,record=local_rulers(chart,Href)
        rows=dict(id=chart['id'],source_cells=len(chart['cells']),max_corner_error=error,**record)
        for arm,warp in zip(ARMS,warps):
            result=resample(chart,warp,mask,rgb,points)
            output[arm].append(result)
            rows[arm]=len(result['cells'])
        evidence.append(rows)
    # Reuse the v2 framing proof verified in the native-mask baseline replay.
    def checked_read(path):
        data=read(path)
        if case['dataset']=='visible-builds-v2' and path==folder/'frame.json':
            audit=read(ROOT/'training/framing-audit-v2.json')
            proof=next(r for r in audit['frames'] if r['id']==name)
            assert proof['accepted'] and proof['rgb_sha256']==case['rgb_sha256']
            return dict(data,framing=proof)
        return data
    for arm in ARMS:
        root=OUT/arm
        packet_path=root/'charts/sky'/name/'charts.json'
        save(packet_path,dict(packet,charts=output[arm],ruler_arm=arm,native_packet_sha256=digest(native_file)))
        mode=(root/'charts','charts.json',False,1,2.5,.5,False)
        with patch.object(paths,'RUN',NATIVE), patch.object(assembly,'ASSEMBLY',root/'assembly'), \
             patch.dict(assembly.MODES,{'connected-atlas':mode}), patch.object(expanded_scores,'read',checked_read):
            assembly.evaluate((case,'sky','connected-atlas'))
    save(OUT/'evidence'/(name+'.json'),dict(dataset=case['dataset'],id=name,charts=evidence,
        native_packet_sha256=digest(native_file),source_sha256=digest(src),atlas=atlas().provenance))
    print(name,len(evidence),'eligible charts',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--case',nargs='+')
    args=p.parse_args()
    jobs=[case for case in cases() if (not args.case or case['id'] in args.case)]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,jobs))
