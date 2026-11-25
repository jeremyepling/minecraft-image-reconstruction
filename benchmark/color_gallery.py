"""Score frozen reconstruction crop coordinates using known building voxels and live renders."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.fixtures import clear_sightline
from calibration.retrieval import features, rank
from engine.reconstruction import exemplar_bank
from calibration.projection import camera_project, south_quad
from engine.geometry import project, transform, face


def load(path):
    return json.loads(path.read_text())


def sample_labels(cells, pose, blocks):
    """Intersect 25 deterministic subcell camera rays with occupied cubes, keeping the nearest hit."""
    uv = np.stack(np.meshgrid((np.arange(5)+0.5)/5, (np.arange(5)+0.5)/5), axis=-1)
    xy = np.stack([project(transform(cell['quad']), uv) for cell in cells]).reshape(-1, 2)
    yaw, pitch = np.radians(pose['yaw_pitch_deg'])
    eye = np.array(pose['eye_world'])
    forward = np.array([-np.sin(yaw)*np.cos(pitch), -np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    right = np.array([-np.cos(yaw), 0, -np.sin(yaw)])
    width, height = pose['image_size']
    focal = height/(2*np.tan(np.radians(pose['vertical_fov_deg'])/2))
    rays = forward + (xy[:, :1]-width/2)/focal*right + (height/2-xy[:, 1:])/focal*np.cross(right, forward)
    best = np.full(len(rays), np.inf)
    labels = np.zeros(len(rays), int)
    names = ['background', *sorted({b['block_id'] for b in blocks})]
    inverse = 1/np.where(abs(rays)<1e-12, 1e-12, rays)
    for b in blocks:
        lo = np.array([b[k] for k in 'xyz'])
        a = (lo-eye)*inverse
        z = (lo+1-eye)*inverse
        near = np.minimum(a,z).max(axis=1)
        far = np.maximum(a,z).min(axis=1)
        hit = (near>0) & (near<=far) & (near<best)
        best[hit] = near[hit]
        labels[hit] = names.index(b['block_id'])
    hits = [Counter(np.asarray(names)[row]) for row in labels.reshape(-1,25)]
    return [h.most_common(1)[0][0] if h.most_common(1)[0][1]>=20 and h.most_common(1)[0][0]!='background'
            else None for h in hits], sum(h['background']>=13 for h in hits)


def colors(rgb, quads):
    return np.stack([face(rgb, q, 16)[2:-2,2:-2].mean(axis=(0,1)) for q in quads])


def live_error(run, cells, source, indices):
    pose = load(run/'captures/reconstruction.json')
    live = load(run/'layout.json')['blocks']
    offset = min(b['y'] for b in live) - min(c['xyz'][1] for c in cells)
    quads = []
    for i in indices:
        x,y,z = cells[i]['xyz']
        quads.append(camera_project(south_quad(x,y+offset,z), pose))
    target = colors(np.array(Image.open(run/'captures/reconstruction.png').convert('RGB')),quads)
    # Compare chromaticity separately by removing the mean channel brightness.
    delta = source[indices]-target
    return dict(rgb_mae=float(abs(delta).mean()), chromatic_mae=float(abs(delta-delta.mean(axis=1,keepdims=True)).mean()))


def aligned_controls(rgb, pose, blocks, banks, cells):
    """Compare matchers on projected unit south faces with nine clear sightlines."""
    occupied = {tuple(b[k] for k in 'xyz') for b in blocks}
    tiles = []
    truth = []
    estimated = np.array([c['quad'] for c in cells])
    centers = estimated.mean(axis=1)
    edges = np.linalg.norm(estimated-np.roll(estimated,1,axis=1),axis=2)
    size_ratios = []

    eye = pose['eye_world']
    for b in blocks:
        x,y,z = (b[k] for k in 'xyz')
        if eye[2]<=z+1 or (x,y,z+1) in occupied:
            continue
        q = camera_project(south_quad(x,y,z),pose)
        if np.any(q<0) or np.any(q>=pose['image_size']):
            continue
        if all(clear_sightline(eye,[x+u,y+v,z+1],occupied) for u in (.05,.5,.95) for v in (.05,.5,.95)):
            tiles.append(face(rgb,q,16))
            truth.append(b['side_texture'])
            delta = np.roll(q,-1,axis=0)-q
            rel = centers[:,None,:]-q
            cross = delta[:,0]*rel[:,:,1]-delta[:,1]*rel[:,:,0]
            inside = (cross>=0).all(axis=1)|(cross<=0).all(axis=1)
            true_edges = np.linalg.norm(q-np.roll(q,1,axis=0),axis=1)
            ratios = edges[inside]/true_edges
            size_ratios.extend(np.stack([ratios[:,[1,3]].mean(1),ratios[:,[0,2]].mean(1)],axis=1))
    scores = {}
    for name,(train,labels,method,exposure) in banks.items():
        scores[name] = sum(names[0]==actual for (names,_,_),actual in
                           zip(rank(np.stack(tiles),train,labels,method,exposure),truth))
    return dict(total=len(tiles),correct=scores, automatic_crops_with_centers_on_these_faces=len(size_ratios),
                median_crop_to_face_width_height_ratio=np.median(size_ratios,axis=0).tolist() if size_ratios else None)


def control_banks(calibration):
    atlas = calibration/'atlas'
    catalog = load(atlas/'placement-index.json')
    banks = {}
    for name,lib in [('atlas',None),('rendered',calibration/'face_textures/softened-library.npz')]:
        tiles,labels,method = exemplar_bank(atlas,lib,catalog)
        use = np.isin(labels,list(catalog['choices']['south']))
        train = features(tiles[use])
        banks[name] = train,labels[use],method,False
        if lib:
            banks['original'] = train,labels[use],method,True
    return banks


def score(fixture, original, trials, calibration, motif):
    banks = control_banks(calibration)
    blocks = load(fixture/'layout.json')['blocks']
    results = {}
    for ref in sorted(original.glob('test-*-front')):
        view = ref.name
        building = view.rsplit('-',1)[0]
        source_blocks = [b for b in blocks if b['building_id']==building]
        palette = sorted({b['block_id'] for b in source_blocks})
        src = fixture/'captures'
        pose = load(src/(view+'.json'))
        cells = load(ref/'final-build/geometry.json')['cells']
        truth, background = sample_labels(cells,pose,source_blocks)
        indices = [i for i,t in enumerate(truth) if t]
        rgb = np.array(Image.open(src/(view+'.png')).convert('RGB'))
        controls = aligned_controls(rgb,pose,source_blocks,banks,cells)
        source = colors(rgb, [c['quad'] for c in cells])
        metrics = {}
        configs = [('original',ref/'final-build',ref/'final-verify')]
        updated = trials/view
        configs += [(p.name,p,updated/'verify' if p.name=='build' else None) for p in sorted(updated.iterdir())
                    if (p/'placements.json').is_file()]
        if motif:
            configs.append(('motif',motif/view/'build',motif/view/'verify'))
        for name, build, check in configs:
            placed = {tuple(b[k] for k in 'xyz'): b['block_id'] for b in load(build/'placements.json')['blocks']}
            predicted = [placed[tuple(cells[i]['xyz'])] for i in indices]
            count = Counter(predicted)
            metrics[name] = dict(exact=sum(p==truth[i] for p,i in zip(predicted,indices)),
                                 in_building_palette=sum(count[k] for k in palette),
                                 different_ids=len(count), most_frequent=count.most_common(8))
            if check and (check/'verification.json').exists():
                metrics[name]['live_color'] = live_error(check,cells,source,indices)
        results[view] = dict(crops=len(cells),majority_background=background,clean_material_crops=len(indices),
                             palette=palette, methods=metrics, aligned_south_faces=controls,
                             evaluation='Diagnostic reuse of four previously seen gallery views; each crop is an observation, and multiple crops may map to one export coordinate.')
    return results


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['fixture','original','trials','out','calibration']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--motif', type=Path)
    args=parser.parse_args()
    rows=score(args.fixture,args.original,args.trials,args.calibration,args.motif)
    args.out.write_text(json.dumps(rows,indent=2)+'\n')
