"""Compare automatic masks and their axis assignments with recorded canopy faces."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark.camera_metrics import registration
from benchmark.canopy_fixture_metrics import PROBES
from benchmark.texture_camera_eval import checked, read, save
from benchmark.visible_depth_transfer import rgb_path
from engine.surface_model import intrinsic
from engine.visible_raster import SAMPLE_STRIDE, observed_support
from training.visible_labels import visible_labels


def coverage(frame, upstream, spec):
    key = frame['id']
    rgb = rgb_path(frame)
    pose = read(checked(rgb.with_suffix('.json'),frame['files']['camera.json']))
    raw = read(checked(rgb.with_name(frame['source_id']+'-blocks.json'),frame['files']['world-readback.json']))
    states = {tuple(b[k] for k in 'xyz'):b for b in raw if b['y']>64}
    labels, native_ids, _ = visible_labels(pose,states,(0,0,0))
    # Take the native pixel containing each center used by the production four-pixel grid.
    ids = native_ids[SAMPLE_STRIDE//2::SAMPLE_STRIDE, SAMPLE_STRIDE//2::SAMPLE_STRIDE]
    size = pose['image_size']
    P = np.asarray(frame['camera']['P'])
    P = P/np.linalg.norm(P[2,:3])
    rotation = np.linalg.solve(intrinsic(size,np.linalg.norm(P[0,:3]-.5*size[0]*P[2,:3])),P[:,:3])
    world_rotation = registration(P,pose)[:3,:3]
    masks = [np.zeros(ids.shape, bool) for _ in range(3)]
    all_masks = np.zeros(ids.shape,bool)
    for part in read(upstream/key/'surfaces.json')['planes']:
        if not part.get('support'):
            continue
        mask = observed_support(part['support'],ids.shape,size)
        all_masks |= mask
        if part.get('normal_camera') is not None:
            axis = np.argmax(abs(world_rotation @ rotation.T @ part['normal_camera']))
            masks[axis] |= mask
    counts = Counter()
    row_limits = {}
    for y, row in enumerate(native_ids):
        for id_ in np.unique(row):
            if id_ in row_limits:
                row_limits[id_][1] = y
            else:
                row_limits[id_] = [y,y]
    for face in labels['faces']:
        p = face['xyz_world']
        if p[1] != spec['roof_y']:
            continue
        category = {1:'top',0:'rim',-1:'underside'}[face['normal_world'][1]]
        reference = ids == face['id']
        n = int(reference.sum())
        axis = np.argmax(abs(np.asarray(face['normal_world'])))
        any_mask = int(np.count_nonzero(reference & all_masks))
        oriented = int(np.count_nonzero(reference & masks[axis]))
        low, high = row_limits[face['id']]
        counts.update({category+'_faces':1, category+'_sampled_faces':int(n>0),
                       category+'_one_row_faces':int(high==low),
                       category+'_at_most_two_row_faces':int(high-low<2),
                       category+'_samples':n,category+'_any_mask_samples':any_mask,
                       category+'_same_axis_samples':oriented,
                       category+'_majority_any_faces':int(n>=4 and 2*any_mask>=n),
                       category+'_majority_axis_faces':int(n>=4 and 2*oriented>=n)})
    return dict(id=key,counts=dict(counts))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    selected = read(args.inputs/'screen.json')['selected']
    specs = read(PROBES)
    rows = [coverage(f,args.inputs/'upstream',specs[f['id']]) for f in selected if f['id'] in specs]
    counts = sum((Counter(r['counts']) for r in rows),Counter())
    save(args.out,dict(summary=dict(counts),cases=rows))
    print(dict(counts))
