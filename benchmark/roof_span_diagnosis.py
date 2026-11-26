"""Locate missing canopy blocks in the original photos using scoring-only labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmark.camera_metrics import project_visible, registration
from benchmark.framed_depth_eval import cases, photo, roof_specs, transfer
from benchmark.relative_canopy_spacing import shift
from benchmark.texture_camera_eval import ROOT, checked, read, save
from engine.camera_grid import plane_coordinates
from engine.surface_model import intrinsic
from engine.visible_horizontal_faces import _surface_masks
from training.visible_labels import visible_labels


def inspect(f, prior, split, old, rgb, parts, camera):
    path = transfer.rgb_path(f)
    pose = read(checked(path.with_suffix('.json'), f['files']['camera.json']))
    raw = read(checked(path.with_name(f['source_id']+'-blocks.json'), f['files']['world-readback.json']))
    states = {tuple(b[k] for k in 'xyz'): b for b in raw}
    labels, ids, _ = visible_labels(pose, states, (0,0,0))
    roof_y = roof_specs()[f['id']]['roof_y']
    expected = {tuple(b['xyz_world']) for b in labels['blocks'] if b['xyz_world'][1] == roof_y}
    offset = prior['roof']['spacing']['offset']
    recovered = expected.intersection(shift(old,offset)) if offset is not None else set()
    assert len(recovered) == prior['roof']['spacing']['anchored_roof_cells']
    P = np.asarray(camera['P'],dtype=float)
    P = P/np.linalg.norm(P[2,:3])
    T = registration(P,pose)
    focal = np.linalg.norm(P[0,:3]-.5*rgb.width*P[2,:3])
    rotation = np.linalg.solve(intrinsic(rgb.size,focal),P[:,:3])
    mask, oriented = _surface_masks(parts,rgb.size,np.arange(rgb.width)+.5,np.arange(rgb.height)+.5,rotation)
    horizontal = np.bincount(ids[oriented[1]].ravel(),minlength=len(labels['faces'])+1)
    silhouette = np.bincount(ids[mask].ravel(),minlength=len(labels['faces'])+1)
    by_block = {}
    error = {}
    for face in labels['faces']:
        xyz = tuple(face['xyz_world'])
        if xyz not in expected:
            continue
        item = by_block.setdefault(xyz, Counter())
        i = face['id']
        item['any'] += int(silhouette[i])
        if face['normal_world'][1]:
            item['horizontal'] += face['visible_pixels']
            item['normal_match'] += int(horizontal[i])
        for world, pixel, visible in zip(face['corners_world'],face['corners_px'],face['corner_visible']):
            if visible:
                here = project_visible(P @ np.linalg.inv(T),[world])[0]
                error[tuple(world)] = np.linalg.norm(here-pixel)
    miss = expected-recovered
    assert set(by_block)==expected
    def count(which,threshold):
        return sum(by_block[q][which]>=threshold for q in miss)
    # Supply the measured height and true top/underside labels only to this diagnostic.
    # Test each observed pixel against its own labeled cell, never a neighboring face.
    matched = {}
    for dy in (-1,0,1):
        hits = Counter()
        for sign in (-1,1):
            lookup = {face['id']: tuple(face['xyz_world']) for face in labels['faces']
                      if face['normal_world'][1]==sign and tuple(face['xyz_world']) in miss}
            if not lookup:
                continue
            y,x = np.nonzero(oriented[1] & np.isin(ids,list(lookup)))
            if not len(y):
                continue
            xy = np.column_stack((x+.5,y+.5))
            with np.errstate(divide='ignore',invalid='ignore'):
                xz = np.floor(plane_coordinates(P @ np.linalg.inv(T),xy,1,roof_y+int(sign>0)+dy))
            faces = ids[y,x]
            for face_id,cell in lookup.items():
                hits[cell] += np.count_nonzero((faces==face_id) & (xz[:,0]==cell[0]) & (xz[:,1]==cell[2]))
        matched[str(dy)] = int(sum(n>=4 for n in hits.values()))
    return dict(id=f['id'],split=split,rgb_sha256=f['files']['rgb.png'],roof=len(expected),
                anchored=len(recovered),missed=len(miss),miss_with_4_horizontal_pixels=count('horizontal',4),
                miss_with_4_horizontal_pixels_in_horizontal_mask=count('normal_match',4),
                miss_with_4_pixels_in_any_mask=count('any',4),measured_roof_corners=len(error),
                camera_p90_corner_error_px=float(np.percentile(list(error.values()),90)) if error else None,
                misses_with_4_correct_xz_witnesses_by_assumed_height_offset=matched)


def run(out):
    review = ROOT/'benchmark/reviews/framed-depth-01'
    prior = {(r['split'],r['id']):r for r in read(checked(review/'cases.json',read(review/'summary.json')['cases_sha256']))}
    rows = []
    with ZipFile(checked(review/'scores.zip',read(review/'shape-check.json')['archive_sha256'])) as archive:
        for split,f,h in cases(('old','fresh','extra')):
            if f['id'] not in roof_specs():
                continue
            r = prior[split,f['id']]
            old = set(map(tuple,json.loads(archive.read(f"{split}/{f['id']}/cells.json"))['world']))
            rgb,parts,_,camera = photo(split,f,h)
            rows.append(inspect(f,r,split,old,rgb,parts,camera))
    assert len(rows)==18
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,dict(method='Scoring-only mask and registered-camera measurements for previously missed post-anchored roof blocks',rows=rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args().out)
