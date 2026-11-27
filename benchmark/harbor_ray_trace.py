"""Trace one decoded wrong cube back through its source ray and the selected wall carrier."""

from pathlib import Path
import numpy as np
from benchmark.harbor_carrier_audit import BASE, CASES, REVIEW, ROOT, SOURCE, build_dir, decoded_pack, frames, read, save
from benchmark.harbor_plane_replay import sampled_ids
from benchmark.typed_structures import aligned_sites
from engine.visible_placement import _frontmost_cells
from engine.visible_raster import observed_support


def trace(entry, case):
    key = entry['id']
    source = Path(entry['frame'])
    original = build_dir(entry, case)
    prior = decoded_pack(original / 'reconstruction.zip')
    split = decoded_pack(ROOT / 'runs/harbor-placement-evidence-01' / key / 'split/reconstruction.zip')
    split_status = read(REVIEW / f'{key}-split-score.json')
    labels = read(source/'labels.json')
    ids = sampled_ids(labels, 0)
    faces = {f['id']: f for f in labels['faces']}
    truth = {tuple(b['xyz_world']): b['block_id'] for b in labels['blocks']}
    parts = read(BASE/entry['dataset']/key/'handoffs/placement-input.json.gz')['parts']
    tag = read(REVIEW/(key+'.json'))['carriers'][0]['id']
    part = next(p for p in parts if p['id']==tag)
    mask = observed_support(part['support'], ids.shape, labels['camera']['image_size'])
    status = read(original/'build-manifest.json')['placement']
    decision = next(r for r in status['plane_offsets'] if r['id']==tag)
    P = np.asarray(status['P'])
    T = np.asarray(case['registration']['transform'])
    eye = np.linalg.solve(P[:,:3],-P[:,3])
    def render(blocks):
        cells = list(blocks)
        owner, axes = _frontmost_cells(cells,P,eye,ids.shape,face_axes=True)
        return cells, owner, axes
    original_cells, first, axes = render(prior)
    split_cells, next_hit, split_axes = render(split)
    axis, level = decision['axis'], decision['level']
    reference_face_level = level-int(eye[axis]>level)
    candidates=[]
    for y,x in zip(*np.nonzero(mask & (ids>0) & (first>=0) & (next_hit>=0) & (axes==axis) & (split_axes==axis))):
        before, after = original_cells[first[y,x]], split_cells[next_hit[y,x]]
        if before in split or after in prior or before[axis]!=reference_face_level:
            continue
        visible=faces[int(ids[y,x])]
        raw_world=aligned_sites([before,after], T, (0,0,0))
        if raw_world[0] in truth or raw_world[1] != tuple(visible['xyz_world']):
            continue
        before_score = aligned_sites([before], T, case['translation'])[0]
        after_score = aligned_sites([after], T, split_status['translation'])[0]
        candidates.append(dict(xy=[int(x)*4+2,int(y)*4+2], source_face_id=visible['id'],
            visible_id=truth[raw_world[1]], visible_world=raw_world[1],axis=axis,
            carrier=tag, selected_level=level, before_local=before, after_local=after,
            before_export_id=prior[before],after_export_id=split[after],
            before_world_registered=raw_world[0], after_world_registered=raw_world[1],
            before_world_score=before_score, after_world_score=after_score,
            before_counted_correct=before_score in truth, after_counted_correct=after_score in truth))
    if candidates:
        preferred=[r for r in candidates if not r['before_counted_correct'] and r['after_counted_correct']]
        example=(preferred or candidates)[0]
        save(REVIEW/(key+'-ray-trace.json'),example)
        print(example)


if __name__=='__main__':
    cases={r['id']:r for r in read(CASES)}
    for frame in frames(SOURCE,['visible-builds-v3']):
        if frame['id']=='validation-harbor-02-kiln-high-right':
            trace(frame,cases[frame['id']])
