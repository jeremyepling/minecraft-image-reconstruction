"""Compare four material matchers before and after same-block face selection."""
import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from engine.atlas import digest
from engine.atlas_count import load_atlas, rank_tile
from engine.block_exposure import ExposureMatcher
from engine.block_face_groups import group_faces
from engine.blocker_shadow import cell_shadow, corrected, surface_shadow_map
from engine.blocker_shape import ShapeMatcher
from engine.face_evidence import MIN_FACE_WIDTH, effective_width
from engine.geometry import face
from engine.surface_model import canonical

METHODS = ('atlas', 'exposure', 'texture', 'shadow')
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'benchmark/reviews/crop-split-01'
SOURCES = ROOT / 'benchmark/reviews/mask-four-01'
STATE = {}


def load_matchers(calibration):
    atlas_dir = calibration/'atlas'
    library = calibration/'face_textures/softened-library.npz'
    STATE.update(atlas=load_atlas(atlas_dir), exposure=ExposureMatcher(atlas_dir, library),
                 texture=ShapeMatcher(atlas_dir),
                 cases={row['key']: row for row in json.loads((FIXTURE/'cases.json').read_bytes())},
                 sources={r['key']:SOURCES/r['source'] for r in json.loads((SOURCES/'cases.json').read_bytes())},
                 hashes=json.loads((FIXTURE/'manifest.json').read_bytes())['artifacts'])


def block_id(result):
    return result['block_id'].removeprefix('minecraft:') if result.get('block_id') else None


def score_materials(rgb, observations, field):
    predictions = []
    exposure_indices = []
    exposure_crops = []
    for idx, obs in enumerate(observations):
        q = canonical(obs['quad'])
        votes = dict.fromkeys(METHODS)
        predictions.append(votes)
        if effective_width(q) < MIN_FACE_WIDTH:
            continue
        patch = face(rgb,q,32)
        votes['atlas'] = block_id(rank_tile({'rgb':patch}, STATE['atlas']))
        votes['texture'] = block_id(STATE['texture'].rank_patch(patch))
        gain, _ = cell_shadow(field,q,32)
        lit = corrected(patch,gain)
        votes['shadow'] = votes['texture'] if np.array_equal(lit,patch) else block_id(STATE['texture'].rank_patch(lit))
        exposure_indices.append(idx)
        exposure_crops.append(face(rgb,q,16))
    if exposure_crops:
        for idx, result in zip(exposure_indices,STATE['exposure'].rank_patches(exposure_crops),strict=True):
            predictions[idx]['exposure'] = block_id(result)
    return predictions


def frame_predictions(frame):
    case = STATE['cases'][frame['key']]
    img_path = STATE['sources'][frame['key']]
    mask_path = FIXTURE/case['predictions']
    if not (digest(img_path) == case['sha256'] == frame['sha256'] and
            digest(mask_path) == STATE['hashes'][case['predictions']] == frame['mask_sha256']):
        raise ValueError(f"Frame or predicted masks changed for {frame['key']}")
    with Image.open(img_path) as im:
        rgb = np.asarray(im.convert('RGB'))
    masks = json.loads(gzip.decompress(mask_path.read_bytes()))['split']
    normals = {m['id']:m.get('normal_camera') for m in masks}
    obs = []
    for record in frame['faces']:
        surface = record.get('surface_id',record.get('proposal'))
        obs.append(dict(id=record['id'],quad=record['quad'],surface_id=surface,
                        normal_camera=record.get('normal_camera',normals.get(surface))))
    if len({r['id'] for r in obs}) != len(obs):
        raise ValueError('Duplicate face observation')
    # Group using only image geometry before touching any material ranking.
    groups = group_faces(obs,(rgb.shape[1],rgb.shape[0]))
    field = surface_shadow_map(rgb,masks)
    ranks = score_materials(rgb,obs,field)
    rows = [dict(id=item['id'],quad=item['quad'],effective_width_px=effective_width(item['quad']),
                 independent=ranks[i]) for i,item in enumerate(obs)]
    groups_out = []
    covered = []
    for index,group in enumerate(groups):
        selected = group['selected_index']
        indices = group['face_indices']
        covered.extend(indices)
        if selected is not None and (selected not in indices or rows[selected]['effective_width_px'] < MIN_FACE_WIDTH):
            raise ValueError('Selected an ineligible source face')
        chosen = obs[selected]['id'] if selected is not None else None
        for i in indices:
            rows[i].update(grouped=ranks[selected] if selected is not None else dict.fromkeys(METHODS),
                           group_id=index,selected_id=chosen)
        groups_out.append(dict(id=index,face_ids=[obs[i]['id'] for i in indices],selected_id=chosen,
                               evidence=group.get('evidence',[]),ambiguous_neighbors=group.get('ambiguous_neighbors',False)))
    if sorted(covered)!=list(range(len(obs))):
        raise ValueError('Grouping did not partition the observations')
    return dict(key=frame['key'],kind=frame['kind'],sha256=case['sha256'],mask_sha256=digest(mask_path),
                faces=rows,groups=groups_out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--faces',nargs='+',required=True,type=Path)
    p.add_argument('--calibration',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workers',type=int,default=1)
    args=p.parse_args()
    frames=[f for file in args.faces for f in json.loads(file.read_bytes())['frames']]
    if args.workers==1:
        load_matchers(args.calibration)
        result=list(map(frame_predictions,frames))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=load_matchers,initargs=(args.calibration,)) as pool:
            result=list(pool.map(frame_predictions,frames))
    provenance={str(f.name):digest(f) for f in args.faces}
    sources=['atlas_count','face_evidence','blocker_shape','blocker_shadow','block_face_groups','block_exposure']
    files=[ROOT/'engine'/f'{s}.py' for s in sources]+[Path(__file__),args.calibration/'face_textures/softened-library.npz',
                                                       args.calibration/'atlas/atlas.png',args.calibration/'atlas/placement-index.json']
    provenance.update({f.name:digest(f) for f in files})
    report=dict(methods=METHODS,minimum_width=MIN_FACE_WIDTH,sha256=provenance,frames=result)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(report,separators=(',',':')).encode('utf-8')
    args.out.write_bytes(gzip.compress(payload,mtime=0) if args.out.suffix=='.gz' else payload)
    print(f'{len(result)} frame passes, {sum(len(f["faces"]) for f in result)} observations')

if __name__=='__main__':
    main()

