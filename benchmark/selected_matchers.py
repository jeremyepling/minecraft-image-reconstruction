"""Replay material selection on one eligible visible face per predicted block group."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
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
from engine.surface_tiles import tile_sample

METHODS = ('atlas', 'exposure', 'texture', 'shadow')
MATCHERS = {}


def prepare(calibration):
    catalog = calibration/'atlas'
    library = calibration/'face_textures/softened-library.npz'
    MATCHERS.update(atlas=load_atlas(catalog), exposure=ExposureMatcher(catalog, library),
                    texture=ShapeMatcher(catalog))


def rank_selected(tile, rgb, field):
    patch = tile['rgb']
    gain, guidance = cell_shadow(field, tile['quad'], len(patch))
    ranks = dict(atlas=rank_tile(tile,MATCHERS['atlas']),
                 exposure=MATCHERS['exposure'].rank_patch(face(rgb,tile['quad'],16)),
                 texture=MATCHERS['texture'].rank_patch(patch),
                 shadow=MATCHERS['texture'].rank_patch(corrected(patch,gain)))
    return {m:r['block_id'].removeprefix('minecraft:') for m,r in ranks.items()}, guidance


def predict(frame):
    image_path, mask_path = Path(frame['image']), Path(frame['masks'])
    if digest(image_path)!=frame['sha256'] or digest(mask_path)!=frame['mask_sha256']:
        raise ValueError(f"Frozen input changed for {frame['key']}")
    masks = json.loads(gzip.decompress(mask_path.read_bytes()))[frame['mask_variant']]
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    rgb = np.asarray(image)
    observations = frame['faces']
    groups = group_faces(observations,image.size)
    rows = [dict(id=f['id'],quad=f['quad'],effective_width_px=effective_width(f['quad'])) for f in observations]
    output_groups = []
    calls = 0
    field = None
    for index, group in enumerate(groups):
        selected = group['selected_index']
        members = group['face_indices']
        material = dict.fromkeys(METHODS)
        guidance = None
        if selected is not None:
            if selected not in members or rows[selected]['effective_width_px'] < MIN_FACE_WIDTH:
                raise ValueError('Ineligible face selected')
            if field is None:
                field = surface_shadow_map(rgb,masks)
            tile = tile_sample(image,observations[selected]['quad'])
            material, guidance = rank_selected(tile,rgb,field)
            calls += 1
        chosen = rows[selected]['id'] if selected is not None else None
        for i in members:
            if 'group_id' in rows[i]:
                raise ValueError('Face assigned twice')
            rows[i].update(group_id=index,selected_id=chosen)
        output_groups.append(dict(id=index,face_ids=[rows[i]['id'] for i in members],
                                  selected_id=chosen,material=material,shadow_guidance=guidance,
                                  evidence=group['evidence'],ambiguous_neighbors=group['ambiguous_neighbors']))
    if any('group_id' not in row for row in rows):
        raise ValueError('Ungrouped observation')
    return dict(key=frame['key'],kind=frame['kind'],sha256=frame['sha256'],mask_sha256=frame['mask_sha256'],
                mask_variant=frame['mask_variant'],matcher_calls_per_method=calls,faces=rows,groups=output_groups)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--faces',type=Path,nargs='+',required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    frames=[f for path in args.faces for f in json.loads(path.read_bytes())['frames']]
    if args.workers == 1:
        prepare(args.calibration)
        result=list(map(predict,frames))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=prepare,initargs=(args.calibration,)) as pool:
            result=list(pool.map(predict,frames))
    root=Path(__file__).resolve().parents[1]
    modules=['engine/atlas.py','engine/atlas_count.py','engine/block_exposure.py','engine/blocker_shape.py',
             'engine/blocker_shadow.py','engine/block_face_groups.py','engine/surface_tiles.py','engine/surface_model.py',
             'engine/geometry.py','engine/face_evidence.py','calibration/retrieval.py','engine/reconstruction.py']
    source={name:digest(root/name) for name in modules}
    source['benchmark/selected_matchers.py']=digest(Path(__file__))
    artifacts=['atlas/atlas.png','atlas/atlas.json','atlas/placement-index.json','face_textures/softened-library.npz']
    source.update({name:digest(args.calibration/name) for name in artifacts})
    source.update({path.name:digest(path) for path in args.faces})
    payload=dict(methods=METHODS,minimum_width=MIN_FACE_WIDTH,sha256=source,frames=result)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0))
    print(f"{len(result)} frames including diagnostic controls, {sum(f['matcher_calls_per_method'] for f in result)} calls per matcher")

if __name__=='__main__':
    main()
