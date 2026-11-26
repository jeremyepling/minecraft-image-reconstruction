"""Score the RGB-only local-plane fallback after decoding exported Minecraft packs."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark.camera_metrics import truth_camera
from benchmark.mask_blocks import _write_read
from benchmark.relative_geometry import best_translation
from benchmark.texture_camera_eval import DATA, ROOT, STABLE, checked, digest, frames, read, save
from engine.front_wall_grid import local_grid


def cardinal_frame(cells, pose):
    # Local +X follows screen right. Use pose only to resolve compass for scoring;
    # gravity, distances, and which cells are occupied come from the prediction.
    Q = truth_camera(pose)
    right = Q[0,:3] - pose['image_size'][0]*.5*Q[2,:3]
    axis = 0 if abs(right[0]) >= abs(right[2]) else 2
    sign = int(np.sign(right[axis]))
    return {(sign*x,y,0) if axis==0 else (0,y,sign*x) for x,y,_ in cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--atlas',type=Path,default=STABLE/'runs/26.3/atlas')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rows = []
    for f in frames(DATA,'regression'):
        key = f['id']
        inputs = read(STABLE/'runs/visible-depth-rounds/inputs'/f'{key}.json')
        if inputs['camera']['P'] is not None:
            continue
        src = DATA/'data'/'regression'/f['directory']
        with Image.open(checked(src/'rgb.png',f['files']['rgb.png'])) as rgb:
            cells, evidence = local_grid(rgb,atlas_path=args.atlas)
        dest = args.out/key
        dest.mkdir(exist_ok=True)
        pack = dest/'reconstruction.zip'
        cells = _write_read(pack,cells)
        pose = read(checked(src/'camera.json',f['files']['camera.json']))
        truth = {tuple(b['xyz_world']) for b in read(checked(src/'labels.json',f['files']['labels.json']))['blocks']}
        placed = cardinal_frame(cells,pose)
        shift,correct = best_translation(placed,truth)
        rows.append(dict(id=key,sha256=inputs['rgb_sha256'],pack_sha256=digest(pack),target=len(truth),
                         placed=len(placed),correct=correct,extras=len(placed)-correct,offset=shift,evidence=evidence))
    archive = ROOT/'benchmark/reviews/texture-camera-01/scoring.zip'
    with ZipFile(archive) as z:
        screen = json.loads(z.read('native-inputs/index.json'))
        cameras = {r['rgb_sha256']:r['candidate']['metric'] for r in json.loads(z.read('native70/fb9d5b4/cases.json'))}
    native = []
    for f in screen:
        if cameras[f['sha256']]:
            continue
        path = STABLE.parent/'workspace/builds'/f['fixture_directory']/'captures'/(f['id']+'.png')
        with Image.open(checked(path,f['sha256'])) as rgb:
            cells, evidence = local_grid(rgb,atlas_path=args.atlas)
        native.append(dict(id=f['id'],sha256=f['sha256'],placed=len(cells),evidence=evidence))
    report = dict(source_sha256=digest(ROOT/'engine/front_wall_grid.py'),rows=rows,native_abstentions=native)
    save(args.out/'manifest.json',report)
    print([(r['id'],r['correct'],r['extras']) for r in rows])
    print('native screen',len(native),sum(bool(r['placed']) for r in native))


if __name__=='__main__':
    main()

