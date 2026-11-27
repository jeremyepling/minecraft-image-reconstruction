"""Frozen challenger: compare signed atlas seam gradients on independent texels."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_geometry import best_translation
from engine import placed_materials
from engine.atlas_count import load_atlas
from engine.thin_atlas_faces import _corr


class SeamCatalog(placed_materials.Catalog):
    def face(self, pixels, uv, normal_gain, side):
        losses = super().face(pixels, uv, normal_gain, side)
        texels, group, count = np.unique(uv, return_inverse=True, return_counts=True)
        if len(texels) < 32:
            return losses
        gray = np.full(256, np.nan, dtype=np.float32)
        gray[texels] = np.bincount(group, weights=pixels.mean(1))/count
        starts, ends = [], []
        for direction in (1,16):
            valid = texels % 16 < 15 if direction == 1 else texels // 16 < 15
            start = texels[valid]
            start = start[np.isfinite(gray[start+direction])]
            starts.extend(start)
            ends.extend(start + direction)
        if len(starts) < 35:
            return losses
        starts, ends = np.array(starts), np.array(ends)
        measured = gray[ends] - gray[starts]
        if measured.std() < 2 or np.count_nonzero(abs(measured) > 2) < 12:
            return losses
        first = ((starts%16)//2 + (starts//16)//2)%2 == 0
        if min(first.sum(), (~first).sum()) < 10:
            return losses
        shortlist = np.argsort(losses,kind='stable')[:16]
        allowed = self.valid[side] & np.isin(self.candidates,shortlist)
        sprites, to_unique = np.unique(self.sprites[allowed],return_inverse=True)
        # Compare a face and its half-resolution mip. Geometry can register a
        # seam one texel either side of its canonical atlas position.
        art = self.maps[sprites,:2]
        best = np.zeros(len(sprites))
        for dx,dy in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
            a = (starts%16+dx)%16 + 16*((starts//16+dy)%16)
            b = (ends%16+dx)%16 + 16*((ends//16+dy)%16)
            slopes = art[...,b]-art[...,a]
            corr = np.minimum(_corr(slopes[...,first], measured[first]),
                              _corr(slopes[...,~first], measured[~first])).max((1,2))
            best = np.maximum(best,corr)
        agreement = np.zeros(len(self.blocks))
        np.maximum.at(agreement,self.candidates[allowed],best[to_unique])
        return losses - 17 * np.maximum(0,agreement-.42)


def compare(args):
    source, builds, name = args
    frame = source/'data'/name.split('-')[0]/'frames'/name
    manifest = json.loads((builds/name/'build-manifest.json').read_bytes())
    camera = manifest['placement']
    cells = sorted(decoded_pack(builds/name/'reconstruction.zip'))
    digest = sha256((frame/'rgb.png').read_bytes()).hexdigest()
    if not cells or camera.get('P') is None:
        return dict(id=name, rgb_sha256=digest, placed=len(cells), correct_geometry=0)
    # The two predictors finish before the evaluator opens the camera pose or labels.
    with Image.open(frame/'rgb.png') as image:
        atlas = load_atlas()
        base,_ = placed_materials.assign(image,camera,cells,atlas)
        atlas._placed_catalog = SeamCatalog(atlas)
        seam,_ = placed_materials.assign(image,camera,cells,atlas)
    pose = json.loads((frame/'camera.json').read_bytes())
    transform = registration(np.array(camera['P']),pose)
    world = np.floor((np.array(cells)+.5) @ transform[:3,:3].T + transform[:3,3]).astype(int)
    labels = json.loads((frame/'labels.json').read_bytes())['blocks']
    truth = {tuple(b['xyz_world']):b['block_id'].removeprefix('minecraft:') for b in labels}
    offset,_ = best_translation(set(map(tuple,world)),set(truth))
    before, after = Counter(), Counter()
    for cell, location in zip(cells,world+offset):
        actual = truth.get(tuple(location))
        if actual is not None:
            before[actual,base[cell]] += 1
            after[actual,seam[cell]] += 1
    return dict(id=name,rgb_sha256=digest,placed=len(cells),visible_truth=len(truth),translation=offset,
                correct_geometry=sum(before.values()),baseline=sum(v for (a,b),v in before.items() if a==b),
                seam=sum(v for (a,b),v in after.items() if a==b),
                differences=[[a,b,after[a,b]-before[a,b]] for a,b in sorted(before|after) if before[a,b]!=after[a,b]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--builds',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ids',nargs='+',required=True)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(compare,[(args.source,args.builds,n) for n in args.ids]))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(results,indent=2)+'\n')
