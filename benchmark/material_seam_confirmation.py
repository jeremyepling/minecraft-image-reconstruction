"""Frozen challenger: compare signed atlas seam gradients on independent texels."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.count_export import _archive_pack
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_geometry import best_translation
from benchmark.strip_visibility_audit import horizontal_slivers
from engine import placed_materials
from engine.atlas_count import load_atlas
from engine.export import JAVA_26_3_FORMAT, write_pack
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


def pack_hash(ids):
    raw, archive = BytesIO(), BytesIO()
    write_pack(raw, ids, JAVA_26_3_FORMAT)
    with ZipFile(archive,'w') as outer:
        _archive_pack(outer,'reconstruction.zip',raw)
    with ZipFile(archive) as outer:
        payload = outer.read('reconstruction.zip')
    assert decoded_pack(BytesIO(payload)) == {c:'minecraft:'+v for c,v in ids.items()}
    return sha256(payload).hexdigest()


TARGET_PAIRS = (('polished_andesite','stone_bricks'),('stone_bricks','polished_andesite'),('tuff','light_gray_wool'),
                ('polished_deepslate','tuff'),('smooth_stone','stripped_pale_oak_wood'),('white_concrete','iron_block'),
                ('white_concrete','polished_andesite'),('spruce_planks','stripped_dark_oak_wood'))


def pair_counts(before,after):
    return [[a,b,before[a,b],after[a,b]] for a,b in TARGET_PAIRS if before[a,b] or after[a,b]]


def truth_and_optional(frame):
    labels = json.loads((frame/'labels.json').read_bytes())
    with np.load(frame/'labels.npz') as regions:
        optional = horizontal_slivers(labels,regions['face_id'])
    truth = {tuple(b['xyz_world']):b['block_id'].removeprefix('minecraft:') for b in labels['blocks']}
    return truth,optional


def compare(args):
    source, builds, name = args
    frame = source/'data'/name.split('-')[0]/'frames'/name
    manifest = json.loads((builds/name/'build-manifest.json').read_bytes())
    camera = manifest['placement']
    cells = sorted(decoded_pack(builds/name/'reconstruction.zip'))
    digest = sha256((frame/'rgb.png').read_bytes()).hexdigest()
    if manifest['image_sha256'] != digest:
        raise ValueError(f'Build and source hashes disagree for {name}')
    geometry_hash = sha256((builds/name/'reconstruction.zip').read_bytes()).hexdigest()
    if not cells or camera.get('P') is None:
        truth,optional = truth_and_optional(frame)
        return dict(id=name, rgb_sha256=digest, pack_sha256=geometry_hash, placed=len(cells),
                    baseline_typed_sha256=pack_hash({}), seam_typed_sha256=pack_hash({}),
                    visible_truth=len(truth), correct_geometry=0, baseline=0, seam=0, gained=0,lost=0,
                    required_truth=len(truth)-len(optional),optional_placed=0,facade_before=0,facade_after=0,
                    target_pairs=[],differences=[])
    # The two predictors finish before the evaluator opens the camera pose or labels.
    with Image.open(frame/'rgb.png') as image:
        atlas = load_atlas()
        base,_ = placed_materials.assign(image,camera,cells,atlas)
        atlas._placed_catalog = SeamCatalog(atlas)
        seam,_ = placed_materials.assign(image,camera,cells,atlas)
    pose = json.loads((frame/'camera.json').read_bytes())
    transform = registration(np.array(camera['P']),pose)
    world = np.floor((np.array(cells)+.5) @ transform[:3,:3].T + transform[:3,3]).astype(int)
    truth,optional = truth_and_optional(frame)
    offset,_ = best_translation(set(map(tuple,world)),set(truth))
    before, after = Counter(), Counter()
    gained = lost = 0
    facade_before = facade_after = optional_placed = 0
    for cell, location in zip(cells,world+offset):
        actual = truth.get(tuple(location))
        if actual is not None:
            before[actual,base[cell]] += 1
            after[actual,seam[cell]] += 1
            gained += base[cell] != actual and seam[cell] == actual
            lost += base[cell] == actual and seam[cell] != actual
            if tuple(location) in optional:
                optional_placed += 1
            else:
                facade_before += base[cell] == actual
                facade_after += seam[cell] == actual
    return dict(id=name,rgb_sha256=digest,pack_sha256=geometry_hash,placed=len(cells),visible_truth=len(truth),translation=offset,
                correct_geometry=sum(before.values()),baseline=sum(v for (a,b),v in before.items() if a==b),
                seam=sum(v for (a,b),v in after.items() if a==b),
                baseline_typed_sha256=pack_hash(base),seam_typed_sha256=pack_hash(seam),
                gained=gained,lost=lost,target_pairs=pair_counts(before,after),
                required_truth=len(truth)-len(optional),optional_placed=optional_placed,
                facade_before=facade_before,facade_after=facade_after,
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
