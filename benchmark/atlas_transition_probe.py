"""Controlled test of unit scale at a visible joint between two coplanar faces.

Reference geometry supplies the shared edge and its endpoints. The atlas matcher
gets RGB crops only. This measures one ingredient of camera recovery.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.texture_camera_eval import checked, frames
from engine.geometry import UNIT, project, transform
from engine.surface_model import _material_assets


def pair_pool(faces):
    usable = {(tuple(f['xyz']), tuple(f['normal'])): f for f in faces
              if f['visible_fraction'] >= .98 and all(f['corner_visible']) and f['visible_pixels'] >= 32}
    for (loc, normal), left in usable.items():
        for axis in range(3):
            if normal[axis]:
                continue
            def neighbor(step):
                xyz = list(loc)
                xyz[axis] += step
                return usable.get((tuple(xyz), normal))
            right = neighbor(1)
            if right is None:
                continue
            kind = 'repeat' if right['block_id'] == left['block_id'] else 'change'
            before, after = neighbor(-1), neighbor(2)
            isolated = bool(kind == 'change' and before and after and
                            before['block_id'] != left['block_id'] and after['block_id'] != right['block_id'])
            plane = 'horizontal' if normal[1] else 'wall'
            yield (plane, kind), (left, right, axis, isolated)


def crop_quad(f, axis, seam, size):
    xyz = np.array(f['world_corners'])
    edge_center = UNIT[xyz[:, axis] == seam].mean(0)
    uv = edge_center + size * (UNIT - edge_center)
    return project(transform(f['image_corners']), uv)


def measure(matcher, rgb, pair):
    left, right, axis, isolated = pair
    seam = right['xyz'][axis]
    def assess(size):
        return [matcher.rank(rgb, crop_quad(f, axis, seam, size)) for f in (left, right)]
    full = assess(1.)
    half = assess(.5)
    # Both hypotheses use the same genuine boundary, stay on visible pixels,
    # and can select different atlas entries on either side.
    score = lambda matches: min(m['scores'][0] for m in matches)
    names = [m['block_id'].removeprefix('minecraft:') for m in full]
    truth = [f['block_id'].removeprefix('minecraft:') for f in (left, right)]
    return dict(xyz=left['xyz'], normal=left['normal'], axis=axis, isolated=isolated,
                truth=truth, predicted=names, textures=[m['texture'] for m in full],
                full_score=score(full), half_score=score(half),
                exact_pair=names == truth, classified=(names[0] == names[1]) == (truth[0] == truth[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=['train','validation','test','regression'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pairs-per-group', type=int, default=3)
    args = parser.parse_args()
    _, matcher, path, errors = _material_assets('26.3','texture-guided-repeat','shape',None)
    if matcher is None:
        raise ValueError(f'Prepared 26.3 atlas required: {errors}')
    rows, counts = [], Counter()
    root = args.data / args.split / 'frames'
    for frame in sorted(frames(args.data.parent,args.split),key=lambda r:r['id']):
        directory = root/frame['id']
        refs = json.loads(checked(directory/'reference-faces.json',frame['files']['reference-faces.json']).read_bytes())
        pools = defaultdict(list)
        for group, pair in pair_pool(refs['faces']):
            key = f'{directory.name}/{pair[0]["xyz"]}/{pair[0]["normal"]}/{pair[2]}'
            pools[group].append((hashlib.sha256(key.encode()).hexdigest(), pair))
        with Image.open(checked(directory/'rgb.png',frame['files']['rgb.png'])) as img:
            rgb = np.asarray(img.convert('RGB'))
        for (plane, kind), pool in pools.items():
            counts[plane + '_' + kind] += len(pool)
            for _, pair in sorted(pool)[:args.pairs_per_group]:
                rows.append(dict(frame=directory.name, plane=plane, kind=kind, **measure(matcher, rgb, pair)))
    def summarize(items):
        return dict(pairs=len(items), identified_relation=sum(r['classified'] for r in items),
                    both_ids_exact=sum(r['exact_pair'] for r in items),
                    unit_beats_half=sum(r['full_score'] > r['half_score'] for r in items),
                    tie=sum(r['full_score'] == r['half_score'] for r in items))
    groups = sorted(set((r['plane'], r['kind']) for r in rows))
    summary = {p+'_'+k:summarize([r for r in rows if (r['plane'],r['kind'])==(p,k)]) for p,k in groups}
    summary['isolated_changes'] = summarize([r for r in rows if r['isolated']])
    meta = json.loads((Path(path)/'atlas.json').read_bytes())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(split=args.split, atlas_sha256=meta['atlas_sha256'],
        pairs_per_frame_group=args.pairs_per_group, eligible=counts, summary=summary, rows=rows), separators=(',',':')))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
