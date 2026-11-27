"""Replay the frozen signed-gradient challenger on the published typed geometry."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from benchmark.mask_blocks import decoded_pack
from benchmark.material_seam_confirmation import SeamCatalog, pack_hash, pair_counts
from engine.atlas_count import load_atlas
from engine.placed_materials import assign


@lru_cache(maxsize=1)
def assets():
    atlas = load_atlas()
    atlas._placed_catalog = SeamCatalog(atlas)
    return atlas


def replay(task):
    cache, release, name = task
    source = json.loads((cache/'inputs'/f'{name}.json').read_bytes())
    rgb = Path(source['rgb'])
    if sha256(rgb.read_bytes()).hexdigest() != source['source_hash']:
        raise ValueError(f'Source hash mismatch for {name}')
    cells = list(map(tuple,source['cells']))
    with Image.open(rgb) as image:
        prediction,_ = assign(image,source['camera'],cells,assets())
    with ZipFile(release/'typed-exports.zip') as z:
        baseline_bytes = z.read(f'{name}/reconstruction.zip')
        baseline = decoded_pack(BytesIO(baseline_bytes))
    if set(cells) != prediction.keys() or set(cells) != baseline.keys():
        raise ValueError(f'Geometry changed for {name}')
    ref = json.loads((cache/'reference'/f'{name}.json').read_bytes())
    truth = {tuple(row[:3]):row[3].removeprefix('minecraft:') for row in ref['truth']}
    optional = set(map(tuple,ref['optional']))
    right_before = right_after = facade_before = facade_after = positions = 0
    before_counts, after_counts = Counter(),Counter()
    gained = lost = 0
    for c,loc in zip(cells,ref['aligned']):
        loc = tuple(loc)
        if loc in truth:
            positions += 1
            before = baseline[c].removeprefix('minecraft:') == truth[loc]
            after = prediction[c] == truth[loc]
            before_counts[truth[loc],baseline[c].removeprefix('minecraft:')] += 1
            after_counts[truth[loc],prediction[c]] += 1
            gained += after and not before
            lost += before and not after
            right_before += before
            right_after += after
            facade_before += before and loc not in optional
            facade_after += after and loc not in optional
    return dict(id=name, source_hash=source['source_hash'], positions=positions, baseline=right_before, seam=right_after,
                geometry_pack_sha256=source['geometry_pack_sha256'],baseline_typed_sha256=sha256(baseline_bytes).hexdigest(),
                seam_typed_sha256=pack_hash(prediction),gained=gained,lost=lost,target_pairs=pair_counts(before_counts,after_counts),
                facade_before=facade_before, facade_after=facade_after)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    cases = json.loads((args.release/'cases.json').read_bytes())
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(replay,[(args.cache,args.release,r['id']) for r in cases]))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rows,indent=2)+'\n')
