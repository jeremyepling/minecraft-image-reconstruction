"""Compare bounded crop adjustments with rendered exposure matching on count-01."""
import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from calibration.retrieval import features, rank
from engine.atlas import digest
from engine.geometry import UNIT, project, transform, face
from engine.surface_model import canonical
from engine.reconstruction import exemplar_bank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite = root / 'benchmark/reviews/count-01'
    calibration = args.calibration.resolve()
    catalog = json.loads((calibration/'atlas/placement-index.json').read_bytes())
    library = calibration/'face_textures/softened-library.npz'
    tiles, labels, method = exemplar_bank(calibration/'atlas', library, catalog)
    mapping = defaultdict(set)
    for side in catalog['choices'].values():
        for texture, ids in side.items():
            mapping[texture].update(ids)
    eligible = np.isin(labels, list(mapping))
    train = features(tiles[eligible])
    labels = labels[eligible]
    choices = [(0., 0., 1.)] + [v for v in product((0., -.12, .12), (0., -.12, .12), (1., .92, 1.08)) if v != (0,0,1)]
    paths = {}
    for idx in (root/'benchmark/data').glob('*/index.json'):
        for frame in json.loads(idx.read_bytes()).get('frames', []):
            if 'image' in frame:
                paths[frame['image_sha256']] = idx.parent/frame['image']
    frames = {f['key']: f for f in json.loads((suite/'manifest.json').read_bytes())['frames']}
    rows = []
    for pic in json.loads((suite/'materials.json').read_bytes())['pictures']:
        key = pic['key']
        sha = frames[key]['sha256']
        path = paths[sha]
        assert digest(path) == sha
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB'))
        groups = defaultdict(list)
        for i, record in enumerate(pic['known']):
            if record['fixed_sample']:
                groups['fixed', i].append(record)
            if record['paired_tile'] is not None:
                groups['paired_known',i].append(record)
        for record in pic['predicted']:
            groups['predicted',record['proposal']].append(record)
        for (kind, group), records in groups.items():
            # Recorded corners are a diagnostic. Only predicted cells need boundary recovery.
            variants = choices if kind == 'predicted' and len(records)>=2 else choices[:1]
            charts = [transform(canonical(r['quad'])) for r in records]
            proposals = [[project(chart,(UNIT-.5)*s+.5+[x,y]) for chart in charts] for x,y,s in variants]
            samples = np.stack([face(rgb,q,16) for quads in proposals for q in quads])
            ranks = list(rank(samples, train, labels, method, True))
            batches = [ranks[i:i+len(records)] for i in range(0,len(ranks),len(records))]
            scores = [np.mean([r[1][0] for r in batch]) for batch in batches]
            selected = int(np.argmin(scores))
            def pick(label):
                return min(mapping[label],key=lambda b:(b!=label,b))
            for i,(record,ranking) in enumerate(zip(records,batches[selected])):
                names, distances, margin = ranking
                rows.append(dict(key=key,kind=kind,quad=record['quad'],adjusted_quad=proposals[selected][i].tolist(),
                    block_id=pick(names[0]),texture=names[0],ranked_ids=[pick(n) for n in names],
                    direction=record['direction'],truth=record['truth'],xyz=record['xyz'],image=str(path),baseline_id=pick(batches[0][i][0][0]),
                    offset=variants[selected],margin=margin,distances=distances,atlas_id=record['block_id']))
    args.out.mkdir(exist_ok=True,parents=True)
    result=dict(rows=rows,version='26.3',method=method,variants=choices,library_sha256=digest(library),
                atlas_sha256=catalog['atlas_sha256'],frame_sha256={k:f['sha256'] for k,f in frames.items()},
                code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                script_sha256=digest(Path(__file__)),input_sha256=digest(suite/'materials.json'))
    (args.out/'results.json').write_text(json.dumps(result,indent=2))
    for kind in ('predicted','fixed'):
        sub=[r for r in rows if r['kind']==kind]
        print(kind,len(sub),{m:sum(r[m]==r['truth'] for r in sub) for m in ('atlas_id','baseline_id','block_id')})

if __name__=='__main__':
    main()

