"""Independent replay of the Blocker Picker exposure-only control."""
import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
from engine.geometry import face
from engine.surface_model import canonical
from calibration.retrieval import features, rank

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(calibration, out):
    root = Path(__file__).resolve().parents[1]
    reviews = root/'benchmark/reviews'
    materials = reviews/'count-01/materials.json'
    frames = json.loads((reviews/'mask-four-01/manifest.json').read_text())['frames']
    hashes = {f['key']: f['sha256'] for f in frames}
    library = calibration/'face_textures/softened-library.npz'
    index = calibration/'atlas/placement-index.json'
    catalog = json.loads(index.read_text())
    choices = defaultdict(set)
    for side in catalog['choices'].values():
        for tex, ids in side.items():
            choices[tex].update(ids)
    with np.load(library) as bank:
        assert str(bank['version']) == '26.3' and str(bank['atlas_sha256']) == catalog['atlas_sha256']
        eligible = np.isin(bank['labels'], list(choices))
        features_ref = features(bank['tiles'][eligible])
        labels = bank['labels'][eligible]
        method = str(bank['method'])
    rows, crops = [], []
    for picture in json.loads(materials.read_text())['pictures']:
        key = picture['key']
        name = key.split('/')[-1]+'.png'
        image = next(p for p in (root/'benchmark/data').glob('*/captures/'+name) if sha(p)==hashes[key])
        rgb = np.asarray(Image.open(image).convert('RGB'))
        for group in ('known', 'predicted'):
            for idx, item in enumerate(picture[group]):
                if group == 'known' and not (item['fixed_sample'] or item['paired_tile'] is not None):
                    continue
                if 'truth' not in item:
                    continue
                crops.append(face(rgb,np.asarray(canonical(item['quad'])),16))
                rows.append(dict(key=key,group=group,index=idx,truth=item['truth'],xyz=item['xyz'],
                                 fixed=item.get('fixed_sample',False),paired=item.get('paired_tile') is not None))
    for row, (names, distances, gap) in zip(rows,rank(np.stack(crops),features_ref,labels,method,True),strict=True):
        row.update(textures=names,block=min(choices[names[0]],key=lambda x:(x!=names[0],x)),distances=distances,margin=gap)
    scores = {}
    for name, subset in [('fixed',[r for r in rows if r['fixed']]),('paired_known',[r for r in rows if r['paired']]),('predicted',[r for r in rows if r['group']=='predicted'])]:
        scores[name] = [sum(r['block']==r['truth'] for r in subset),len(subset)]
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(scores=scores,library_sha256=sha(library),index_sha256=sha(index),materials_sha256=sha(materials),frames=hashes,atlas_sha256=catalog['atlas_sha256'],method=method,rows=rows)))
    print(scores)
if __name__ == '__main__':
    run(*map(Path,sys.argv[1:]))
