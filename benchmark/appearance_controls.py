"""Validate optional visual matching on mixed-material, correctly aligned Minecraft faces."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from benchmark.textures import extract_split
from benchmark.color_gallery import sample_labels
from engine.routing import alignment_scores, trusted_cells
from calibration.projection import camera_project, south_quad
from calibration.retrieval import features
from engine.appearance import cost_matrix, palette_cost, prototypes
from engine.geometry import face
from engine.reconstruction import exemplar_bank, opaque_sprites


def load(path):
    return json.loads(path.read_text())


def catalog_groups(root):
    layout = load(root / 'layout.json')
    pages = defaultdict(list)
    for block in layout['blocks']:
        if block['split'] == 'test' and block.get('side_texture') and not block['invisible']:
            pages[block['page']].append(block)
    for view in load(root / 'views.json'):
        page = pages[view['page']]
        if not page or view.get('kind') == 'adjacent':
            continue
        source = root / 'captures' / view['id']
        image = np.asarray(Image.open(source.with_suffix('.png')).convert('RGB'))
        pose = load(source.with_suffix('.json'))
        states = {(b['x'],b['y'],b['z']): b['block_id'].removeprefix('minecraft:')
                  for b in load(source.with_name(source.name+'-blocks.json'))}
        tiles, ids, widths = [], [], []
        for b in page:
            pos = (b['x'],b['y'],b['z'])
            if states.get(pos) != b['block_id']:
                continue
            q = camera_project(south_quad(*pos),pose)
            if q.min()<0 or (q >= np.array(pose['image_size'])-1).any():
                continue
            tiles.append(face(image,q,16))
            widths.append(np.linalg.norm(q[1]-q[0]))
            ids.append(b['side_texture'])
        if tiles:
            yield 'registry',tiles,ids,widths


def run(args):
    root = args.calibration
    atlas = root/'atlas'
    catalog = load(atlas/'placement-index.json')
    bank, labels, method = exemplar_bank(atlas,root/'face_textures/softened-library.npz',catalog)
    atlas_tiles, candidates = opaque_sprites(atlas,catalog)
    centers = prototypes(atlas_tiles,candidates,bank,labels)
    eligible = np.isin(candidates,list(catalog['choices']['south']))
    centers, candidates = centers[eligible], candidates[eligible]
    eligible = np.isin(labels,list(catalog['choices']['south']))
    train,labels = features(bank[eligible]),labels[eligible]
    groups = list(catalog_groups(root)) if args.split=='test' else []
    rows,_ = extract_split(args.shapes,['normal','small','warm-shader'],args.split)
    views = defaultdict(list)
    for r in rows:
        if r['crop'] == 'known':
            views[r['condition'],r['view']].append(r)
    for (condition,_), obs in views.items():
        groups.append((condition,[r['tile'] for r in obs],[r['texture'] for r in obs],[r['px'] for r in obs]))
    if args.gallery:
        for folder in sorted(args.gallery.glob('test-*-front')):
            source=args.shapes/'captures'/folder.name
            rgb=np.asarray(Image.open(source.with_suffix('.png')).convert('RGB'))
            geometry=load(folder/'final-build/geometry.json')['cells']
            truth,_=sample_labels(geometry,load(source.with_suffix('.json')),
                                  load(source.with_name(source.name+'-blocks.json')))
            planes=defaultdict(list)
            for cell,label in zip(geometry,truth):
                q=np.asarray(cell['quad'])
                planes[cell['plane']].append((face(rgb,q,16),label,np.linalg.norm(q[1]-q[0])))
            for rows in planes.values():
                tiles,truth,widths=zip(*rows)
                groups.append(('gallery',tiles,truth,widths))
    counts = defaultdict(lambda: defaultdict(int))
    for condition, tiles, truth, widths in groups:
        batch = np.stack(tiles)
        cost = cost_matrix(batch,centers)
        ranked,evidence = alignment_scores(batch,train,labels,method)
        base_trust = trusted_cells(evidence)
        trusted = trusted_cells(evidence,widths)
        guesses = {'exposure': np.array([r[0][0] for r in ranked])}
        for shared in [False, True]:
            scores = palette_cost(cost,shared)
            guesses['shared' if shared else 'local'] = candidates[scores.argmin(1)]
        guesses['fixed_threshold_route'] = np.where(base_trust,guesses['exposure'],guesses['shared'])
        guesses['routed'] = np.where(trusted,guesses['exposure'],guesses['shared'])
        labeled = np.array([t is not None for t in truth])
        truth = np.asarray(truth,dtype=object)
        counts[condition]['faces'] += int(labeled.sum())
        counts[condition]['routed_to_exposure'] += int((trusted & labeled).sum())
        counts[condition]['strong_spatial_match'] += int(((evidence>=.8) & labeled).sum())
        for mode, predicted in guesses.items():
            counts[condition][mode] += int(np.count_nonzero((np.asarray(predicted)==truth) & labeled))
    args.out.write_text(json.dumps(dict(known_face_texture_ids=counts),indent=2)+'\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shapes',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gallery',type=Path)
    p.add_argument('--split',choices=['dev','test'],default='test')
    run(p.parse_args())
