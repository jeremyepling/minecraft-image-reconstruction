"""Measure the answer-assisted ceiling of the photographed-neighbor proposal pool."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.texture_camera_eval import DATA, STABLE, checked, frames, read, save
from benchmark.visible_depth_transfer import rgb_path
from training.visible_labels import visible_labels


def evaluate(frame, cache, runs, move):
    key = frame['id']
    if frame.get('split') == 'train':
        folder = DATA/'data'/'train'/frame['directory']
        png = folder/'rgb.png'
        inp = read(STABLE/'runs/visible-depth-rounds/inputs'/(key+'.json'))
        camera, parts, counts = inp['camera'], inp['parts'], inp['counts']
        pose = read(folder/'camera.json')
        truth = {tuple(b['xyz_world']) for b in read(folder/'labels.json')['blocks']}
    else:
        png = rgb_path(frame)
        upstream = runs/'upstream'/key
        camera = frame['camera']
        parts = read(upstream/'surfaces.json')['planes']
        counts = read(upstream/'counts.json')
        pose = read(png.with_suffix('.json'))
        states = read(png.with_name(frame['source_id']+'-blocks.json'))
        states = {tuple(r[k] for k in 'xyz'):r for r in states}
        visible, _, _ = visible_labels(pose, states, (0,0,0))
        truth = {tuple(b['xyz_world']) for b in visible['blocks']}
    with Image.open(checked(png,frame['files']['rgb.png'])) as image:
        digest = hashlib.sha256(image.convert('RGB').tobytes())
        digest.update(json.dumps(['26.3', image.size, parts, counts, camera], sort_keys=True).encode())
    intermediate = read(cache/f'{digest.hexdigest()}.json')
    projection = registration(np.array(camera['P']), pose)
    def world(point):
        return tuple((np.floor((np.array(point)+.5) @ projection[:3,:3].T+projection[:3,3])
                      + np.asarray(move)).astype(int).tolist())
    base = {world(cell) for cell in intermediate['base']}
    proposals = {world(cell) for cell, *_ in intermediate['rows']}
    return dict(target=len(truth), baseline_correct=len(base & truth), baseline_extra=len(base-truth),
                proposed=len(proposals), recoverable=len((truth-base) & proposals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--scored', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    transfer = STABLE/'runs/visible-depth-rounds/transfer'
    summary = {}
    for split, samples in [('train',frames(DATA, 'train')),('old', read(transfer/'screen.json')['selected'])]:
        translations = {r['id']:r['baseline_translation'] for r in read(args.scored/split/'cases.json')}
        counts = Counter()
        for frame in samples:
            counts.update(evaluate(frame, args.cache, transfer, translations[frame['id']]))
        total = dict(counts)
        total['oracle_f1'] = 2*(counts['baseline_correct']+counts['recoverable'])/(
            counts['target']+counts['baseline_correct']+counts['baseline_extra']+counts['recoverable'])
        summary[split] = total
    save(args.out, summary)
    print(summary)


if __name__ == '__main__':
    main()
