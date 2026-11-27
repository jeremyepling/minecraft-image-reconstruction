"""Prompt MobileSAM and SAM ViT-B from existing source-only support; export native masks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read
from engine.surface_perimeter import _native_alpha
from engine.plane_consensus import row_runs

IMPLEMENTATION = WORKTREES / 'surface-recovery-round/runs/mobile-sam-source'
IMPLEMENTATION_REV = 'f706ad9c4eb7f219c00d9050e46328518ffb65d2'
BASELINE = WORKTREES / 'floating-visible-builds/runs/floating-cloud-gate-01/sky'
OUTPUT = ROOT / 'runs/sky-crop-models'
MODELS = {
    'mobile': ('vit_t', IMPLEMENTATION / 'weights/mobile_sam.pt',
               '6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f'),
    'sam-b': ('vit_b', OUTPUT / 'sam_vit_b_01ec64.pth',
              'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912')
}


def prompts(mask, left, top):
    """Pick spaced positive seeds inside the existing mask and negatives in its gaps."""
    points, labels = [], []
    for label, support, limit in [(1, mask, 3), (0, ~mask, 2)]:
        work = np.pad(support.astype(np.uint8), 1)
        for _ in range(limit):
            distance = cv2.distanceTransform(work, cv2.DIST_L2, 5)
            y, x = np.unravel_index(distance.argmax(), work.shape)
            if distance[y, x] <= 0:
                break
            points.append([left + x - 1, top + y - 1])
            labels.append(label)
            cv2.circle(work, (int(x), int(y)), max(3, int(distance[y, x])), 0, -1)
    return np.asarray(points), np.asarray(labels)


def write_json(file, value):
    file.write_text(json.dumps(value, separators=(',', ':')), encoding='utf8')


@torch.inference_mode()
def run(name, zoom=False):
    registry_key, weights, weight_hash = MODELS[name]
    source_rev = subprocess.check_output(['git', '-C', str(IMPLEMENTATION), 'rev-parse', 'HEAD'], text=True).strip()
    if source_rev != IMPLEMENTATION_REV or digest(weights) != weight_hash:
        raise ValueError('Unexpected source or checkpoint')
    sys.path.insert(0, str(IMPLEMENTATION))
    from mobile_sam import SamPredictor, sam_model_registry
    model = sam_model_registry[registry_key]()
    model.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
    predictor = SamPredictor(model.eval().to('cuda'))
    strategies = ['zoom'] if zoom else ['box', 'points']
    outputs = {kind: {} for kind in strategies}
    dataset = SOURCE / 'floating-build-grid-01/fixtures.json'
    frames = {}
    for case in read(dataset)['cases']:
        frame = case['id']
        image_path = Path(case['variants']['sky']['image'])
        planes_path = BASELINE / frame / 'source-handoff/surfaces.json'
        image_hash = digest(image_path)
        if image_hash != case['variants']['sky']['source_sha256']:
            raise ValueError(f'Unexpected screenshot for {frame}')
        image = np.asarray(Image.open(image_path).convert('RGB'))
        h, w = image.shape[:2]
        planes = read(planes_path)['planes']
        if not zoom:
            predictor.set_image(image)
        frames[frame] = dict(split=case['split'], source_sha256=image_hash,
                             surfaces_sha256=digest(planes_path), proposals=len(planes))
        for kind in strategies:
            outputs[kind][frame] = {}
        for plane in planes:
            alpha, (left, top, right, bottom) = _native_alpha(plane['support'], (w, h))
            mask = np.asarray(alpha) > 0
            box = np.array([left, top, right - 1, bottom - 1])
            points, labels = prompts(mask, left, top)
            origin_x, origin_y = 0, 0
            if zoom:
                # Keep surrounding negative context while enlarging small supports.
                margin = max(16, max(right - left, bottom - top) // 2)
                origin_x, origin_y = max(0, left - margin), max(0, top - margin)
                patch = image[origin_y:min(h, bottom + margin), origin_x:min(w, right + margin)]
                predictor.set_image(np.ascontiguousarray(patch))
                points -= [origin_x, origin_y]
                box -= [origin_x, origin_y, origin_x, origin_y]
            for kind in strategies:
                extra = {} if kind == 'box' else dict(point_coords=points, point_labels=labels)
                masks, scores, _ = predictor.predict(box=box, multimask_output=True, **extra)
                # Use the model's own predicted IoU; the chooser gets each whole proposed mask.
                best = int(np.argmax(scores))
                runs = row_runs(masks[best])
                if zoom:
                    runs = [[y + origin_y, a + origin_x, b + origin_x] for y, a, b in runs]
                outputs[kind][frame][plane['id']] = dict(size=[w, h], runs=runs)
        print(name, frame, len(planes), flush=True)
    if sum(f['proposals'] for f in frames.values()) != 193:
        raise ValueError('Expected 193 source crops')
    hashes = {}
    for kind in strategies:
        path = OUTPUT / f'{name}-{kind}.json'
        write_json(path, dict(frames=outputs[kind]))
        hashes[kind] = digest(path)
    manifest = f'{name}-zoom-manifest.json' if zoom else f'{name}-manifest.json'
    write_json(OUTPUT / manifest, dict(frames=frames, model=name, checkpoint_sha256=weight_hash,
               implementation_rev=source_rev, fixtures_sha256=digest(dataset), output_sha256=hashes,
               torch_version=torch.__version__, gpu=torch.cuda.get_device_name()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--zoom', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run(args.model, args.zoom)
