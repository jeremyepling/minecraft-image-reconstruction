"""Generate native-coordinate ScaleLSD v2 evidence from the frozen screenshot suite.

Clone the pinned ant-research/scalelsd revision and download the official checkpoint
listed below. Use the package versions in the evidence manifest. Run from the repo:
python -m benchmark.scalelsd_fields --upstream runs/upstream --checkpoint runs/scalelsd-vitbase-v2-train-sa1b.pt
"""
import argparse
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'https://github.com/ant-research/scalelsd'
UPSTREAM_REV = 'e2751fcd0eb82694abeb12ef7ec9d46237e2ca06'
MODEL_REV = 'f0a119fb3d4a12c53c317a8746db80d267fb2a0f'
MODEL_NAME = 'scalelsd-vitbase-v2-train-sa1b.pt'
MODEL_URL = f'https://huggingface.co/cherubicxn/scalelsd/resolve/{MODEL_REV}/{MODEL_NAME}'
MODEL_SHA256 = 'd0a7333c4de518a70af7d6ea4a283aa0ce9ffe1047525373b94070b1d1d4003f'
SETTINGS = dict(input_size=[512, 512], line_support_strictly_above=10,
                junction_heatmap_threshold=.008, max_junctions=512, use_nms=False,
                use_lsd=False, seed=0)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def revision(source):
    return subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()


def capture(model, gray, device, verify_transform):
    import cv2
    import numpy as np
    import torch

    height, width = gray.shape
    small = cv2.resize(gray, tuple(SETTINGS['input_size']), interpolation=cv2.INTER_LINEAR)
    tensor = (torch.from_numpy(small).float() / 255.)[None, None].to(device)
    meta = dict(width=width, height=height, filename='', use_lsd=SETTINGS['use_lsd'],
                use_nms=SETTINGS['use_nms'])
    if device == 'cuda':
        torch.cuda.synchronize()
    start = perf_counter()
    output = model(tensor, meta)[0][0]
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = perf_counter() - start
    # The official forward pass multiplies x and y by native size / field size.
    # Its tensors are already in the source coordinate frame.
    if verify_transform:
        network = model(tensor, dict(meta, width=512, height=512))[0][0]
        for key in ('lines_pred', 'juncs_pred'):
            fixed = network[key].reshape(-1, 2).cpu().numpy()
            np.testing.assert_allclose(output[key].reshape(-1, 2).cpu().numpy(),
                                       fixed * [width / 512., height / 512.], rtol=2.e-5, atol=.01)
    accepted = output['lines_score'] > SETTINGS['line_support_strictly_above']
    segments = output['lines_pred'][accepted].cpu().numpy()
    junctions = output['juncs_pred'].cpu().numpy()
    for points in (segments.reshape(-1, 2), junctions):
        if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > [width, height]):
            raise ValueError('ScaleLSD returned a coordinate outside the native image')
    return dict(image_size=[width, height], segments=segments.tolist(), junctions=junctions.tolist(),
                segment_scores=output['lines_score'][accepted].cpu().tolist(),
                junction_scores=output['juncs_score'].cpu().tolist(), inference_seconds=elapsed,
                transform=dict(grayscale='OpenCV IMREAD_GRAYSCALE', resize='OpenCV INTER_LINEAR',
                               input_size=SETTINGS['input_size'], padding_ltrb=[0, 0, 0, 0],
                               input_to_native_xy=[width / 512., height / 512.],
                               coordinates='native XY via official forward_test jscales'))


def generate(upstream, checkpoint, out, cases, device):
    if revision(upstream) != UPSTREAM_REV or digest(checkpoint) != MODEL_SHA256:
        raise ValueError('ScaleLSD checkout or checkpoint differs from the pinned artifact')
    tracked_changes = subprocess.check_output(
        ['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'])
    if tracked_changes:
        raise ValueError('ScaleLSD inference requires the pristine upstream source')

    import cv2
    import torch
    sys.path.insert(0, str(upstream.resolve()))
    from scalelsd.ssl.misc.train_utils import fix_seeds, load_scalelsd_model

    fix_seeds(SETTINGS['seed'])
    model = load_scalelsd_model(str(checkpoint), device)
    model.junction_threshold_hm = SETTINGS['junction_heatmap_threshold']
    model.num_junctions_inference = SETTINGS['max_junctions']
    rows = json.loads(cases.read_bytes())
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    frame_ids = {}
    seconds = 0.
    for row in rows:
        source = cases.parent / row['source']
        sha = row['sha256']
        if sha != digest(source):
            raise ValueError(f'Screenshot hash mismatch for {source}')
        frame_ids[row['key']] = sha
        if sha in files:
            continue
        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f'Unreadable source screenshot {source}')
        with torch.inference_mode():
            result = capture(model, gray, device, verify_transform=not files)
        result['image_sha256'] = sha
        payload = json.dumps(result, separators=(',', ':'), allow_nan=False).encode()
        dest = out / (sha + '.json.gz')
        tmp = dest.with_suffix('.tmp')
        tmp.write_bytes(gzip.compress(payload, mtime=0))
        tmp.replace(dest)
        files[sha] = digest(dest)
        seconds += result['inference_seconds']
        print(f"{len(files)}  {row['key']}  {len(result['segments'])} lines", flush=True)
    manifest = dict(method='ScaleLSD v2', upstream_url=UPSTREAM, upstream_revision=UPSTREAM_REV,
                    checkpoint_url=MODEL_URL, checkpoint_revision=MODEL_REV,
                    checkpoint_sha256=MODEL_SHA256, wrapper_revision=revision(ROOT),
                    wrapper_sha256=digest(Path(__file__)), fixture_sha256=digest(cases),
                    settings=SETTINGS, files=files, frame_ids=frame_ids,
                    inference_seconds=seconds, cache_read_seconds=0.,
                    timing='Synchronized model forward only, once per unique image. Model loading, IO and one additional transform self-test excluded.',
                    transform_check='Re-ran first frame with 512x512 output metadata and verified every native x/y coordinate independently.',
                    platform=platform.platform(), python=sys.version, device=device,
                    gpu=torch.cuda.get_device_name() if device == 'cuda' else None, cuda=torch.version.cuda,
                    packages={k: version(k) for k in ('torch', 'torchvision', 'timm', 'opencv-python-headless',
                                                        'numpy', 'matplotlib', 'python-json-logger')})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=ROOT / 'benchmark/reviews/normals-01/cases.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'benchmark/data/mask-lines-v1/scalelsd')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    generate(args.upstream, args.checkpoint, args.out, args.cases, args.device)
