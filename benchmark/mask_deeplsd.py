"""Cache MegaDepth DeepLSD lines and endpoint junctions from the frozen screenshots."""
import argparse
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://github.com/cvg/DeepLSD'
REVISION = 'f7d9d6258c0cd25d4f6eea882853565403d289be'
PYTLSD = '21381ca4c5c8da297ef44ce07dda115075c5a648'
CHECKPOINT = 'https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar'
WEIGHTS_SHA256 = '56e9bae263977caa289ae49802987a67d4f00f376ae7e542176c8ae0a6cc2083'
MAX_SIDE = 960
JUNCTION_RADIUS_NATIVE = 5.
MIN_CORNER_DEGREES = 20.
CONF = dict(multiscale=False, detect_lines=True, line_detection_params=dict(
    merge=False, filtering='normal', grad_thresh=3, grad_nfa=True))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rev(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip()


def junctions(lines):
    # DeepLSD's quickstart produces line endpoints. Derive a junction only when
    # endpoints of two nonparallel segments meet within the declared tolerance.
    segments = lines.reshape(-1, 2, 2)
    ends = segments.reshape(-1, 2)
    if not len(segments):
        return []
    vectors = segments[:, 1] - segments[:, 0]
    directions = vectors / np.maximum(1e-8, np.linalg.norm(vectors, axis=1, keepdims=True))
    candidates = []
    for a, b in sorted(cKDTree(ends).query_pairs(JUNCTION_RADIUS_NATIVE)):
        if a//2 != b//2 and abs(directions[a//2] @ directions[b//2]) <= np.cos(np.radians(MIN_CORNER_DEGREES)):
            candidates.append((ends[a]+ends[b])/2)
    return np.round(candidates, 5).tolist()


def infer(path, net, device, sha):
    if digest(path) != sha:
        raise ValueError(f'Screenshot checksum differs: {path}')
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1., MAX_SIDE/max(w, h))
    nw, nh = round(w*scale), round(h*scale)
    start = time.perf_counter()
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(small).to(device=device, dtype=torch.float32)[None,None]/255.
    with torch.inference_mode():
        # The official detector converts final endpoints from its learned fields into NumPy here.
        model_xy = np.asarray(net({'image': tensor})['lines'][0], np.float64).reshape(-1,2,2)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    native = (model_xy+.5) * np.array([w/nw, h/nh])
    points = junctions(native)
    elapsed = time.perf_counter()-start
    if not np.isfinite(native).all():
        raise ValueError('Nonfinite DeepLSD segment')
    return dict(image_sha256=sha, image_size=[w,h], segments=native.reshape(-1,4).round(5).tolist(),
                junctions=points, inference_seconds=elapsed, transform=dict(resized_size=[nw,nh],
                padding_ltrb=[0,0,0,0], native_from_model_center_xy=[[w/nw,0,.5*w/nw],[0,h/nh,.5*h/nh],[0,0,1]]))


def main():
    from importlib.metadata import version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'benchmark/data/mask-lines-v1/deeplsd')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if rev(args.upstream) != REVISION or rev(args.upstream/'third_party/pytlsd') != PYTLSD:
        raise ValueError('Use the pinned upstream and pytlsd revisions')
    if digest(args.checkpoint) != WEIGHTS_SHA256:
        raise ValueError('MegaDepth checkpoint checksum differs')
    sys.path.insert(0, str(args.upstream.resolve()))
    from deeplsd.models.deeplsd_inference import DeepLSD
    import pytlsd

    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    net = DeepLSD(CONF)
    state = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False)
    net.load_state_dict(state['model'], strict=True)
    net.to(device).eval()
    del state
    startup = time.perf_counter()-start
    review = ROOT/'benchmark/reviews/normals-01'
    rows = json.loads((review/'cases.json').read_text())
    if len(rows)!=56 or len({r['key'] for r in rows})!=56 or len({r['sha256'] for r in rows})!=52:
        raise ValueError('Unexpected frozen fixture set')
    args.output.mkdir(parents=True, exist_ok=True)
    files, times, counts = {}, {}, {}
    for row in rows:
        sha = row['sha256']
        if sha in files:
            continue
        data = infer(review/row['source'], net, device, sha)
        path = args.output/(sha+'.json.gz')
        path.write_bytes(gzip.compress(json.dumps(data, separators=(',',':'), allow_nan=False).encode(), mtime=0))
        files[sha] = digest(path)
        times[sha] = data['inference_seconds']
        counts[sha] = len(data['segments'])
        print(f"{len(files)}/52 {row['key']} {counts[sha]} segments in {times[sha]:.2f}s", flush=True)
    manifest = dict(method='DeepLSD MegaDepth official quickstart', upstream=SOURCE, upstream_revision=REVISION,
                    native_detector_revision=PYTLSD, native_detector_binary_sha256=digest(pytlsd.__file__),
                    checkpoint_url=CHECKPOINT, checkpoint_sha256=WEIGHTS_SHA256, model_settings=CONF,
                    transform=dict(max_long_side=MAX_SIDE,resize='aspect preserving, INTER_AREA, rounded size',
                                   padding='none',coordinates='native source pixels, origin at upper-left image edge; model pixel (0,0) center is (+0.5,+0.5) before inverse resize'),
                    junctions=dict(algorithm='midpoints of nearby nonparallel segment endpoints',
                                   radius_native_px=JUNCTION_RADIUS_NATIVE, minimum_angle_degrees=MIN_CORNER_DEGREES),
                    inference_wrapper_revision=rev(ROOT), inference_wrapper_sha256=digest(__file__),
                    environment=dict(python=platform.python_version(), platform=platform.platform(), device=args.device,
                                     torch_threads=args.threads, torch=torch.__version__, numpy=np.__version__,
                                     opencv=cv2.__version__, scipy=version('scipy'), scikit_image=version('scikit-image'),
                                     omegaconf=version('omegaconf'), actions_run=os.environ.get('GITHUB_RUN_ID')),
                    inference_timing='resize + CNN + official PyTLSD extraction/filter + coordinate conversion and junctions; excludes loading, image decode and cache writing',
                    model_load_seconds=startup, inference_seconds=sum(times.values()), per_image_seconds=times,
                    fixture_ids={row['key']:row['sha256'] for row in rows}, segment_count=counts, files=files)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
