"""Require camera-conditioned relative depth evidence before moving RGB wall components."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

from benchmark import wall_color_partition as walls
from benchmark.expanded_baseline import SOURCE, frames, read, digest, save, WORKTREES, ROOT
from engine import surface_fields

OUTPUT = ROOT / 'runs/camera-conditioned-wall-separation-01'
CASES = ROOT / 'benchmark/reviews/camera-native-recovery-controls-01/independent-default-cases.json.gz'
BASE = WORKTREES / 'camera-native-floor-inverse/runs/camera-native-floor-inverse-01/builds'

def interior(mask):
    return np.asarray(Image.fromarray(mask.astype('uint8')).filter(ImageFilter.MinFilter(3))) > 0

def gate_for(field):
    """Compare eroded child and nearby parent interiors with their camera-predicted slope removed."""
    sampled = None
    def gate(image, parent, child, ray, eye, initial, candidate):
        nonlocal sampled
        if sampled is None:
            depth = field() if callable(field) else field
            sampled = np.asarray(Image.fromarray(depth.astype('float32')).resize(
                image.size, Image.Resampling.BILINEAR))[2::4, 2::4]
        d = sampled[:parent.shape[0], :parent.shape[1]]
        inside = interior(child)
        near = np.asarray(Image.fromarray(child.astype('uint8')).filter(ImageFilter.MaxFilter(21))) > 0
        outside = interior(parent & ~child) & near
        evidence = dict(proposed=int(candidate), interior=int(inside.sum()), neighbor=int(outside.sum()))
        if min(inside.sum(), outside.sum()) < 16:
            return initial, evidence
        # A parallel plane's inverse depth varies linearly with its camera-axis ray coefficient.
        # Estimate field slope only on nearby parent pixels, robustly trimming residual outliers.
        x = ray[outside] / (initial-eye)
        fit = np.ones(len(x), bool)
        if np.ptp(x) < 1e-6:
            return initial, evidence
        for _ in range(3):
            matrix = np.stack([x[fit], np.ones(fit.sum())], axis=1)
            a,b = np.linalg.lstsq(matrix, d[outside][fit], rcond=None)[0]
            errors = np.abs(d[outside] - (a*x+b))
            fit = errors <= np.percentile(errors, 80)
        if a <= 0 or not np.isfinite(a):
            return initial, evidence
        prev = a * ray[inside] / (initial-eye) + b
        nxt = a * ray[inside] / (candidate-eye) + b
        offset = float(np.median(d[inside]-prev))
        move = float(np.median(nxt-prev))
        noise = max(float(np.median(errors))*1.4826,
                    float(np.percentile(d[parent],90)-np.percentile(d[parent],10))*.025, 1e-5)
        gain = float(np.median(abs(d[inside]-prev))-np.median(abs(d[inside]-nxt)))
        evidence.update(offset=offset, predicted_change=move, noise=noise, residual_gain=gain)
        accept = offset*move > 0 and abs(offset) > 2*noise and gain > noise
        return (candidate if accept else initial), evidence
    return gate

def job(args):
    entry, case, mode = args
    key = entry['id']
    install = walls.install_trial
    original_folder = walls.baseline_folder
    if 'frame' in entry:
        inputs = walls.BASE / entry['dataset'] / key
        depth_path = inputs/'handoffs/depth.npz'
        with np.load(depth_path) as data:
            field = data['value']
        path = BASE/entry['dataset']/key
        if not (path/'reconstruction.zip').is_file() or digest(path/'reconstruction.zip') != case['pack_sha256']:
            path = original_folder(entry,case)
    else:
        def field():
            from PIL import Image
            with Image.open(entry['rgb']) as im:
                return surface_fields.infer(im, 'depth')
        path = None
    def trial(_, audit):
        return install('control' if mode=='control' else 'ranked', audit,
                       gate_for(field) if mode=='depth' else None)
    # Win32 requires an extended path for the long fixed Colonnade frame ID.
    run_root = Path(chr(92)*2 + '?' + chr(92) + str(OUTPUT)) if os.name == 'nt' else OUTPUT
    with patch.object(walls,'install_trial',trial), patch.object(walls,'baseline_folder',return_value=path), patch.object(walls,'REVIEW',OUTPUT/'scores'), patch.object(walls,'ROOT',run_root):
        walls.run((entry, case, mode))
    scorefile = OUTPUT/'scores'/f'{key}-{mode}.json'
    row=read(scorefile)
    row['baseline_revision']='e3bd220'
    row['depth_source'] = digest(depth_path) if 'frame' in entry else surface_fields.MODELS['depth'][2]
    save(scorefile,row)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ids',nargs='+')
    parser.add_argument('--mode', choices=['control','depth','ranked'], default='depth')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    entries=list(frames(SOURCE,['visible-builds-v3','decorative-builds-v1']))
    if set(args.ids)-{entry['id'] for entry in entries}:
        from benchmark.expanded_fixed_handoffs import fixed_frames
        entries.extend(fixed_frames())
    cases={row['id']:row for row in read(CASES)}
    jobs=[(entry,cases[entry['id']],args.mode) for entry in entries if entry['id'] in args.ids]
    assert len(jobs)==len(args.ids)
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(job,jobs))

if __name__=='__main__':
    main()
