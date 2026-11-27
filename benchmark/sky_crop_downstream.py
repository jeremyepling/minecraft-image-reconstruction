"""Replay one proposed crop packet through floating counts, joins and block export."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageFilter

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save
from engine import floating_sky
from engine.plane_consensus import row_runs, with_mask


def apply_crops(image, proposals, replacements, keep_quad):
    result = []
    if replacements.keys() - {p["id"] for p in proposals}:
        raise ValueError("Crop packet contains unrecognized or rejected proposal IDs")
    for part in proposals:
        support = replacements.get(part["id"])
        if support is None:
            result.append(part)
            continue
        if list(image.size) != support['size']:
            raise ValueError("Expected a native source-image mask")
        pixels = np.zeros(image.size[::-1], np.uint8)
        for y, left, right in support['runs']:
            pixels[y,left:right] = 1
        size = tuple(part['support']['size'])
        mask = np.asarray(Image.fromarray(pixels).resize(size,Image.Resampling.NEAREST)) != 0
        if mask.any():
            original = np.zeros(size[::-1], bool)
            for y, left, right in part['support']['runs']:
                original[y,left:right] = True
            if np.array_equal(mask, original):
                result.append(part)
                continue
            if keep_quad:
                result.append(dict(part, support=dict(size=list(size),runs=row_runs(mask)),
                                   support_area=int(mask.sum())))
                continue
            gray = np.asarray(image.resize(size, Image.Resampling.BOX).convert('L').filter(
                ImageFilter.GaussianBlur(.9)), dtype=np.float32)[...,None]
            result.append(with_mask(part,mask,size,gray,np.array(image.size)/size))
    return result


def reference_proposals(image, parts, case, targets, cache):
    """Diagnostic supplying all visible full-cube planes; normals and depth still come from image models."""
    from engine.normal_regions import _field
    from engine.mask_first import _image_depth
    frame = Path(case['frame'])
    labels = frame/'labels.npz'
    if digest(labels) != case['reference_sha256']['labels.npz']:
        raise ValueError(f"Invalid reference: {labels}")
    with np.load(labels) as ref:
        owner = ref['face_id']
    size = tuple(parts[0]['support']['size'])
    # Reuse the ordinary support grid and pixel-center convention.
    xs = ((2*np.arange(size[0])+1)*image.width)//(2*size[0])
    ys = ((2*np.arange(size[1])+1)*image.height)//(2*size[1])
    owner = owner[np.ix_(ys,xs)]
    with np.load(cache/'normal.npz') as file:
        normal = _field(file['value'],size)
    with np.load(cache/'depth.npz') as file:
        depth = _image_depth(file['value'],size)
    gray = np.asarray(image.convert('L').resize(size,Image.Resampling.BOX).filter(
        ImageFilter.GaussianBlur(.9)),dtype=np.float32)[...,None]
    result=[]
    for target in targets:
        mask=np.isin(owner,target['face_ids'])
        if not mask.any():
            continue
        direction=np.median(normal[mask],axis=0)
        direction /= max(float(np.linalg.norm(direction)),1e-9)
        seed=dict(id=target['id'],normal_camera=direction.tolist(),
                  relative_inverse_depth=float(np.median(depth[mask])))
        result.append(with_mask(seed,mask,size,gray,np.array(image.size)/size))
    return sorted(result,key=lambda p:-p['support_area'])


def replay(job):
    arm, case, replacements, packet_hash, keep_quad, all_targets = job
    # Bind stage paths before importing downstream modules in this worker.
    from benchmark import floating_cloud_replay as paths
    root = ROOT/'runs/sky-crop-downstream'/arm
    paths.RUN = root
    paths.SHARED_ROWS = root/'shared-corner-rows'
    paths.NATIVE_COUNTER = root/'surface-crop-first'
    paths.ANCHOR_EXTENSION = root/'garden-anchor-extension'
    paths.ASSEMBLY = root/'relative-face-assembly'
    from benchmark import floating_build_local_run as build, shared_corner_rows as shared
    from benchmark import native_surface_counter as counter, native_corner_registration as registration
    from benchmark import garden_anchor_extension as atlas, relative_face_round as assembly
    from benchmark.mask_blocks import decoded_pack
    cv2.setNumThreads(1)
    spec = case['variants']['sky']
    name = case['id']
    source = Path(spec['image'])
    if digest(source) != spec['source_sha256']:
        raise ValueError("Unexpected source")
    handoff = paths.source_handoff(name)
    cache = SOURCE/'floating-build-grid-01/baseline/sky'/case['dataset']/name/'handoffs'
    original = floating_sky.filter_proposals

    def replace_after_sky(image, parts):
        kept, stats = original(image, parts)
        if all_targets:
            return reference_proposals(image, kept, case, replacements, cache), stats
        return apply_crops(image, kept, replacements, keep_quad), stats

    receipt = handoff/'build-manifest.json'
    if receipt.exists() and (read(receipt).get('crop_packet_sha256') != packet_hash or bool(read(receipt).get('keep_sampling_quad')) != keep_quad):
        raise ValueError(f"Choose a new arm name for changed crop packets: {root}")
    with patch.object(floating_sky,'filter_proposals',replace_after_sky):
        build.build_handoff(source, cache, handoff)
    save(receipt, read(receipt) | dict(crop_packet_sha256=packet_hash,keep_sampling_quad=keep_quad))
    shared.predict(case)
    counter.predict(case)
    registration.predict(case)
    atlas.all_walls((case,read(atlas.baseline_file(name))))
    assembly.evaluate((case,'sky','connected-atlas'))
    folder = paths.ASSEMBLY/'connected-atlas/sky'/name
    base = WORKTREES/'floating-visible-builds/runs/floating-cloud-gate-01'
    old = base/'relative-face-assembly/connected-atlas/sky'/name
    same = decoded_pack(folder/'reconstruction.zip') == decoded_pack(old/'reconstruction.zip')
    if not replacements and not same:
        raise ValueError(f'No-op control changed decoded export on {name}')
    counts = read(handoff/'counts.json')['surfaces']
    previous = {r['id']:r.get('counts') for r in read(base/'sky'/name/'source-handoff/counts.json')['surfaces']}
    return dict(id=name,dataset=case['dataset'],split=case['split'],same_export=same,
                changed_counts=sum(c.get('counts')!=previous.get(c['id']) for c in counts),
                removed_count_rows=len(previous.keys()-{c['id'] for c in counts}),
                score=read(folder/'score.json'),baseline=read(old/'score.json'))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('arm')
    cli.add_argument('--packet', type=Path)
    cli.add_argument('--all-targets',type=Path,help='Diagnostic using every visible full-cube target from the reference crop score report.')
    cli.add_argument('--keep-quad',action='store_true',help='Change support alone and keep the original sampling chart.')
    cli.add_argument('--workers', type=int, default=3)
    args=cli.parse_args()
    if Path(args.arm).name!=args.arm or args.arm in ('.','..'):
        cli.error("arm must be a directory name")
    cases=read(SOURCE/'floating-build-grid-01/fixtures.json')['cases']
    packet=read(args.packet)['frames'] if args.packet else {c['id']:{} for c in cases}
    if set(packet)!={c['id'] for c in cases}:
        cli.error('expected one entry for every source view')
    if args.all_targets:
        if args.packet or args.keep_quad:
            cli.error('All-target diagnostic creates new proposals and charts.')
        report=read(args.all_targets)
        packet={c['id']:c['arms']['baseline']['targets'] for c in report['cases']}
    input_file=args.all_targets or args.packet
    fingerprint=digest(input_file) if input_file else 'noop'
    jobs=[(args.arm,c,packet[c['id']],fingerprint,args.keep_quad,bool(args.all_targets)) for c in cases]
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(replay,jobs))
    save(ROOT/'runs/sky-crop-downstream'/args.arm/'summary.json.gz',rows)
