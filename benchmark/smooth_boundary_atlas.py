"""Test whether a shaded vanilla atlas distinguishes supported smooth mask growth from pavement."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
import importlib.util
import inspect
from pathlib import Path

import numpy as np

from benchmark import facade_inset_round as suite
from benchmark.shared_face_error_budget import current_stages, LATEST
from benchmark.texture_camera_eval import read, save
from benchmark.wall_texture_evaluate import metrics
from engine import atlas_count
from engine.pillar_seams import native_mask


@lru_cache(maxsize=1)
def catalog():
    atlas = atlas_count.load_atlas(suite.ATLAS, '26.3')
    # A class is observable even when several smooth block IDs have similar colors.
    return atlas, atlas.spreads.max(axis=1) <= 3


@lru_cache(maxsize=1)
def source_module(path):
    spec = importlib.util.spec_from_file_location('frozen_flat_extent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def material(pixels):
    atlas, smooth = catalog()
    if len(pixels) < 12:
        return None
    pixels = pixels[::max(1, len(pixels)//512)]
    _, loss = atlas_count._shortlist(pixels, np.ones(len(pixels), bool), atlas)
    si = np.flatnonzero(smooth)[np.argmin(loss[smooth])]
    ti = np.flatnonzero(~smooth)[np.argmin(loss[~smooth])]
    return dict(smooth=atlas.labels[si], textured=atlas.labels[ti],
                margin=float(loss[ti]-loss[si]), median=np.median(pixels, axis=0).tolist())


def host_witness(rgb, host, event, target):
    # Use a full native window on the sponsor. Its pixels have never passed the
    # proposal's color test; having a compatible median doesn't censor its texture.
    x0, y0, x1, y1 = event['box']
    h, w = host.shape
    left, top = max(0, x0-20), max(0, y0-20)
    right, bottom = min(w, x1+20), min(h, y1+20)
    witnesses = []
    for y in range(top, bottom-7, 6):
        for x in range(left, right-7, 6):
            seen = host[y:y+8, x:x+8]
            if seen.sum() < 48:
                continue
            pixels = rgb[y:y+8, x:x+8][seen]
            error = abs(np.median(pixels, axis=0)-target).max()
            witnesses.append((float(error), x, y, pixels))
    if not witnesses:
        return None
    error, x, y, sample = min(witnesses, key=lambda p:p[:3])
    return dict(**material(sample), error=error, source_xy=[x, y])


def boundary_witness(rgb, event, target):
    # The margin is observed in the same image rows as the proposal. Retain fine
    # signed RGB differences that disappear after chromaticity normalization.
    _, y0, _, y1 = event['box']
    height, width, _ = rgb.shape
    best = None
    for y in range(max(0,y0-3), min(height-7,y1), 4):
        for x in (0,8,16,width-24,width-16,width-8):
            pixels = rgb[y:y+8,x:x+8].reshape(-1,3)
            error = float(abs(np.median(pixels, axis=0)-target).max())
            if best is None or error < best[0]:
                best = error, x, y, pixels
    if best is None:
        return None
    error, x, y, sample = best
    return dict(**material(sample), error=error, source_xy=[x,y])


def evaluate(job):
    packing, expected, path, cached = job
    split, frame, _, _ = packing
    key = frame['id']
    result = dict(id=key, rgb_sha256=frame['files']['rgb.png'], split=split,
                  cohort='fixed' if split in suite.framed.EVALUATION else 'transfer', events={})
    for mode in cached:
        assert cached[mode]['source'] == frame['files']['rgb.png']
        result['events'][mode] = []
    if not any(any('both_axes' not in e for e in r['events']) for r in cached.values()):
        return result
    image, parts, _, _ = current_stages(packing, expected)
    rgb = np.asarray(image, dtype=np.int16)
    hosts = {p['id']:p for p in parts}
    proposals = source_module(path).proposals
    image_evidence = {}
    for mode, row in cached.items():
        events = proposals(image, parts, row_veto=mode=='row')
        frozen = [e for e in row['events'] if 'both_axes' not in e]
        assert [(e['surface'],e['box'],e['pixels']) for e in events] == [(e['surface'],e['box'],e['pixels']) for e in frozen], (key,mode)
        for event in events:
            pixels = np.concatenate([rgb[y,a:b] for y,a,b in event['runs']])
            proposal = material(pixels)
            target = np.array(proposal['median'])
            sponsor = host_witness(rgb,native_mask(hosts[event['surface']],image.size),event,target)
            background = boundary_witness(rgb,event,target)
            # Require both independently sampled patches to prefer smooth atlas
            # sprites to a textured competitor, allowing the fitted face shade.
            atlas_gate = bool(sponsor and sponsor['error']<=3 and min(proposal['margin'],sponsor['margin'])>=1.0)
            good_ground = bool(background and background['error']<=3 and background['margin']>=1.0)
            image_evidence[(mode,event['surface'],tuple(event['box']))] = (event, dict(
                surface=event['surface'],box=event['box'],pixels=event['pixels'],proposal=proposal,sponsor=sponsor,
                margin_sample=background,accepted_atlas=atlas_gate,accepted_atlas_ground=atlas_gate and not good_ground))
    # Grading only: reference IDs never enter candidate filtering above.
    truth = suite.visible_map(split,frame)
    by_id = {r['id']:r for r in metrics(parts, truth, image.size)['parts']}
    for (mode, surface, _), (e, record) in image_evidence.items():
        sid = by_id[surface]['dominant']
        ts = np.concatenate([truth[y,a:b] for y,a,b in e['runs']])
        record.update(on_host=int((ts==sid).sum()),background=int((ts==0).sum()),neighbor=int(((ts!=sid)&(ts!=0)).sum()))
        if key=='legacy-light_gray_concrete-holdout-registry-plaster-high-right':
            record['roi_pixels'] = int((ts==10).sum())
        if key=='legacy-light_gray_concrete-dev-shopfront-plaster-low-left':
            record['roi_pixels'] = sum(max(0,min(b,850)-max(a,790)) for y,a,b in e['runs'] if 490<=y<545)
        result['events'][mode].append(record)
    return result


def counts(rows, mode, rule):
    total = Counter()
    for row in rows:
        total['views'] += 1
        accepted = [e for e in row['events'][mode] if rule is None or e[rule]]
        total['affected_views'] += int(bool(accepted))
        total['events'] += len(accepted)
        total['abstentions'] += len(row['events'][mode])-len(accepted)
        for e in accepted:
            for k in ('pixels','on_host','background','neighbor','roi_pixels'):
                total[k] += e.get(k,0)
    return dict(total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-checkout',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    src = (args.proposal_checkout/'engine/flat_surface_extent.py').resolve()
    prior = args.proposal_checkout/'benchmark/reviews/shopfront-outline-gaps-01'
    expected = {r['id']:r for r in read(LATEST)['rows']}
    manifest = {}
    caches = {}
    for mode, folders in [('raw',('.','transfer')),('row',('row-fixed','row-transfer'))]:
        data = []
        for folder in folders:
            p = prior/folder/'cases.json'
            manifest[f'{mode}/{folder}'] = sha256(p.read_bytes()).hexdigest()
            data.extend(read(p))
        caches[mode] = {r['id']:r for r in data}
    jobs = [(j,expected[j[1]['id']],str(src),{mode:r[j[1]['id']] for mode,r in caches.items()}) for j in suite.jobs([*suite.framed.SIZES,'shelter'])]
    assert all({j[0][1]['id'] for j in jobs} == set(r) for r in caches.values()) and len(jobs)==204
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(evaluate, jobs))
    cols = {}
    for cohort in ('fixed','transfer','shopfront','registry','portico','courtyard'):
        subset = [r for r in rows if r['cohort']==cohort or (r['cohort']=='fixed' and cohort in r['id'])]
        cols[cohort] = {f'{mode}/{name}':counts(subset, mode, rule) for mode in ('raw','row')
                         for name,rule in [('proposals',None),('atlas','accepted_atlas'),('atlas_ground','accepted_atlas_ground')]}
    atlas, smooth = catalog()
    args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'summary.json', dict(baseline='a95816d', stage='Surface Finder', placement_rerun=False,
         proposal_sha256=sha256(src.read_bytes()).hexdigest(), source_manifests=manifest, atlas=atlas.provenance,
         smooth_sprites=int(smooth.sum()), results=cols))
    save(args.out/'cases.json',rows)


if __name__=='__main__':
    main()

