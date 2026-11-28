"""Replay inherited wall grids after splitting cells at observed same-facing depth boundaries."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import relative_face_assembly as solver
from benchmark.expanded_scores import load_reference
from benchmark.floating_expanded_run import score_export
from benchmark.floating_package_rechecks import partition_charts
from benchmark.floating_selective_replay import assemble_arm
from benchmark.floating_stage_report import read
from benchmark.floating_verified_wall_expansion import digest, save
from benchmark.mask_blocks import decoded_pack
from engine.geometry import project


def samples(chart, field):
    uv = np.asarray([[row['u'] + .5, row['v'] + .5] for row in chart['cells']])
    x, y = project(np.asarray(chart['H']), uv).astype(int).T
    assert np.all((x >= 0) & (x < field.shape[1]) & (y >= 0) & (y < field.shape[0]))
    return field[y, x]


def reference_parts(chart, frame):
    """Identify actual same-facing parent planes at source-cell centers in native pixels."""
    with np.load(frame / 'labels.npz') as archive:
        native = archive['face_id']
    faces = read(frame / 'labels.json')['faces']
    hit = Counter(samples(chart, native))
    sides = Counter()
    for face in faces:
        sides[face['direction']] += hit[face['id']]
    side, support = sides.most_common(1)[0]
    if support < .6 * len(chart['cells']) or side in ('up', 'down'):
        return None, {}, dict(side=side, support=int(support))
    levels = Counter()
    table = {}
    for face in faces:
        if face['direction'] == side:
            axis = np.flatnonzero(face['normal_world'])[0]
            level = face['xyz_world'][axis] * face['normal_world'][axis]
            levels[int(level)] += hit[face['id']]
            table[face['id']] = int(level)
    # A small native strip or single texture edge cannot trigger this intervention.
    chosen = [p for p, n in levels.most_common() if n >= max(24, .05 * len(chart['cells']))]
    detail = dict(side=side, support=int(support), plane_cell_counts=dict(levels), planes=chosen)
    if len(chosen) < 2:
        return None, {}, detail
    lut = np.zeros(native.max() + 1, np.int32)
    for fid, level in table.items():
        if level in chosen and fid < len(lut):
            lut[fid] = chosen.index(level) + 1
    return lut[native], {i+1: p for i, p in enumerate(chosen)}, detail


def source_parts(chart, planes, rgb, field_root, receipt, mode):
    """Retain a coarse cut only at a long signed depth break between parallel wall patches."""
    from benchmark import floating_depth_before_atlas as depth_stage
    from benchmark.floating_package_rechecks import native_mask
    from engine.normal_regions import _unit
    from engine.surface_fields import MODELS

    field_receipt = read(field_root / 'receipt.json')
    assert field_receipt['source_sha256'] == receipt['source_sha256']
    height, width = rgb.shape[:2]
    scale = min(1., 480 / max(height, width))
    size = (round(width * scale), round(height * scale))
    fields = {}
    for name in ('normal', 'depth'):
        field_path = field_root / (name + '.npz')
        assert field_receipt['model_sha256'][name] == MODELS[name][2]
        assert digest(field_path) == field_receipt['fields_sha256'][name] == receipt['field_sha256'][name]
        with np.load(field_path) as archive:
            fields[name] = cv2.resize(archive['value'].astype('f'), size)
    normal = _unit(fields['normal'])
    depth = fields['depth']
    image = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    plane = next(p for p in planes if p['id'] == chart['id'])
    mask = cv2.resize(native_mask(plane, (width, height)).astype('B'), size,
                      interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
    if mode == 'source-appearance':
        coarse = depth_stage.prior.partition(image, mask, depth).astype('uint16')
        meta = [dict(id=int(k), axis=0, normal_camera=_unit(np.median(normal[coarse == k], axis=0)))
                for k in np.unique(coarse[mask])]
    else:
        coarse, meta = depth_stage.partition(image, mask, normal, depth)
    native = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
    n_cells = Counter(samples(chart, native))
    # Distances and sampled boundary lengths below are in a 480-pixel-long image.
    main = sorted((row for row in meta if n_cells[row['id']] >= max(24, .08 * len(chart['cells']))),
                  key=lambda row: -n_cells[row['id']])
    detail = dict(region_counts={int(k): int(v) for k, v in n_cells.items()}, size=size, pairs=[])
    if len(main) < 2:
        return None, detail
    residual = depth_stage.signed_residual(depth, mask)
    noise = max(.012 * abs(float(np.median(depth[mask]))), 1.e-6)
    interiors = {}
    for row in main:
        region = coarse == row['id']
        inside = cv2.erode(region.astype('B'), np.ones((5, 5), 'B')).astype(bool)
        ring = inside & ~cv2.erode(inside.astype('B'), np.ones((3, 3), 'B')).astype(bool)
        pixels = np.column_stack(np.nonzero(ring))
        if len(pixels):
            dist, keys = cv2.distanceTransformWithLabels((~ring).astype('B'), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            lut = np.zeros((int(keys.max()) + 1, 2), int)
            lut[keys[ring]] = pixels
            interiors[row['id']] = pixels, dist, lut[keys]
    winner = None
    # Prefer the two largest cell-supported faces with an actual paired discontinuity.
    for i, first in enumerate(main):
        for second in main[i+1:]:
            a, b = first['id'], second['id']
            if a not in interiors or b not in interiors:
                continue
            agreement = float(np.dot(first['normal_camera'], second['normal_camera']))
            if agreement < .95 or first['axis'] != second['axis']:
                continue
            pixels = interiors[a][0]
            _, distances, nearest = interiors[b]
            distance = distances[tuple(pixels.T)]
            close = distance <= 18
            # Count unique contact runs, not all contacts to one corner or an isolated texture edge.
            p, q = pixels[close], nearest[tuple(pixels[close].T)]
            if len(p) < 16 or len(np.unique(q, axis=0)) < 12:
                continue
            delta = residual[tuple(p.T)] - residual[tuple(q.T)]
            normals_match = (normal[tuple(p.T)] * normal[tuple(q.T)]).sum(1) >= .94
            jump = float(np.median(delta))
            support = normals_match & (delta * np.sign(jump) >= 1.5 * noise)
            spread = len(np.unique(((p[support] + q[support]) // 12), axis=0))
            paired = dict(parts=[a, b], probes=len(p), distinct_boundary_bins=spread,
                          signed_jump=jump, noise=noise, fraction_supported=float(support.mean()),
                          normal_cosine=agreement)
            detail['pairs'].append(paired)
            if abs(jump) >= 2 * noise and support.mean() >= .7 and spread >= 5:
                if winner is None:
                    winner = second['id']
    if winner is None:
        return None, detail
    detail['isolated_region'] = winner
    # Keep all nonsplit cells together, including any whose center lands outside their native mask.
    return np.where(native == winner, 2, 1).astype('B'), detail


def parts(charts, target, ids):
    if ids is None:
        return charts, {}, {}
    output, groups = partition_charts(charts, {target['id']}, {target['id']: ids})
    counts = groups[target['id']]
    members = {f"{target['id']}~p{k}": target['id'] for k in counts}
    return output, members, counts


def chart_distance(chart, size):
    h = np.asarray(chart['H'])
    width, height = size
    directions = h[:2, :2] - np.outer([width / 2, height / 2], h[2, :2])
    u, v = directions.T
    a, b = h[2, :2]
    # Fit one squared focal length to the square metric's orthogonality and equal lengths.
    lhs = np.array([2*a*b, a*a-b*b])
    rhs = np.array([-2*u@v, v@v-u@u])
    focal2 = lhs @ rhs / max(lhs @ lhs, 1.e-20)
    info = dict(focal2=float(focal2))
    if not (.28*height)**2 <= focal2 <= (3*height)**2:
        return dict(info, reason='focal length outside usable range'), None
    inv_k = np.linalg.inv(np.array([[np.sqrt(focal2), 0., width/2], [0., np.sqrt(focal2), height/2], [0,0,1.]]))
    physical = inv_k @ h
    tangents = physical[:, :2]
    lengths = np.linalg.norm(tangents, axis=0)
    skew = abs(tangents[:,0] @ tangents[:,1]) / np.prod(lengths)
    unit = lengths.mean()
    normal = np.cross(tangents[:,0], tangents[:,1])
    normal /= np.linalg.norm(normal)
    facing = abs(normal @ np.array(chart['normal_camera']))
    distance = abs(normal @ physical[:,2]) / unit
    info.update(skew=float(skew), tangent_ratio=float(min(lengths)/max(lengths)),
                facing=float(facing), distance_blocks=float(distance))
    if skew > .25 or min(lengths)/max(lengths) < .75 or facing < .78:
        return dict(info, reason='square-grid metric or facing inconsistent'), None
    return dict(info, unit=float(unit)), float(distance)


def image_pitch_offsets(chart, groups, rgb):
    """Estimate separation from the two periods of the photographed projected block lattice."""
    h = np.asarray(chart['H'])
    info, distance = chart_distance(chart, rgb.shape[1::-1])
    if distance is None:
        return {}, info
    yy, xx = np.nonzero(groups)
    # Deterministic subsampling while preserving native image gradients.
    stride = max(1, len(xx) // 100000)
    xx, yy = xx[::stride], yy[::stride]
    xy = np.column_stack((xx+.5, yy+.5))
    uv = project(np.linalg.inv(h), xy)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype('f')
    smooth = cv2.GaussianBlur(grayscale, (3,3), .7)
    gradient = np.column_stack([cv2.Sobel(smooth,cv2.CV_32F,1,0)[yy,xx],
                                cv2.Sobel(smooth,cv2.CV_32F,0,1)[yy,xx]])
    du = project(h, uv+[.05,0])-xy
    dv = project(h, uv+[0,.05])-xy
    def directional(v):
        return np.abs((gradient*v).sum(1)/np.maximum(np.linalg.norm(v,axis=1),1.e-8))
    gu, gv = directional(du), directional(dv)
    source_counts = Counter(samples(chart, groups))
    keys = [int(k) for k,n in source_counts.most_common() if k and n>=24]
    periods = np.arange(.75,1.351,.005)
    estimates, rulers = {}, {}
    for k in keys:
        inside = cv2.erode((groups==k).astype('B'), np.ones((5,5),'B'))[yy,xx].astype(bool)
        axes = []
        for i,weight in enumerate((gu-.65*gv, gv-.65*gu)):
            if not inside.any():
                break
            accepted = inside & (weight>max(8.,np.quantile(weight[inside],.70)))
            q, w = uv[accepted,i], weight[accepted]
            if len(q)<60 or np.quantile(q,.90)-np.quantile(q,.10)<6:
                break
            sub = max(1,len(q)//3200)
            q, w = q[::sub], w[::sub]
            # Cap influence from a single isolated, very high-contrast border.
            w = np.minimum(w,np.quantile(w,.95))
            scores = abs(np.exp(2j*np.pi*q[:,None]/periods).T@w)/w.sum()
            ix = int(np.argmax(scores))
            elsewhere = abs(periods-periods[ix])>.055
            axes.append(dict(period=float(periods[ix]), strength=float(scores[ix]),
                             margin=float(scores[ix]-scores[elsewhere].max())))
        rulers[k] = axes
    info['rulers'] = rulers
    def usable(k):
        return len(rulers[k])==2 and all(v['strength']>=.09 and v['margin']>=.012 for v in rulers[k])
    if not keys or not usable(keys[0]):
        return {}, dict(info, reason='anchor lacks two periodic axes')
    anchor = np.array([v['period'] for v in rulers[keys[0]]])
    estimates[keys[0]]=0
    for k in keys[1:]:
        estimates[k]=0
        if not usable(k):
            continue
        ratio = np.array([v['period'] for v in rulers[k]])/anchor
        if abs(np.log(ratio[0]/ratio[1])) > .045:
            continue
        offset = distance*(1.-1./np.mean(ratio))
        if abs(offset)<=min(12,.35*distance):
            estimates[k]=int(np.rint(offset))
    return estimates, dict(info, estimates=estimates)


def image_depth_offsets(chart, groups, field_root, receipt, size):
    """Estimate integer separation from image depth calibrated by a source planar unit grid."""
    path = field_root / 'depth.npz'
    assert digest(path) == receipt['field_sha256']['depth']
    with np.load(path) as archive:
        field = cv2.resize(archive['value'].astype('f'), size)
    uv = np.array([[p['u'] + .5, p['v'] + .5, 1.] for p in chart['cells']])
    h = np.asarray(chart['H'])
    info, distance = chart_distance(chart, size)
    if distance is None:
        return {}, info
    unit = info['unit']
    plane_ids = samples(chart, groups)
    model = samples(chart, field)
    # Only positive substantial IDs receive cross-plane links. Zero is the reference residual.
    keys = [int(k) for k,n in Counter(plane_ids).most_common() if k and n >= 24]
    if len(keys) < 2:
        return {}, dict(info, reason='fewer than two supported regions')
    ideal_inverse_z = unit / (uv @ h[2])
    if np.any(ideal_inverse_z <= 0):
        return {}, dict(info, reason='invalid projected chart depth')
    anchor = plane_ids == keys[0]
    x, y = ideal_inverse_z[anchor], model[anchor]
    lo, hi = np.quantile(x, [.25,.75])
    low, high = x<=lo, x>=hi
    dx = np.median(x[high])-np.median(x[low])
    if abs(dx) < 1.e-8 or np.std(y) < 1.e-7:
        return {}, dict(info, reason='insufficient depth range')
    slope = float((np.median(y[high])-np.median(y[low]))/dx)
    correlation = float(np.corrcoef(x,y)[0,1])
    info.update(correlation=correlation, slope=slope)
    if slope <= 0 or correlation < .7:
        return {}, dict(info, reason='source field disagrees with anchor slope')
    intercept = float(np.median(y-slope*x))
    predicted = slope*ideal_inverse_z
    residual = model-intercept-predicted
    noise = max(float(np.median(abs(residual[anchor])))*1.4826, .005*abs(float(np.median(y))))
    info.update(anchor=keys[0], intercept=intercept, noise=noise)
    estimates, detail = {keys[0]:0}, {}
    for k in keys[1:]:
        selected = plane_ids == k
        change = np.median(residual[selected])
        ratio = np.median((model[selected]-intercept)/predicted[selected])
        offset = distance*(1.-1./ratio) if ratio > 0 else 0.
        estimable = ratio > 0 and abs(change)>=2*noise and abs(offset)<=min(12, .35*distance)
        estimates[k] = int(np.rint(offset)) if estimable else 0
        detail[k] = dict(relative_scale=float(ratio), signed_offset=float(offset),
                         residual=float(change), applied=estimates[k])
    info['estimates']=detail
    return estimates, info


def appearance_parts(chart, planes, rgb, field_root, receipt):
    from benchmark import floating_mixed_wall_refit as appearance
    from benchmark.floating_package_rechecks import native_mask

    height, width = rgb.shape[:2]
    scale = min(1., 480 / max(height,width))
    size = (round(width*scale), round(height*scale))
    plane = next(p for p in planes if p['id'] == chart['id'])
    mask = cv2.resize(native_mask(plane, (width,height)).astype('B'), size, interpolation=cv2.INTER_NEAREST_EXACT)>0
    field_path = field_root/'depth.npz'
    assert digest(field_path) == receipt['field_sha256']['depth']
    with np.load(field_path) as archive:
        depth = cv2.resize(archive['value'].astype('f'), size)
    region = appearance.partition(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), mask, depth)
    ids = cv2.resize(region.astype('uint16'), (width,height), interpolation=cv2.INTER_NEAREST_EXACT)
    counts = Counter(samples(chart, ids))
    chosen = [int(k) for k,n in counts.most_common() if k and n >= max(24, .08*len(chart['cells']))]
    if len(chosen)<2:
        return None, {}
    # Merge small/unassigned pieces with the incumbent anchor. Every source cell is retained.
    lut = np.full(int(ids.max())+1, chosen[0], dtype='uint16')
    lut[chosen] = chosen
    return lut[ids], dict(major=chosen)


def offset_links(parent, levels):
    """Supply signed integer separation only; keep incoming tangent origins at zero."""
    propose = solver.propose_joins

    def proposal(charts, *args, **kwargs):
        joins, vetoes = propose(charts, *args, **kwargs)
        eligible = [(i, levels[int(c['id'].rsplit('~p', 1)[1])], c)
                    for i, c in enumerate(charts) if c['id'] in parent
                    and int(c['id'].rsplit('~p', 1)[1]) in levels]
        if eligible:
            i, anchor, a = eligible[0]
            reference = []
            for j, level, b in eligible[1:]:
                if a['normal_axis'] != b['normal_axis'] or a['axes'] != b['axes']:
                    continue
                shift = solver.axis_vector(a['normal_axis']) * (level - anchor)
                reference.append(dict(i=i, j=j, shift=shift.tolist(), kind='reference-depth-only',
                                      edges=0, error=0, image_edges=[]))
            joins = reference + joins
        return joins, vetoes
    return patch.object(solver, 'propose_joins', proposal)


def replay(job):
    base, output, case, mode, fields = job
    cv2.setNumThreads(1)
    marker = case['id']
    dest = output / marker
    dest.mkdir(parents=True, exist_ok=True)
    record_path = dest / (mode + '.json')
    if record_path.exists():
        return read(record_path)
    original = base / 'original' / marker
    frame = Path(case['frame'])
    receipt = read(original / 'case.json')
    assert digest(original / 'charts.json') == receipt['charts_sha256']
    assert digest(original / 'reconstruction.zip') == receipt['score']['pack_sha256']
    assert digest(frame / 'rgb.png') == receipt['source_sha256'] == case['rgb_sha256']
    packet = read(original / 'charts.json')
    image = np.asarray(Image.open(frame / 'rgb.png').convert('RGB'))
    planes = read(original / 'source-handoff/surfaces.json')['planes']
    charts = packet['charts']
    # Stable, source-only candidate eligibility. A previously split grid keeps its existing handling.
    eligible = [c for c in charts if '~p' not in c['id'] and len(c['cells']) >= 256
                and c.get('normal_camera') and abs(c['normal_camera'][1]) < .67]
    target = max(eligible, key=lambda c: len(c['cells']), default=None)
    record = dict(id=marker, split=case['split'], rgb_sha256=case['rgb_sha256'],
                  dataset_manifest_sha256=case['dataset_manifest_sha256'],
                  baseline_chart_sha256=receipt['charts_sha256'], baseline_score=receipt['score'],
                  target=target['id'] if target else None, arms={})
    reference = None
    groups, levels, detail = None, {}, {}
    if target is not None:
        if mode.startswith('reference'):
            groups, levels, detail = reference_parts(target, frame)
        elif mode in ('source-image-offset', 'source-pitch-offset'):
            groups, detail = appearance_parts(target, planes, image, fields / marker / 'fields', receipt)
        else:
            groups, detail = source_parts(target, planes, image, fields / marker / 'fields', receipt, mode)
    image_levels = {}
    if groups is not None and mode in ('source-image-offset', 'reference-image-offset', 'source-pitch-offset', 'reference-pitch-offset'):
        if 'pitch' in mode:
            from benchmark.floating_package_rechecks import native_mask
            plane = next(p for p in planes if p['id'] == target['id'])
            support = native_mask(plane, image.shape[1::-1])
            image_levels, calibration = image_pitch_offsets(target, np.where(support, groups, 0), image)
        else:
            image_levels, calibration = image_depth_offsets(target, groups, fields/marker/'fields', receipt, image.shape[1::-1])
        detail['image_calibration'] = calibration
        if mode.startswith('reference-'):
            # Fail back to a zero-depth connection for every substantial native sibling.
            image_levels = {key: image_levels.get(key, 0) for key in levels}
        elif not any(image_levels.values()):
            groups = None
    proposed, members, counts = parts(charts, target, groups) if target else (charts, {}, {})
    record.update(details=detail, partition_cells=counts)
    variants = [('identity', charts, {}, {})] if mode == 'reference' else []
    offsets = image_levels
    if mode == 'reference-zero':
        offsets = dict.fromkeys(levels, 0)
    elif mode == 'reference-offset-trace':
        offsets = levels
    variants.append((mode, proposed, members, offsets))
    if mode == 'reference' and levels:
        variants.append(('reference-offset', proposed, members, levels))
    for arm, values, family, offsets in variants:
        if arm != 'identity' and not counts:
            record['arms'][arm] = dict(score=receipt['score'], unchanged=True, paired=dict(
                correct_gained=0, correct_lost=0, extras_added=0, extras_removed=0))
            continue
        if reference is None:
            reference = load_reference(case)
        location = dest / arm
        attempt = 1
        while location.exists():
            attempt += 1
            location = dest / f'{arm}-attempt-{attempt}'
        with offset_links(family, offsets) if offsets else nullcontext():
            assembled = assemble_arm(dict(packet, charts=values), planes, image, family, 2.5, location, save_trace=mode == 'reference-offset-trace')
        pack = location / 'reconstruction.zip'
        score, aligned = score_export(case, pack, assembled, reference=reference)
        before = {tuple(row[:3]): row[3] for row in read(original / 'aligned-placements.json.gz')}
        if arm == 'identity':
            assert decoded_pack(pack) == decoded_pack(original / 'reconstruction.zip')
            assert score['geometry'] == receipt['score']['geometry'] and score['scores'] == receipt['score']['scores']
            assert aligned == before
        added, lost = aligned.keys() - before.keys(), before.keys() - aligned.keys()
        paired = dict(correct_gained=len(added & reference.truth.keys()), correct_lost=len(lost & reference.truth.keys()),
                      extras_added=len(added - reference.truth.keys()), extras_removed=len(lost - reference.truth.keys()))
        record['arms'][arm] = dict(score=score, paired=paired, pack_directory=location.name, selected=assembled['selected'],
                                   joins=len(assembled['joins']), unchanged=False)
    save(record_path, record)
    print(marker, mode, [(key, val['score']['geometry']['strict']['correct'], val['score']['geometry']['strict']['extras'])
                         for key, val in record['arms'].items()], flush=True)
    return record


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fields', type=Path, required=True)
    parser.add_argument('--mode', choices=['source', 'source-appearance', 'source-image-offset', 'reference', 'reference-zero', 'reference-image-offset', 'reference-pitch-offset', 'source-pitch-offset', 'reference-offset-trace'], required=True)
    parser.add_argument('--case', nargs='*')
    parser.add_argument('--workers', type=int, default=2)
    options = parser.parse_args()
    cases = read(options.base / 'fixtures.json')['cases']
    if options.case:
        cases = [c for c in cases if c['id'] in options.case or c['id'].split('-')[1] in options.case]
        assert len(cases) == len(options.case)
    jobs = [(options.base, options.output, c, options.mode, options.fields) for c in cases]
    with ProcessPoolExecutor(options.workers) as pool:
        list(pool.map(replay, jobs))


if __name__ == '__main__':
    main()
