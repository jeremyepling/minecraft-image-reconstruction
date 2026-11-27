"""Infer wall lattice scale, phase and material on masked source pixels."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, WORKTREES, read, save, digest
from benchmark.full_block_lattice import atlas
from benchmark.native_surface_counter import HANDOFF, ordinary, project, unpack
from engine.atlas_count import _shortlist

OUTPUT = ROOT / 'runs/garden-anchor-extension-01'
REGISTERED = Path('runs/surface-crop-first-01/registered-seam/sky')


def baseline_file(name):
    relative = REGISTERED / name / 'charts.json'
    local = ROOT / relative
    return local if local.is_file() else WORKTREES / 'floating-visible-builds' / relative


def line_segments(rgb):
    detected = cv2.createLineSegmentDetector().detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))[0]
    return np.empty((0, 2, 2)) if detected is None else detected[:, 0].reshape(-1, 2, 2)


def pencils(rgb, segments):
    """Robust vanishing pencils from long high-contrast segments in the full screenshot."""
    if len(segments) < 9:
        return None
    lines = segments
    center = np.array(rgb.shape[1::-1]) / 2
    lines = (lines - center) / rgb.shape[1]
    delta = lines[:, 1] - lines[:, 0]
    length = np.linalg.norm(delta, axis=1)
    order = np.argsort(-length)[:400]
    mid = lines[order].mean(1)
    length = length[order]
    tangent = delta[order] / length[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    eq = np.column_stack((normal, -(normal * mid).sum(1)))
    a, b = np.triu_indices(min(len(eq), 100), 1)
    vp = np.cross(eq[a], eq[b])
    vp /= np.maximum(1e-08, np.linalg.norm(vp, axis=1))[:, None]
    def sinerr(q):
        ray = q[:, :2, None] - q[:, 2, None, None] * mid.T
        return np.abs(q @ eq.T) / np.maximum(0.0001, np.linalg.norm(ray, axis=1))
    err = sinerr(vp)
    weights = np.minimum(length, 0.15)
    result = []
    for i in range(3):
        score = np.maximum(0, 1 - (err / 0.008) ** 2) @ weights
        v = vp[score.argmax()]
        for _ in range(5):
            ray = v[:2] - mid * v[2]
            w = np.maximum(0, 1 - (sinerr(v[None])[0] / 0.008) ** 2) * weights
            e = eq * (np.sqrt(w) / np.maximum(0.01, np.linalg.norm(ray, axis=1)))[:, None]
            v = np.linalg.svd(e, full_matrices=False)[2][-1]
        weights[sinerr(v[None])[0] < 0.018] = 0
        result.append(np.r_[v[:2] * rgb.shape[1] + center * v[2], v[2]])
    result = np.array(result)
    return result if np.all(np.abs(result[:, 2]) > 1e-10) else None


def inputs():
    for case in read(ordinary.SHARED / 'fixtures.json')['cases']:
        base = baseline_file(case['id'])
        packet = read(base)
        if packet['source_sha256'] != digest(Path(case['variants']['sky']['image'])):
            raise ValueError(f"Source image differs from registered chart input for {case['id']}")
        yield (case, packet)


def rectify(rgb, mask, vp, edges):
    if vp is None:
        return None
    ys, xs = np.where(mask)
    origin = np.array([np.median(xs), np.median(ys)]) + 0.5
    pts = vp[:, :2] / vp[:, 2:]
    center = np.array(rgb.shape[1::-1]) / 2
    vertical = np.argmax(abs(pts[:, 1] - center[1]) / (abs(pts[:, 0] - center[0]) + 1))
    others = [i for i in range(3) if i != vertical]
    # Use the locally supported horizontal pencil; downweight axis-aligned image borders.
    mid = edges.mean(1)
    tang = edges[:, 1] - edges[:, 0]
    length = np.linalg.norm(tang, axis=1)
    tang /= length[:, None]
    eligible = (cv2.dilate(mask, np.ones((7, 7), np.uint8))[mid[:, 1].astype(int), mid[:, 0].astype(int)] > 0) & (length > 12)
    def vote(i):
        ray = pts[i] - mid
        sine = np.abs(ray[:, 0] * tang[:, 1] - ray[:, 1] * tang[:, 0]) / np.linalg.norm(ray, axis=1)
        return np.minimum(length, 150) @ (eligible * np.maximum(0, 1 - (sine / 0.025) ** 2))
    horizontal = max(others, key=vote)
    pair = pts[[horizontal, vertical]]
    focal = np.sqrt(max(1, -((pair[0] - center) @ (pair[1] - center))))
    if not 0.28 * rgb.shape[0] < focal < 3 * rgb.shape[0]:
        return None
    rays = np.column_stack(((pair - center) / focal, np.ones(2)))
    # Scale both axes identically in world units, preserving square-face aspect.
    vectors = (pair - origin) / np.linalg.norm(rays, axis=1)[:, None]
    sign = np.sign(np.diag(vectors))
    scale = np.linalg.norm(vectors[1])
    H = np.column_stack([*(np.r_[p, 1] * s / scale / np.linalg.norm(r) for p, r, s in zip(pair, rays, sign)), np.r_[origin, 1]])
    if abs(np.linalg.det(H)) < 1e-09:
        return None
    samples = np.column_stack((xs, ys)) + 0.5
    uv = project(np.linalg.inv(H), samples)
    if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 32760:
        return None
    low = np.floor(uv.min(0)).astype(int) - 2
    high = np.ceil(uv.max(0)).astype(int) + 2
    if np.prod(high - low) > 3000000.0 or max(high - low) >= 32767:
        return None
    H = H @ np.array([[1, 0, low[0]], [0, 1, low[1]], [0, 0, 1.0]])
    u, v = np.meshgrid(np.arange(high[0] - low[0]) + 0.5, np.arange(high[1] - low[1]) + 0.5)
    xy = (project(H, np.stack((u, v), axis=-1)) - 0.5).astype('f')
    image = cv2.remap(rgb, xy, None, cv2.INTER_LINEAR)
    support = cv2.remap(mask, xy, None, cv2.INTER_NEAREST)
    return (H, image, support)


def reviewed_garden_planes(case, rgb):
    """Only the targeted ablations use Jeremy’s three reviewed surface ranks."""
    planes = sorted(read(HANDOFF / case['id'] / 'source-handoff/surfaces.json')['planes'], key=lambda p: -p['support_area'])
    for j in (0, 1, 9):
        p = planes[j]
        mask = cv2.resize(unpack(p), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        yield (p, mask)


def periods(img, support):
    """Directional autocorrelation proposes one-block periods and paired-block harmonics."""
    mono = img.astype('f').mean(2)
    mask = cv2.erode(support, np.ones((7, 7), 'B')).astype(bool)
    peaks = []
    for axis in (1, 0):
        gray = mono - cv2.GaussianBlur(mono, (13, 1) if axis == 1 else (1, 13), 2)
        arr = gray if axis == 1 else gray.T
        valid = mask if axis == 1 else mask.T
        response = []
        for k in range(6, min(90, arr.shape[1] - 2)):
            ok = valid[:, :-k] & valid[:, k:]
            x = arr[:, :-k][ok]
            y = arr[:, k:][ok]
            response.append(float(x @ y / max(1, np.linalg.norm(x) * np.linalg.norm(y))) if len(x) > 75 else -1.0)
        options = []
        for j, r in enumerate(response[1:-1], 1):
            if r < 0.14 or r <= response[j - 1] or r < response[j + 1]:
                continue
            delta = 0.5 * (response[j - 1] - response[j + 1]) / (response[j - 1] - 2 * r + response[j + 1])
            options.append((r, j + 6 + delta))
        peaks.append(sorted(options, reverse=True)[:4])
    proposed = {}
    for axis, options in enumerate(peaks):
        for r, k in options:
            for factor in (1, 0.5):
                a = k * factor
                if a < 10:
                    continue
                for b in [a] + [s * f for _, s in peaks[1 - axis] for f in (1, 0.5) if 0.8 < s * f / a < 1.22]:
                    x, y = (a, b) if axis == 0 else (b, a)
                    key = (round(x * 2) / 2, round(y * 2) / 2)
                    proposed[key] = max(proposed.get(key, 0), r * (0.85 if factor == 0.5 else 1))
    return (sorted(proposed, key=proposed.get, reverse=True)[:30], peaks)


def seeds(img, support):
    dist = cv2.distanceTransform(support, cv2.DIST_L2, 5)
    result = []
    for i in range(12):
        _, value, _, (x, y) = cv2.minMaxLoc(dist)
        if value < 6:
            break
        radius = max(5, int(value * 0.75))
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        valid = ((xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2) & (support > 0)
        pixels = img[valid]
        shortlist, prior = _shortlist(pixels, np.ones(len(pixels), bool), atlas())
        result.append((x, y, shortlist[:20], prior))
        cv2.circle(dist, (x, y), max(22, int(2.2 * value)), 0.0, -1)
    return result


def anchors(img, support, steps, samples):
    cat = atlas()
    gray = img.astype('f').mean(2)
    candidates = []
    for dx, dy in steps:
        locals_ = []
        for sx, sy, tiles, prior in samples:
            # The observed seed can lie anywhere in its block. Search over local phase.
            left = sx - dx
            top = sy - dy
            u, v = np.meshgrid(left + (np.arange(32) + 0.5) * dx / 16, top + (np.arange(32) + 0.5) * dy / 16)
            grid = np.stack((u - 0.5, v - 0.5), axis=-1).astype('f')
            patch = cv2.remap(gray, grid, None, cv2.INTER_LINEAR)
            valid = cv2.remap(support, grid, None, cv2.INTER_NEAREST).astype('f')
            full = cv2.matchTemplate(valid, np.ones((16, 16), 'f'), cv2.TM_CCORR) > 246
            if not full.any():
                continue
            best = None
            for idx in tiles:
                if cat.labels[idx] not in cat.face_choices['north']:
                    continue
                for tile in cat.patterns[idx]:
                    corr = cv2.matchTemplate(patch, tile, cv2.TM_CCOEFF_NORMED)
                    corr[~full] = -1
                    _, r, _, (x, y) = cv2.minMaxLoc(corr)
                    loss = r - min(prior[idx], 60) / 160
                    if best is None or loss > best[0]:
                        best = (loss, r, int(idx), left + x * dx / 16, top + y * dy / 16, dx, dy)
            if best is not None and best[1] > 0.43:
                locals_.append(best)
        candidates.extend(sorted(locals_, reverse=True)[:5])
    return sorted(candidates, reverse=True)


UNIT_SAMPLES = np.stack(np.meshgrid((np.arange(16) + 0.5) / 16, (np.arange(16) + 0.5) / 16), axis=-1)


def remap_rows(data, uv, interpolation):
    return np.concatenate([cv2.remap(data, uv[i:i+16000], None, interpolation)
                           for i in range(0,len(uv),16000)])


def assess(img, mask, proposal, tile_ids):
    """Rank tile orientations over observed pixels, including partly occluded faces."""
    _, _, _, x, y, dx, dy = proposal
    cat = atlas()
    gx,gy=np.meshgrid(np.arange(int(np.floor(-x/dx)),int(np.ceil((img.shape[1]-x)/dx))),
                      np.arange(int(np.floor(-y/dy)),int(np.ceil((img.shape[0]-y)/dy))))
    cells = np.column_stack((gx.ravel(), gy.ravel()))
    pts = (cells[:, None, None, :] + UNIT_SAMPLES) * [dx, dy] + [x - 0.5, y - 0.5]
    uv = pts.astype('f').reshape(-1, 16, 2)
    support = remap_rows(mask, uv, cv2.INTER_NEAREST).reshape(-1, 256).astype('f')
    patches = remap_rows(img, uv, cv2.INTER_LINEAR).reshape(-1, 256, 3).astype('f')
    keep = support.mean(1) >= 0.32
    cells = cells[keep]
    support = support[keep]
    patches = patches[keep]
    if len(cells) == 0:
        return (0, [])
    n = support.sum(1)
    rgb = (support[..., None] * patches).sum(1) / n[:, None]
    gray = patches.mean(2)
    gray = (gray - (support * gray).sum(1)[:, None] / n[:, None]) * support
    tiles = cat.patterns[tile_ids].reshape(-1, 256)
    numerator = gray @ tiles.T
    var = np.maximum(1, support @ (tiles ** 2).T - (support @ tiles.T) ** 2 / n[:, None])
    corr = numerator / (np.sqrt(var) * np.maximum(1, np.linalg.norm(gray, axis=1))[:, None])
    corr = corr.reshape(len(cells), len(tile_ids), 8).max(2)
    means = cat.means[tile_ids]
    gain = np.clip(rgb @ means.T / np.maximum(1, (means ** 2).sum(1)), 0.42, 1.2)
    color = np.mean(abs(means[None] * gain[..., None] - rgb[:, None]), axis=2)
    quality = corr - color / 65
    best = quality.argmax(1)
    rows = np.arange(len(best))
    score = (np.maximum(0, quality[rows, best] - 0.4) * n).sum() * dx * dy / 256
    data=[(int(u),int(v),int(tile_ids[idx]),float(r),float(c),float(count/256))
          for (u,v),idx,r,c,count in zip(cells,best,corr[rows,best],color[rows,best],n)]
    return (score, data)


def solve(img, support):
    """Choose and refine the source phase that explains the most observed texture area."""
    steps, peaks = periods(img, support)
    sample = seeds(img, support)
    options = anchors(img, support, steps, sample)
    if not options:
        return None
    tile_ids = sorted({opt[2] for opt in options[:80]})
    ranked = []
    # Include each plausible pitch and both dominant color groups.
    for opt in options:
        score, rows = assess(img, support, opt, tile_ids)
        ranked.append((score, opt))
    score, opt = max(ranked)
    for pos_delta, step_delta in [(1.2, 0.3), (0.6, 0.12)]:
        for j, d in ((3, pos_delta), (4, pos_delta), (5, step_delta), (6, step_delta)):
            trials = []
            for t in (-2, -1, 0, 1, 2):
                p = list(opt)
                p[j] += t * d
                sc, _ = assess(img, support, p, tile_ids)
                trials.append((sc, tuple(p)))
            score, opt = max(trials)
    return (opt, assess(img, support, opt, tile_ids)[1], peaks)


def emit(plane, H, opt, rows, mode):
    cat = atlas()
    strong = {(u, v): idx for u, v, idx, r, color, f in rows if r >= 0.57 and color < 13 and (f >= 0.65)}
    if len(strong) < 2:
        return None
    def neighbors(u, v):
        return [(u + 1, v), (u - 1, v), (u, v + 1), (u, v - 1)]
    if not any((n in strong for uv in strong for n in neighbors(*uv))):
        return None
    cells = []
    for u, v, idx, r, color, f in rows:
        direct = (u, v) in strong
        continuation=(mode=='repeat' and r>=.32 and color<13 and f>=.46 and
                      any(strong.get(n)==idx for n in neighbors(u,v)))
        if not (direct or continuation):
            continue
        texture = cat.labels[idx]
        material = cat.face_choices['north'][texture][0]
        cells.append(dict(u=u,v=v,block_id=material,confidence=round(r,4),fallback=False,
                          texture=texture,observed_fraction=round(f,3)))
    _, _, _, x, y, dx, dy = opt
    chart = H @ np.array([[dx, 0, x], [0, dy, y], [0, 0, 1]])
    return dict(id=plane['id'],normal_camera=plane['normal_camera'],H=chart.tolist(),cells=cells,
                confidence=float(np.mean([f['confidence'] for f in cells])),method='source-repeat-'+mode)


def reviewed_arms():
    cv2.setNumThreads(1)
    for case, packet in inputs():
        by_arm = {mode: [] for mode in ('anchor', 'repeat')}
        if 'walled-garden-1-warm-high-right' in case['id']:
            rgb = np.array(Image.open(case['variants']['sky']['image']).convert('RGB'))
            segments = line_segments(rgb)
            vp = pencils(rgb, segments)
            for plane, mask in reviewed_garden_planes(case, rgb):
                result = rectify(rgb, mask, vp, segments)
                if result is None:
                    continue
                H, img, support = result
                solution = solve(img, support)
                if solution is None:
                    continue
                opt, rows, peaks = solution
                print(plane['id'], opt, flush=True)
                for mode, charts in by_arm.items():
                    chart = emit(plane, H, opt, rows, mode)
                    if chart is not None:
                        charts.append(chart)
                    print(mode, Counter((c['block_id'] for c in chart['cells'])) if chart else {}, flush=True)
                save(OUTPUT/'evidence'/(plane['id']+'.json'),dict(anchor=list(opt),peaks=peaks,
                     native_basis=H.tolist(),observations=rows))
        for mode, charts in by_arm.items():
            replace = {c['id'] for c in charts}
            result = dict(packet, charts=[c for c in packet['charts'] if c['id'] not in replace] + charts)
            save(OUTPUT / mode / 'sky' / case['id'] / 'charts.json', result)





def connected(plane, H, opt, rows):
    """Retain the largest contiguous region certified by adjacent strong full-face anchors."""
    cat = atlas()
    valid = {(u, v): (idx, r, f) for u, v, idx, r, color, f in rows if r >= 0.32 and color < 13 and (f >= 0.46)}
    pending = set(valid)
    best = []
    while pending:
        first = min(pending)
        tile_id = valid[first][0]
        stack = [first]
        pending.remove(first)
        component = []
        while stack:
            u, v = stack.pop()
            component.append((u, v))
            for point in ((u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)):
                if point in pending and valid[point][0] == tile_id:
                    pending.remove(point)
                    stack.append(point)
        strong = {(u, v) for u, v in component if valid[u, v][1] >= 0.76 and valid[u, v][2] >= 0.78}
        if len(strong) < 2 or max((valid[p][1] for p in strong)) < 0.85:
            continue
        if not any(((u + 1, v) in strong or (u, v + 1) in strong for u, v in strong)):
            continue
        if len(component) > len(best):
            best = component
    if not best:
        return None
    cells = []
    for u, v in sorted(best):
        idx, r, f = valid[u, v]
        texture = cat.labels[idx]
        cells.append(dict(u=u,v=v,block_id=cat.face_choices['north'][texture][0],texture=texture,
                          confidence=round(r,4),observed_fraction=round(f,3),fallback=False))
    _, _, _, x, y, dx, dy = opt
    chart = H @ np.array([[dx, 0, x], [0, dy, y], [0, 0, 1]])
    return dict(id=plane['id'],normal_camera=plane['normal_camera'],H=chart.tolist(),cells=cells,
                confidence=float(np.mean([f['confidence'] for f in cells])),method='connected-source-atlas')


def all_walls(job):
    cv2.setNumThreads(1)
    case, packet = job
    rgb = np.array(Image.open(case['variants']['sky']['image']).convert('RGB'))
    segments = line_segments(rgb)
    vp = pencils(rgb, segments)
    accepted = []
    seen = {c['id'] for c in packet['charts']}
    handoff = HANDOFF / case['id'] / 'source-handoff/surfaces.json'
    planes = read(handoff)['planes']
    records = []
    for plane in sorted(planes, key=lambda p: -p['support_area']):
        if plane['id'] in seen or not plane.get('normal_camera') or abs(plane['normal_camera'][1]) >= 0.65:
            continue
        mask = cv2.resize(unpack(plane), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask.sum() < 1800:
            continue
        result = rectify(rgb, mask, vp, segments)
        if result is None:
            continue
        H, img, support = result
        solution = solve(img, support)
        if solution is None:
            continue
        opt, rows, peaks = solution
        chart = connected(plane, H, opt, rows)
        records.append(dict(id=plane['id'], pitch=opt[-2:], observed_cells=len(rows), accepted=bool(chart)))
        if chart is not None:
            accepted.append(chart)
            print(case['id'], chart['id'], len(chart['cells']), chart['cells'][0]['block_id'], flush=True)
    folder = OUTPUT / 'connected' / 'sky' / case['id']
    save(folder/'evidence.json',dict(baseline_sha256=digest(baseline_file(case['id'])),
        source_sha256=packet['source_sha256'],surfaces_sha256=digest(handoff),records=records))
    save(folder / 'charts.json', dict(packet, charts=packet['charts'] + accepted))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['reviewed', 'connected'], default='connected')
    parser.add_argument('--workers', type=int, default=3)
    cfg = parser.parse_args()
    if cfg.arm == 'reviewed':
        reviewed_arms()
    else:
        with ProcessPoolExecutor(cfg.workers) as pool:
            list(pool.map(all_walls, inputs()))

