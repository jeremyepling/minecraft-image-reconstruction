"""Camera from RGB Manhattan lines, ground contact and visible facade evidence."""
from functools import lru_cache

import numpy as np
from PIL import Image

from engine.plane_candidates import foreground


def _norm(arr):
    return arr / np.maximum(1.e-9, np.linalg.norm(arr, axis=-1, keepdims=True))


def _line_data(image):
    import cv2

    cv2.setNumThreads(1)
    rgb = np.asarray(image.convert("RGB"))
    _, mask = foreground(image)
    mask = cv2.resize(mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)
    h, w = rgb.shape[:2]
    bands = np.concatenate([rgb[:, :w//12], rgb[:, -w//12:]], axis=1).astype(float)
    colors = np.median(bands[:, ::4], axis=1)
    row_change = np.linalg.norm(colors[2:] - colors[:-2], axis=1)
    horizon = int(np.argmax(row_change[h//8:7*h//8]) + h//8 + 1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    field = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if field is None:
        return None
    seg = field[:, 0].reshape(-1, 2, 2) + .5
    mid = seg.mean(1)
    length = np.linalg.norm(seg[:, 1]-seg[:, 0], axis=1)
    x, y = np.clip(mid.astype(int), [0, 0], [w-1, h-1]).T
    neighborhood = cv2.dilate(mask, np.ones((13, 13), np.uint8))
    keep = (neighborhood[y, x] > 0) & (length > max(10, .008*w))
    seg = seg[keep][np.argsort(-length[keep], kind="stable")]
    return rgb, mask, seg, horizon


def _vp(seg, horizon, side, size):
    w, h = size
    # The two horizontal vanishing points straddle the principal point in an upright camera.
    center = np.array([w/2, horizon])
    ends = (seg-center)/w
    delta = ends[:, 1]-ends[:, 0]
    length = np.linalg.norm(delta, axis=1)
    line = _norm(np.column_stack([delta[:, 1], -delta[:, 0]]))
    mid = ends.mean(1)
    bias = -(line*mid).sum(1)
    ids = np.flatnonzero(abs(line[:, 1]) > .63)[:170]
    if len(ids) < 4:
        return None
    line, bias, mid, weights = line[ids], bias[ids], mid[ids], np.minimum(length[ids], .17)
    equation = np.column_stack([line, bias])
    i, j = np.triu_indices(len(mid), 1)
    p = np.cross(equation[i], equation[j])
    good = (abs(p[:, 2]) > 1.e-9) & (p[:, 0]*p[:, 2]*side > 0)
    pts = p[good, :2]/p[good, 2:3]
    pts = pts[(abs(pts[:, 1])<.025) & (abs(pts[:, 0])>.19) & (abs(pts[:, 0])<15)]
    if len(pts)==0:
        return None
    best, point = 0., None
    for batch in np.array_split(pts, 1+len(pts)//400):
        dist = np.linalg.norm(batch[:, None]-mid, axis=2)
        angular = abs(batch @ line.T+bias)/np.maximum(.03, dist)
        scores = np.maximum(0, 1-(angular/.007)**2) @ weights
        k = scores.argmax()
        if scores[k] > best:
            best, point = scores[k], batch[k]
    for _ in range(4):
        dist = np.maximum(.03, np.linalg.norm(point-mid, axis=1))
        angular = abs(line @ point+bias)/dist
        wt = np.sqrt(weights*np.maximum(0, 1-(angular/.005)**2))/dist
        if (wt>0).sum()<3:
            return None
        point = np.linalg.lstsq(wt[:, None]*line, -wt*bias, rcond=None)[0]
    return point*w+center


def _optics(seg, horizon, size):
    w, h = size
    center = np.array([w/2, h/2])
    hps = [_vp(seg, horizon, side, size) for side in (-1, 1)]
    if any(p is None for p in hps):
        return None
    a, b = hps
    f2 = -((a-center) @ (b-center))
    if not (.34*h)**2 < f2 < (2.8*h)**2 or abs(a[1]-b[1]) > .018*h:
        return None
    focal = np.sqrt(f2)
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1.]])
    # Intersect the recovered horizon with the optical vertical.
    hy = a[1]+(b[1]-a[1])*(w/2-a[0])/(b[0]-a[0])
    pitch = np.arctan2(h/2-hy, focal)
    ey = np.array([0., -np.cos(pitch), -np.sin(pitch)])
    ex = _norm(np.linalg.solve(K, [b[0], hy, 1.]))
    ex = _norm(ex-ey*(ex @ ey))
    ez = np.cross(ey, ex)
    M = K @ np.stack([ex, ey, ez], axis=1)
    return M


def _ground(image, mask, seg, M):
    h, w = mask.shape
    y_any = mask.any(0)
    bottom = np.where(y_any, h-1-mask[::-1].argmax(0), -1)
    vs = M[:, 1]
    ground_lines, upright = [], []
    for i, ab in enumerate(seg[:350]):
        mid = ab.mean(0)
        direction = _norm(ab[1]-ab[0])
        if abs(direction[1])>.75:
            to_vp = _norm(vs[:2]-mid*vs[2])
            if abs(direction @ to_vp)>.999:
                upright.append(ab)
            continue
        a, b = ab[np.argsort(ab[:, 0])]
        if b[0]-a[0] < 19:
            continue
        x = np.linspace(a[0]+2, b[0]-2, 11)
        y = np.interp(x, [a[0], b[0]], [a[1], b[1]])
        err = bottom[np.clip(x.astype(int), 0, w-1)]-y
        if abs(np.median(err)) > 10 or np.quantile(abs(err), .7)>14:
            continue
        vps = M[:, [0, 2]].T
        sim = abs(_norm(vps[:, :2]-mid*vps[:, 2:3]) @ direction)
        if sim.max() < .9999:
            continue
        ground_lines.append((int(np.argmax(sim))*2, a, b))
    if not ground_lines or not upright:
        return []
    anchors = []
    for axis, a, b in ground_lines:
        gl = np.cross(np.r_[a, 1], np.r_[b, 1])
        for ab in upright:
            foot = ab[ab[:, 1].argmax()]
            if min(np.linalg.norm(foot-a), np.linalg.norm(foot-b))>14:
                continue
            cross = np.cross(gl, np.cross(np.r_[ab[0], 1], np.r_[ab[1], 1]))
            xy = cross[:2]/cross[2]
            ray = np.linalg.solve(M, np.r_[xy, 1])
            if ray[1] >= -.00001:
                continue
            origin = -ray/ray[1]
            anchors.append((float(np.linalg.norm(b-a)), axis, xy, origin))
    return sorted(anchors, key=lambda item: -item[0])[:14]


def _wall_coords(seg, M, anchor):
    _, axis, _, origin = anchor
    normal = 2-axis
    rays = np.linalg.solve(M, np.column_stack([seg.reshape(-1, 2), np.ones(2*len(seg))]).T).T
    den = rays[:, normal:normal+1]
    world = rays*origin[normal]/np.where(abs(den)>1.e-6, den, np.nan)-origin
    return world[:, [axis, 1]].reshape(-1, 2, 2)


def _hypotheses(seg, M, anchors):
    options = []
    for anchor in anchors:
        chart = _wall_coords(seg[:320], M, anchor)
        delta = abs(chart[:, 1]-chart[:, 0])
        for dim in range(2):
            main, off = delta[:, dim], delta[:, 1-dim]
            keep = (main>.032) & (main<1.8) & (off < .017*main)
            # LSD endpoints truncate at corners, particularly on small faces.
            for span in main[keep]:
                height = 1/span
                if .7<height<22:
                    options.append((height, anchor))
    return options


def _lattice(seg, M, anchor, height):
    chart = _wall_coords(seg[:260], M, anchor)*height
    delta = abs(chart[:, 1]-chart[:, 0])
    dim = delta.argmax(1)
    cross = 1-dim
    idx = np.arange(len(dim))
    pos = chart.mean(1)[idx, cross]
    align = delta[idx, cross]/np.maximum(.01, delta[idx, dim])
    length = np.linalg.norm(seg[:260, 1]-seg[:260, 0], axis=1)
    keep = (align<.03) & (abs(pos)<18) & np.isfinite(pos)
    residual = abs(pos-np.rint(pos))
    votes = np.maximum(0, 1-(residual/.11)**2)*np.minimum(90, length)
    votes = np.where(keep, votes, 0)
    # Distinct parallel tracks matter more than multiple detections of one edge.
    tracks = {}
    for i in np.flatnonzero(votes>0):
        key = (int(cross[i]), int(round(pos[i])))
        tracks[key] = max(votes[i], tracks.get(key, 0.))
    return sum(tracks.values()), len(tracks)


def _result(M, anchor, height, evidence):
    P = np.column_stack([M, M @ anchor[3]*height])
    return dict(P=P.tolist(), axes_homogeneous=M.T.tolist(), planes=[], scale_status="supported",
                reason="wall_lattice", evidence=evidence)


def _abstain(reason, **evidence):
    return dict(P=None, planes=[], scale_status="uncertain", reason=reason, evidence=evidence)


def predict(image, version="26.3"):
    data = _line_data(image)
    if data is None:
        return _abstain("missing_lines")
    rgb, mask, seg, horizon = data
    M = _optics(seg, horizon, image.size)
    if M is None:
        return _abstain("missing_manhattan_pencils")
    anchors = _ground(rgb, mask, seg, M)
    if not anchors:
        return _abstain("no_ground_contact")
    best = None
    for anchor in anchors:
        hs = sorted(set(round(h, 2) for h, _ in _hypotheses(seg, M, [anchor])))
        for height in hs:
            score, tracks = _lattice(seg, M, anchor, height)
            if tracks>=5 and (best is None or score>best[0]):
                best = score, tracks, anchor, height
    if best is None:
        return _abstain("no_wall_unit")
    score, tracks, anchor, height = best
    return _result(M, anchor, height, dict(height=height, tracks=tracks, score=score,
                   anchor=anchor[2].tolist(), axis=anchor[1]))

