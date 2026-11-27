"""Verify the native phase and scale of a visible vanilla floor in separated RGB regions."""
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from engine.camera_atlas_floor import _atlas
from engine.camera_repeat_floor import boundary, charts, peaks
from engine.camera_texel_floor import _ground

try:
    import cv2
except (ImportError, OSError):
    cv2 = None


def patches(gray, bounds):
    """Separate source regions; remove slow illumination before pooling by texel."""
    h, w = gray.shape
    low = int(0.76 * h)
    region = gray[low:]
    center = np.median(region[::3, ::5])
    lo, hi = (center - 4, center + 4)
    good = cv2.erode(((region >= lo) & (region <= hi)).astype(np.uint8), np.ones((11, 11), np.uint8))
    region = np.clip(region, lo, hi)
    signal = region - cv2.GaussianBlur(region, (0, 0), 6)
    result = []
    for x0, x1 in bounds:
        y, x = np.mgrid[int(0.79 * h):int(0.98 * h):3, int(x0 * w):int(x1 * w):3]
        keep = good[y - low, x].astype(bool)
        x, y = (x[keep], y[keep])
        screen = np.array([x.ravel() + 0.5, y.ravel() + 0.5, np.ones(x.size)])
        result.append((screen, signal[y - low, x].ravel()))
    return result


def templates(atlas, ids, n):
    """Average linearized texels to model the renderer mip footprint."""
    tiles = atlas.tiles[ids].astype(float)
    mip = (tiles ** 2.2).reshape(len(ids), n, 16 // n, n, 16 // n, 3).mean((2, 4)) ** (1 / 2.2)
    gray = mip.mean(-1)
    return np.stack([np.rot90(a, k, axes=(-2, -1)) for a in (gray, gray[:, :, ::-1]) for k in range(4)], axis=1)


def coords(inv, points, n):
    """Map native pixel centers into periodic coordinates at the given mip size."""
    uv = inv @ points
    return n * uv[:2] / uv[2]


def sample_uv(tile, uv, mode):
    n = tile.shape[0]
    if mode == 'nearest':
        i, j = np.floor(uv).astype(int) % n
        return tile[j, i]
    u, v = uv - 0.5
    i, j = np.floor([u, v]).astype(int)
    a, b = (u - i, v - j)
    top = (1 - a) * tile[j % n, i % n] + a * tile[j % n, (i + 1) % n]
    bottom = (1 - a) * tile[(j + 1) % n, i % n] + a * tile[(j + 1) % n, (i + 1) % n]
    return (1 - b) * top + b * bottom


def correlation(a, b):
    a, b = (a - a.mean(), b - b.mean())
    return float(a @ b / max(1e-09, np.linalg.norm(a) * np.linalg.norm(b)))


def atlas_scan(inv, patches_, atlas, ids):
    """Search models on the west patch. Score its unmodified prediction on the east."""
    best = None
    for n in (16, 8, 4, 2):
        tiles = templates(atlas, ids, n)
        uv = coords(inv, patches_[0][0], n)
        signal = patches_[0][1]
        # FFT over integer phase is exactly a native nearest-texel dot product.
        i, j = np.floor(uv).astype(int) % n
        index = j * n + i
        observed = np.bincount(index, signal - signal.mean(), minlength=n * n).reshape(n, n)
        counts = np.bincount(index, minlength=n * n).reshape(n, n)

        def dots(a, b):
            return np.fft.irfft2(np.conj(np.fft.rfft2(a)) * np.fft.rfft2(b), s=(n, n), axes=(-2, -1))
        sums = dots(counts, tiles)
        variances = dots(counts, tiles ** 2) - sums ** 2 / len(signal)
        corr = dots(observed, tiles) / np.maximum(1e-06, np.sqrt(np.maximum(0, variances)) * np.linalg.norm(signal - signal.mean()))
        idx = np.unravel_index(corr.argmax(), corr.shape)
        ti, orient, dy, dx = map(int, idx)
        tile = tiles[ti, orient]
        for mode in ('nearest', 'linear'):
            phase = np.array([[dx], [dy]])
            scores = [correlation(target, sample_uv(tile, coords(inv, screen, n) + phase, mode)) for screen, target in patches_]
            if best is None or scores[0] > best['fit']:
                best = dict(fit=scores[0], hold=scores[1], n=n, mode=mode, orientation=orient, phase=[dx, dy], tile=atlas.labels[ids[ti]])
    return best


def chart(horizon, vectors, w, h):
    """Screen-to-floor homography from horizontal horizon and two tile translations."""
    ref = 0.91 * h
    d = ref - horizon
    result = np.array([[d, 0, -d * w / 2], [0, -d, d * ref], [0, 1, -horizon]])
    result[:2] = np.linalg.solve(vectors, result[:2])
    return result


def fit_periods(gray, valid, hor, vec, w, h):
    """Subpixel registration of two tile translations with shared horizon and valid native RGB."""
    ref = 0.91 * h
    y, x = np.mgrid[int(0.78 * h):int(0.97 * h):6, 8:w - 8:9]
    ok = valid[y, x]
    x, y = (x[ok], y[ok])
    xp = x + 0.5 - w / 2
    yp = y + 0.5
    if len(x) < 80:
        return hor, vec
    base = gray[y, x]
    valid_float = valid.astype('f')
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1) * 0.5
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1) * 0.5

    def mapping(v, z):
        d = ref - z
        ratio = (1 + v[1] * (yp - z) / d ** 2) ** (-1)
        return ((xp + v[0] * (yp - z) / d) * ratio + w / 2 - 0.5, (yp - z) * ratio + z - 0.5)

    def lookup(field, X, Y):
        # OpenCV remap requires both destination dimensions to be below SHRT_MAX.

        def remap(x, y):
            return cv2.remap(field, x.astype('f'), y.astype('f'), cv2.INTER_LINEAR).ravel()
        if len(X) < 32767:
            return remap(X, Y)
        return np.concatenate([remap(X[i:i + 30000], Y[i:i + 30000]) for i in range(0, len(X), 30000)])
    for _ in range(15):
        J = []
        residual = []
        for i in range(2):
            v = vec[:, i]
            X, Y = mapping(v, hor)
            xs = lookup(gx, X, Y)
            ys = lookup(gy, X, Y)
            measured = lookup(gray, X, Y)
            good = lookup(valid_float, X, Y) > 0.99
            if good.sum() < 80:
                return (hor, vec)
            jac = np.zeros((len(X), 5))
            for j in (0, 1, 2):
                vv = v.copy()
                if j < 2:
                    vv[j] += 0.12
                xx, yy = mapping(vv, hor + (0.3 if j == 2 else 0))
                jac[:, 4 if j == 2 else 2 * i + j] = ((xx - X) * xs + (yy - Y) * ys) / (0.3 if j == 2 else 0.12)
            J.append(jac[good])
            residual.append((base - measured)[good])
        update = np.linalg.lstsq(np.vstack(J), np.concatenate(residual), rcond=None)[0]
        vec += np.clip(update[:4], -0.7, 0.7).reshape(2, 2).T
        hor += np.clip(update[4], -3, 3)
        if np.max(abs(update)) < 0.015:
            break
    return (hor, vec)


def _solve(image, atlas):
    rgb = np.asarray(image.convert('RGB'))
    raw = rgb.mean(2).astype('f')
    h, w = raw.shape
    center = np.median(raw[int(0.8 * h)::4, ::4])
    good = cv2.erode((abs(raw - center) < 4).astype('uint8'), np.ones((9, 9), 'uint8')).astype(bool)
    a = np.clip(raw, center - 4, center + 4)
    gray = (a - cv2.GaussianBlur(a, (0, 0), 6)) * good
    samples = patches(raw, bounds=((0.04, 0.34), (0.61, 0.84), (0.89, 0.99), (0.4, 0.54)))
    evidence = dict(hypotheses=0, checked=0, best_min_correlation=0.0)
    failed = dict(P=None, reason='no native atlas agreement across separated floor regions', scale_status='uncertain', evidence=evidence)
    if any((len(y) < 400 or y.std() < 0.08 for _, y in samples)):
        return failed
    if 'light_gray_concrete' not in atlas.labels:
        return dict(failed, reason='verified floor sprite unavailable')
    ids = [atlas.labels.index('light_gray_concrete')]
    observed = np.median(rgb[int(0.8 * h)::5, ::7].reshape(-1, 3), axis=0)
    expected = atlas.means[ids[0]]
    gain = float(observed.mean() / expected.mean())
    if not 0.7 < gain < 1.25 or np.max(abs(observed - gain * expected)) > 7:
        return dict(failed, reason='floor RGB outside the calibrated light gray concrete range')
    candidates = []
    edge = boundary(rgb.astype(np.int16))
    horizons = np.linspace(-0.35, 0.72, 17) * h if edge is None else edge + np.linspace(-0.18, 0.04, 17) * h
    ref = 0.91 * h
    for hor in horizons:
        if not -0.8 * h < hor < 0.74 * h:
            continue
        for s, inv, v in charts(peaks(gray, hor, ref), hor, ref, w, h)[:15]:
            length = np.linalg.norm(v, axis=0)
            rank = s - 0.004 * sum(length)
            candidates.append((rank, hor, v))
    evidence['hypotheses'] = len(candidates)
    tried = []
    for rank, hor, v in sorted(candidates, key=lambda c: -c[0]):
        if any((abs(hor - z) < 0.02 * h and np.allclose(v, V, atol=2.5) for z, V in tried)):
            continue
        if len(tried) >= 20:
            break
        tried.append((hor, v.copy()))
        hor, v = fit_periods(gray, good, hor, v.copy(), w, h)
        if abs(np.linalg.det(v)) < 2:
            continue
        inv = chart(hor, v, w, h)
        if _ground(inv, w, h) is None:
            continue
        result = atlas_scan(inv, samples[:2], atlas, ids)
        quality = min(result['fit'], result['hold'])
        evidence['checked'] += 1
        evidence['best_min_correlation'] = max(quality, evidence['best_min_correlation'])
        if result['fit'] <= 0.18:
            continue
        verified = refine(inv, result, samples, atlas, ids, w, h)
        if verified is not None:
            final, projection = verified
            worst_region = min(final['native_correlation'])
            if worst_region > evidence.get('best_refined_min_correlation', -1):
                evidence.update(best_refined_min_correlation=worst_region, best_refined=final)
            if projection is None:
                continue
            return dict(P=projection, reason='native atlas fit on two separated floor regions',
                        scale_status='supported', camera_source='native-floor-atlas', evidence=dict(evidence, **final))
    return failed


def refine(inv, result, samples, atlas, ids, w, h):
    """Fit upright projective geometry on the west and far-east regions.

    Two separated native regions are withheld from all seven parameter updates.
    Phase, filtering, and signed atlas orientation remain frozen during refinement.
    """
    n = result['n']
    tile = templates(atlas, ids, n)[0, result['orientation']]
    inv = inv.copy()
    inv[:2] += np.array(result['phase'])[:, None] * inv[2] / n
    ref = 0.9 * h
    T = np.array([[w * 0.5, 0, w * 0.5], [0, h * 0.14, ref], [0, 0, 1.0]])
    local = inv @ T
    local /= local[2, 2]
    pts = np.concatenate([samples[j][0] for j in (0, 2)], axis=1)
    signal = np.concatenate([samples[j][1] for j in (0, 2)])
    X = np.linalg.solve(T, pts)

    def response(matrix, points):
        return sample_uv(tile, coords(matrix, points, n), 'linear')
    for _ in range(100):
        raw = response(local, X)
        y = signal - signal.mean()
        raw -= raw.mean()
        gain = raw @ y / max(raw @ raw, 1e-09)
        residual = y - raw * gain
        pars = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)]
        derivatives = []
        for i, j in pars:
            d = local.copy()
            eps = 0.00015
            d[i, j] += eps
            amp = response(d, X)
            amp -= amp.mean()
            derivatives.append((amp - raw) * gain / eps)
        change = np.linalg.lstsq(np.column_stack(derivatives), residual, rcond=None)[0]
        if not np.isfinite(change).all():
            return None
        # Limit maximum native phase change per iteration.
        trial = local.copy()
        for (i, j), delta in zip(pars, change):
            trial[i, j] += delta
        shift = np.quantile(abs(coords(trial, X, n) - coords(local, X, n)), 0.98)
        change /= max(1, shift / 0.65)
        for (i, j), delta in zip(pars, change):
            local[i, j] += delta
        if max(abs(change)) < 1.5e-05:
            break
    inv = local @ np.linalg.inv(T)
    metric = _ground(inv, w, h)
    if metric is None:
        return None
    scores = [correlation(y, response(inv, X)) for X, y in samples]
    record = dict(native_correlation=scores, mip_size=n, filter='bilinear', atlas_sha256=atlas.provenance['atlas_sha256'])
    if min(scores) < 0.6:
        return record, None
    # Physically distinct harmonic scales get their own west phase before held scoring.
    alternatives = []
    for scale in (0.5, 0.98, 1.02, 2.0):
        alt = inv.copy()
        alt[:2] *= scale
        fit = atlas_scan(alt, samples[:2], atlas, ids)
        alternatives.append(fit['hold'])
    if min(scores) - max(alternatives) < 0.26:
        return dict(record, wrong_scale=alternatives), None
    return dict(record, wrong_scale=alternatives, accepted=True), metric[0]


def _configured_fit(image, version, atlas_path):
    """Use the existing offline recovery interpreter when OpenCV is installed there."""
    configured = os.environ.get('MCIR_RECOVERY_PYTHON')
    if not configured:
        return None
    interpreter = Path(configured).resolve()
    if not interpreter.is_file() or interpreter == Path(sys.executable).resolve():
        return None
    arguments = [str(interpreter), '-m', 'engine.camera_native_floor', version]
    if atlas_path is not None:
        arguments.append(str(Path(atlas_path).resolve()))
    pixels = BytesIO()
    image.convert('RGB').save(pixels, format='PNG', compress_level=1)
    try:
        run = subprocess.run(arguments, cwd=Path(__file__).resolve().parents[1],
                             input=pixels.getvalue(), capture_output=True, timeout=300, check=True,
                             env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                                  'MKL_NUM_THREADS': '1'})
        return json.loads(run.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def predict(image, version='26.3', *, atlas_path=None):
    def abstain(reason):
        return dict(P=None, scale_status='uncertain', reason=reason, evidence={})

    if version != '26.3':
        return abstain('unsupported atlas version')
    if min(image.size) < 256 or max(image.size) > 4096:
        return abstain('outside native texel sampling range')
    if cv2 is None:
        return _configured_fit(image, version, atlas_path) or abstain('native image filters unavailable')
    atlas = _atlas(version, atlas_path)
    if atlas is None:
        return abstain('calibrated atlas unavailable')
    try:
        with np.errstate(divide='raise', invalid='raise', over='raise'):
            return _solve(image, atlas)
    except (np.linalg.LinAlgError, FloatingPointError):
        return abstain('degenerate native floor fit')


if __name__ == '__main__':
    # This internal command takes one RGB image on stdin and returns its decision.
    if cv2 is None:
        sys.exit(1)
    with Image.open(BytesIO(sys.stdin.buffer.read())) as supplied:
        result = predict(supplied, sys.argv[1], atlas_path=sys.argv[2] if len(sys.argv) > 2 else None)
    sys.stdout.write(json.dumps(result, allow_nan=False))
