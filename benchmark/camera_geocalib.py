"""Offer GeoCalib horizon and focal proposals to the bilateral full-tile verifier."""
import math
import numpy as np

from benchmark.camera_horizon_floor import boundary
from benchmark.camera_horizon_repeat import charts, peaks, fit_translation, phase_origin
from engine.camera_atlas_floor import register, _atlas
from engine.camera_texel_floor import _ground

# Apply only after both ordinary camera and unconstrained repeat search abstain.
HORIZON_OFFSETS = (-.08, -.04, -.02, 0., .02, .04, .08)
MAX_FITS = 12


def solve(image, proposal=None):
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    ref = .91*h
    horizon = proposal["horizon_center_px"] if proposal else boundary(rgb.astype(np.int16))
    focal = proposal["focal_xy_px"][1] if proposal else None
    evidence = dict(seed_horizon_px=horizon, seed_focal_px=focal, fits=0)
    rejected = dict(P=None, scale_status="uncertain", reason="no bilaterally verified floor proposal", evidence=evidence)
    if proposal and (horizon is None or not math.isfinite(horizon) or focal <= 0):
        return rejected
    gray = rgb.mean(2)
    limits = np.quantile(gray[int(.82*h)::2, ::3], [.08, .92])
    gray = np.clip(gray, *limits)
    seeds = []
    if horizon is None:
        hypotheses = [(trial*h, 0.) for trial in np.linspace(-.35, .73, len(HORIZON_OFFSETS))]
    else:
        hypotheses = [(horizon + offset*h, offset) for offset in HORIZON_OFFSETS]
    for trial, offset in hypotheses:
        if not -.8*h < trial <= .73*h:
            continue
        for corr, inv, vectors in charts(peaks(gray, trial, ref), trial, ref, w, h):
            _, implied_focal, _ = _ground(inv, w, h)
            focal_penalty = .25*abs(math.log(implied_focal/focal)) if focal is not None else 0.
            rank = corr - focal_penalty - .5*abs(offset)
            seeds.append((rank, trial, vectors))
    tried = []
    for _, trial, vectors in sorted(seeds, key=lambda s: -s[0]):
        if any(abs(trial-other) < .02*h and np.allclose(vectors, vec, atol=2) for other, vec in tried):
            continue
        tried.append((trial, vectors))
        evidence["fits"] = len(tried)
        fitted, vec, residual = fit_translation(gray, trial, ref, vectors)
        d = ref-fitted
        inv = np.array([[d, 0, -d*w/2], [0, -d, d*ref], [0, 1, -fitted]])
        if abs(np.linalg.det(vec)) > 1e-6:
            inv[:2] = 16*np.linalg.solve(vec, inv[:2])
            if _ground(inv, w, h) is not None:
                inv[:2] += phase_origin(gray, inv)[:, None]*inv[2]
                origin, match = register(image, inv, _atlas("26.3"))
                if origin is not None:
                    inv[:2] = (inv[:2]+origin[:, None]*inv[2])/16
                    metric = _ground(inv, w, h)
                    if metric is not None:
                        P, focal_fit, disagreement = metric
                        evidence.update(horizon_px=fitted, focal_px=focal_fit, repeat_residual=residual,
                                        axis_length_disagreement=disagreement, **match)
                        return dict(P=P, scale_status="supported", camera_source="geocalib-repeat" if proposal else "boundary-budget-repeat",
                                    reason="two tile translations registered independently in both halves", evidence=evidence)
        if len(tried) >= MAX_FITS:
            break
    return rejected
