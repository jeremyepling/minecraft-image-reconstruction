"""Test a source-only wall-chart partition against independent local RGB edge lattices."""
from collections import defaultdict
from functools import lru_cache

import cv2
import numpy as np

from engine.geometry import project

# Nonoverlapping membership regions, each with a centered measurement window.
WINDOW = 8
REGION = 4
RATE = 10
PITCH = np.r_[np.arange(.45, 1.61, .05), np.arange(1.7, 2.71, .1)]
SLOPE = np.arange(-.4, .401, .1)
N = WINDOW * RATE
COORD = (np.arange(N) + .5) / RATE - WINDOW / 2
XY = np.stack(np.meshgrid(COORD, COORD), axis=-1)
FOURIER = np.exp(2j * np.pi / PITCH[:, None] * COORD).astype('complex64')


@lru_cache(1)
def shear_phase():
    return np.exp(-2j * np.pi * SLOPE[:, None, None] / PITCH[None, :, None] * COORD).astype('complex64')


def axis_fit(pixels, support):
    # Different rows supply the fit and validation. Remove each row's mean
    # edge strength so the silhouette and mask footprint cannot form a comb.
    gradient = np.mean(np.abs(pixels[:, 2:] - pixels[:, :-2]), axis=2)
    valid = support[:, 2:] & support[:, :-2] & support[:, 1:-1]
    count = valid.sum(1)
    usable = count >= 3 * RATE
    if usable.sum() < 2 * RATE:
        return None
    signals = np.where(valid, gradient - (gradient * valid).sum(1)[:, None] / np.maximum(1, count[:, None]), 0)
    signals[~usable] = 0
    strength = np.sum(np.abs(signals), axis=1)
    spectral = np.einsum('px,yx->py', FOURIER[:, 1:-1], signals.astype('float32'), optimize=False)
    phases = shear_phase() * spectral[None, :, :]
    folds = []
    for fold in (0, 1):
        rows = np.arange(N) // RATE % 2 == fold
        mass = strength[rows].sum()
        if mass < 40 or (rows & usable).sum() < RATE:
            return None
        folds.append(phases[:, :, rows].sum(2) / mass)
    fit, check = folds
    i, j = np.unravel_index(np.abs(fit).argmax(), fit.shape)
    wave = fit[i, j]
    phase = np.angle(wave) / (2 * np.pi)
    validation = (check[i, j] * np.exp(-2j * np.pi * phase)).real
    old_index = np.argmin(np.abs(PITCH - 1))
    old = check[len(SLOPE) // 2, old_index].real
    pitch, slope = float(PITCH[j]), float(SLOPE[i])
    phase_blocks = float(phase * pitch)
    # Require repeatable held-out seams, visibly distinct from the chart.
    disagrees = (abs(pitch - 1) > .18 or abs(slope) > .14 or abs(phase_blocks) > .22)
    rejected = disagrees and abs(wave) >= .26 and validation >= .21 and validation - old >= .14
    return dict(pitch_chart_units=pitch, transverse_slope=slope, phase_at_patch_center=phase_blocks,
                fit_strength=float(abs(wave)), validation_strength=float(validation),
                original_validation=float(old), disagrees=bool(rejected))


def partition(rgb, mask, chart, seed):
    cells = chart['cells']
    if not cells or not seed['cells'] or not seed.get('normal_camera') or abs(seed['normal_camera'][1]) >= .67:
        return chart, []
    assert np.array_equal(seed['H'], chart['H'])
    seeds = {(c['u'], c['v']) for c in seed['cells']}
    regions = defaultdict(list)
    for c in cells:
        regions[c['u'] // REGION, c['v'] // REGION].append(c)
    rejected = set()
    observations = []
    for region, members in sorted(regions.items()):
        center = np.asarray(region) * REGION + REGION // 2
        coords = project(np.asarray(chart['H']), XY + center).astype('float32') - .5
        pixels = cv2.remap(rgb, coords, None, cv2.INTER_LINEAR).astype('float32')
        support = cv2.remap(mask, coords, None, cv2.INTER_NEAREST).astype(bool)
        fits = [axis_fit(pixels, support), axis_fit(pixels.transpose(1, 0, 2), support.T)]
        veto = any(f and f['disagrees'] for f in fits)
        if veto:
            rejected.update((c['u'], c['v']) for c in members if (c['u'], c['v']) not in seeds)
        observations.append(dict(region=list(region), center_uv=center.tolist(), members=len(members),
                                 rejected=bool(veto), fits=fits))
    return dict(chart,cells=[c for c in cells if (c['u'],c['v']) not in rejected]), observations
