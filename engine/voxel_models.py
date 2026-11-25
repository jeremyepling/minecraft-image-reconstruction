"""Single-view geometric baselines, fitted alternatives and a fixed voting rule."""
from importlib import import_module
import numpy as np

FITTED = {'boxes':'voxel_boxes', 'carving':'voxel_carving', 'layers':'voxel_layers', 'recess':'voxel_recess'}


def predict(scene, method):
    hull = scene['hull']
    if method == 'cone':
        return hull.copy()
    if method == 'sheet':
        # A one-block-thick facade, cut at the most occupied depth.
        depth = int(hull.sum((0,1)).argmax())
        blocks = np.zeros_like(hull)
        blocks[:,:,depth] = hull[:,:,depth]
        return blocks
    if method == 'symmetric':
        # Fit the depth reflection plane whose supported intersection explains
        # the greatest volume. Keeping both supporting rays gives a symmetric solid.
        best = np.zeros_like(hull)
        for i in range(2*hull.shape[2]-1):
            j = i-np.arange(hull.shape[2])
            valid = (j>=0)&(j<hull.shape[2])
            proposal = np.zeros_like(hull)
            proposal[:,:,valid] = hull[:,:,valid] & hull[:,:,j[valid]]
            if proposal.sum()>best.sum():
                best = proposal
        return best
    if method in FITTED:
        return np.asarray(import_module('engine.'+FITTED[method]).predict(scene), bool)
    raise ValueError(f'Unknown voxel fitter: {method}')


def vote(models):
    return np.sum(models, axis=0) > len(models)//2
