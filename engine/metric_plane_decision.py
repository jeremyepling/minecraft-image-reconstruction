"""Rank integer plane offsets using visible texture and floor-calibrated depth."""
import numpy as np

from engine.plane_offset_features import features


def combined_features(rows, baseline, measurement):
    rgb_features = features(rows,baseline)
    depth_features = np.zeros((len(rows),3))
    if measurement is not None:
        center, spread = measurement
        error = abs(np.array([row['level'] for row in rows])-center)
        depth_features[:,0] = np.exp(-error/(.6+min(spread,3)))
        depth_features[:,1] = np.minimum(error,6)/6
        depth_features[:,2] = depth_features[:,0] * min(1.,.5/max(spread,.1))
    return np.column_stack((rgb_features,depth_features))


def choose(rows,baseline,measurement,weights):
    if not rows:
        return baseline
    scores = combined_features(rows,baseline,measurement) @ weights
    return rows[int(np.argmax(scores))]['level']
