"""Evidence for ranking competing depths of the same observed planar patch."""
import numpy as np


NAMES = ('atlas', 'atlas_two', 'atlas_five', 'strong_tiles', 'texel_edge', 'boundary',
         'edge_boundary', 'atlas_boundary', 'atlas_edge', 'original', 'shift', 'scene_shift')


def features(rows, baseline):
    values = []
    for r in rows:
        atlas = np.clip(r['atlas'],0,1.)
        edge = max(r['edge'],0.)
        border = r['boundary']
        count = r['tiles']
        values.append([atlas, atlas*min(count/2,1),atlas*min(count/5,1),
                       min(r['distinct_votes']/6,1),edge,border,edge*border,
                       atlas*border,atlas*edge,float(r['level']==baseline),
                       min(abs(r['level']-baseline),12)/12,min(r['scene_delta'],12)/12])
    return np.array(values)


def ranked_choice(rows, baseline, weights, minimum_gain=0.):
    if not rows:
        return baseline
    scores=features(rows,baseline) @ weights
    winner=int(np.argmax(scores))
    before=next((i for i,r in enumerate(rows) if r['level']==baseline),None)
    if before is None or scores[winner]-scores[before] <= minimum_gain:
        return baseline
    return rows[winner]['level']
