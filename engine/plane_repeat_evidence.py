"""Score unit-length translations along an observed planar texture."""
import numpy as np
from functools import lru_cache
import json
from pathlib import Path

from engine.atlas_plane_depth import _sample
from engine.visible_raster import SAMPLE_STRIDE
from engine.plane_offset_features import features, ranked_choice


def near_levels(eye_coordinate, baseline):
    """Cover short planes missed by a distant accidental floor contact."""
    side=1 if baseline>eye_coordinate else -1
    return [round(eye_coordinate)+side*i for i in range(1,25)]


def feature_rows(rows, baseline):
    ordinary=features(rows,baseline)
    hi=np.array([r['repeat_high'] for r in rows])
    lo=np.array([r['repeat_low'] for r in rows])
    return np.column_stack((ordinary,hi,lo,hi*ordinary[:,0],hi*ordinary[:,4],lo*ordinary[:,5]))


def pick(rows, baseline, weights, method):
    narrow=[r for r in rows if min(abs(r['level']-baseline),r['scene_delta'])<=6]
    old=ranked_choice(narrow,baseline,weights)
    if not rows or method=='old':return old
    if method=='wide':return ranked_choice(rows,baseline,weights)
    learned=fit_weights()
    return pick_with_fit(rows,baseline,old,learned,method)


def pick_with_fit(rows,baseline,old,learned,method):
    X=feature_rows(rows,baseline)
    best=int(np.argmax(X@learned))
    winner=rows[best]
    if method=='guarded':
        strong_tile=winner['atlas']>=.70 and winner['distinct_votes']>=2
        strong_repeat=winner['repeat_high']>=.24 and winner['edge']>=.12
        if not strong_tile and not strong_repeat:return old
    return winner['level']


@lru_cache(maxsize=1)
def fit_weights():
    return np.asarray(json.loads(Path(__file__).with_name('repeat_plane_weights.json').read_bytes())['weights'])


def place(image,version,parts,counts,camera,*,method='learned',atlas_path=None):
    from engine.atlas_plane_depth import _place,options
    def choose(part,mask,P,eye,rays,axis,baseline,rgb,atlas,scene,weights):
        rows=options(mask,P,eye,rays,axis,baseline,rgb,atlas,scene,radius=18,extra_levels=near_levels(eye[axis],baseline))
        levels=[r['level'] for r in rows]
        for r,(hi,lo) in zip(rows,repeat_scores(rgb,mask,P,eye,axis,levels),strict=True):
            r.update(repeat_high=hi,repeat_low=lo)
        return pick(rows,baseline,weights,method)
    return _place(image,version,parts,counts,camera,atlas_path,remove_occluded=True,choose_plane=choose)


def guarded(image,version,parts,counts,camera):
    return place(image,version,parts,counts,camera,method='guarded')


def repeat_scores(rgb, mask, P, eye, axis, levels):
    yy, xx = np.unravel_index(_sample(np.flatnonzero(mask), 600), mask.shape)
    # Native texel centers are sampled inside the observed surface.
    x = SAMPLE_STRIDE * xx + SAMPLE_STRIDE // 2
    y = SAMPLE_STRIDE * yy + SAMPLE_STRIDE // 2
    valid = (x < rgb.shape[1]) & (y < rgb.shape[0])
    x, y = x[valid], y[valid]
    if len(x) < 12:
        return [(0., 0.) for _ in levels]
    pixel = np.stack((x+.5, y+.5, np.ones(len(y))), axis=1)
    rays = pixel @ np.linalg.inv(P[:,:3]).T
    h,w = np.minimum(rgb.shape[:2], np.array(mask.shape)*SAMPLE_STRIDE)
    ref = rgb[y,x].astype(float)
    free = [j for j in range(3) if j != axis]
    output = []
    for level in levels:
        depth = np.divide(level-eye[axis], rays[:,axis],
                          out=np.zeros(len(rays)), where=abs(rays[:,axis])>1e-8)
        screen = depth[:,None]*pixel
        directions = []
        for d in free:
            # At a correct block pitch a one-block shift repeats more accurately
            # than either of two nearby noninteger displacements.
            hits, diffs = [], []
            for shift in (1., .8125, 1.1875):
                other = screen + shift*P[None,:,d]
                xy = np.floor(other[:,:2]/np.maximum(.1,other[:,2:])).astype(int)
                a,b = xy.T
                inside = (other[:,2]>1.) & (depth>1.) & (a>=0)&(a<w)&(b>=0)&(b<h)
                ix = np.flatnonzero(inside)
                inside[ix] &= mask[b[ix]//SAMPLE_STRIDE, a[ix]//SAMPLE_STRIDE]
                hits.append(inside)
                diffs.append(np.abs(rgb[b.clip(0,h-1),a.clip(0,w-1)].astype(float)-ref).mean(1))
            valid = np.logical_and.reduce(hits)
            if valid.sum() < 12:
                continue
            exact, offa, offb = [v[valid] for v in diffs]
            off = .5*(offa+offb)
            # An untextured segment supplies no evidence in either direction.
            gain = (off-exact) / (2.+off+exact)
            directions.append(float(gain.mean()) * min(1.,valid.sum()/48))
        output.append((max(directions,default=0.),min(directions,default=0.)))
    return output
