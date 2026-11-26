"""Adjust near-coincident predicted boundaries using registered dense Planarity segments."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, read
from benchmark.wall_join_test import checked
from engine.attached_strip import neighborhood
from engine.plane_consensus import row_runs
from engine.surface_perimeter import _native_alpha


SETTINGS = dict(minimum_region_iou=.70, radius_px=3, additions='Only inside previous visible proposal union')
FIELD_DIR = ROOT/'benchmark/reviews/plane-detail-fields-01'
METHODS = {'stride4': (4,1600), 'stride2': (2,1600), 'raw': (1,1600)}


@lru_cache(maxsize=3)
def field_manifest(folder):
    return read(folder/'manifest.json')


def source_field(sha, tokens=1600):
    folder = FIELD_DIR if tokens==1600 else ROOT/f'benchmark/reviews/plane-detail-fields-{tokens}'
    name = sha+'.png'
    path = checked(folder/name,field_manifest(folder)['artifacts'][name])
    with Image.open(path) as source:
        return np.asarray(source).copy()


def register(field, size, stride):
    # A stride-four cell contains pixels 0..3 and samples pixel 2's center at 2.5.
    idx = []
    for extent, raw_extent in zip(size, field.shape[::-1]):
        centers = np.minimum((np.arange(extent)//stride)*stride+stride//2+.5, extent-.5)
        idx.append((centers * raw_extent / extent).astype(int))
    return field[np.ix_(idx[1],idx[0])]


def prepare(parts, size):
    native = [_native_alpha(part['support'],size) for part in parts]
    union = np.zeros(size[::-1],bool)
    for alpha, (l,t,r,b) in native:
        union[t:b,l:r] |= np.asarray(alpha,bool)
    return native, union


def predict(photo, parts, frame_sha, setting):
    stride,tokens = METHODS[setting]
    labels = register(source_field(frame_sha,tokens),photo.size,stride)
    areas = np.bincount(labels.ravel())
    radius = SETTINGS['radius_px']
    native, union = prepare(parts, photo.size)
    updated = []
    for part,(alpha,(l,t,r,b)) in zip(parts,native):
        owned = np.asarray(alpha,bool)
        hits = np.bincount(labels[t:b,l:r][owned],minlength=len(areas))
        hits[0] = 0
        seed = hits.argmax()
        if not seed or hits[seed]/(areas[seed]+owned.sum()-hits[seed]) < SETTINGS['minimum_region_iou']:
            updated.append(part)
            continue
        x,y = max(0,l-radius),max(0,t-radius)
        right,bottom = min(photo.width,r+radius),min(photo.height,b+radius)
        box = np.s_[y:bottom,x:right]
        parent = np.zeros((bottom-y,right-x),bool)
        parent[t-y:b-y,l-x:r-x] = owned
        plane = labels[box]==seed
        around = neighborhood(parent,radius)
        near_plane = neighborhood(plane,radius)
        full = (2*radius+1)**2
        band = (around>0)&(around<full)&(near_plane>0)&(near_plane<full)
        revised = np.where(band,plane & union[box],parent)
        if not revised.any() or np.array_equal(revised,parent):
            updated.append(part)
            continue
        runs = (np.asarray(row_runs(revised))+[y,x,x]).tolist()
        result = {k:v for k,v in part.items() if k not in ('outline','source_crop','context_crop','sampling_chart')}
        result.update(support=dict(size=list(photo.size),runs=runs),support_area=int(revised.sum()))
        updated.append(result)
    return updated
