"""Fit a small conditional ranker on masked visible face labels, grouped by family."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.floor_depth_round import measurements
from benchmark.plane_offset_ranker import fit_examples
from benchmark.texture_camera_eval import DATA, STABLE
from engine import atlas_plane_depth as atlas_depth
from engine.metric_plane_decision import combined_features
from engine.plane_offset_features import NAMES
from engine.visible_placement import place_with_camera


def read(path):
    return json.loads(path.read_bytes())


def train_frames(folder,cache):
    result=[]
    cache.mkdir(parents=True,exist_ok=True)
    for path in sorted(folder.glob('train-*.json')):
        frame=read(path)
        source=read(STABLE/'runs/visible-depth-rounds/inputs'/path.name)
        assert frame['rgb_sha256']==source['rgb_sha256']
        dest=cache/path.name
        if dest.exists():
            meas=read(dest)
        else:
            with Image.open(DATA/'data/train/frames'/frame['id']/'rgb.png') as photo:
                meas=measurements(photo.convert('RGB'),source['parts'],source['camera'],'inverse')
            dest.write_text(json.dumps(meas))
        for row in frame['rows']:
            row['measurement']=meas.get(row['id'])
        result.append(frame)
    return result


def training_items(frames,include_depth):
    items=[]
    for f in frames:
        for r in f['rows']:
            rows=r['choices']
            if len(rows)<2:continue
            votes=np.array([r['truth'].get(str(o['level']),0) for o in rows],float)
            if votes.sum()<max(24,.2*r['n']):continue
            X=combined_features(rows,r['baseline'],r['measurement'])
            if not include_depth:X=X[:,:len(NAMES)]
            items.append((X,votes/votes.sum(),min(votes.sum()**.5,60.)))
    return items


def accuracy(frames,weights):
    correct=0
    for f in frames:
        for r in f['rows']:
            rows=r['choices']
            if not rows: continue
            X=combined_features(rows,r['baseline'],r['measurement'])
            selected=rows[int(np.argmax(X[:,:len(weights)]@weights))]['level']
            correct+=r['truth'].get(str(selected),0)
    return correct


@lru_cache(maxsize=4)
def load_weights(path,fold):
    artifact=read(Path(path))
    return artifact['folds'][fold] if fold else artifact['weights']


def predict(image,version,parts,counts,camera):
    weights=load_weights(os.environ['MCIR_METRIC_PLANE_FIT'],os.environ.get('MCIR_EXCLUDE_FAMILY'))
    measurements_by_id=measurements(image,parts,camera,'inverse')
    rgb=np.asarray(image.convert('RGB'))
    atlas=atlas_depth._atlas(version,None)
    def select(part,mask,P,eye,rays,axis,baseline,scene_distance):
        rows=atlas_depth.options(mask,P,eye,rays,axis,baseline,rgb,atlas,scene_distance)
        if not rows:return baseline
        X=combined_features(rows,baseline,measurements_by_id.get(part['id']))
        return rows[int(np.argmax(X@weights))]['level']
    return place_with_camera(image,version,parts,counts,camera,select)


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();frames=train_frames(a.features,a.cache)
    families=sorted({f['family'] for f in frames})
    ablations,folds={},{}
    for family in families:
        train=[f for f in frames if f['family']!=family]
        check=[f for f in frames if f['family']==family]
        rgb=fit_examples(training_items(train,False))
        combined=fit_examples(training_items(train,True))
        folds[family]=combined.tolist()
        ablations[family]={'rgb':accuracy(check,rgb),'with_depth':accuracy(check,combined)}
    w=fit_examples(training_items(frames,True))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(features=[*NAMES,'metric_fit','metric_error','low_spread_fit'],
        frames=[f['id'] for f in frames],weights=w.tolist(),folds=folds,grouped=ablations),indent=2)+'\n')
    print(len(frames),ablations)


if __name__=='__main__':main()
