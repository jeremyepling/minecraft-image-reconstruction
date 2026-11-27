"""Scoring-only correct chart and deliberately false ground charts. Never imported by inference."""
from pathlib import Path
from PIL import Image
import numpy as np
from benchmark.camera_metrics import truth_camera
from benchmark.camera_native_floor import atlas_scan, patches
from engine.camera_atlas_floor import _atlas
from benchmark.expanded_baseline import ROOT, SOURCE, read, save

REVIEW=ROOT/'benchmark/reviews/camera-native-floor-inverse-01'
CONTROL=ROOT/'benchmark/reviews/camera-model-controls-01'
RUN=ROOT/'runs/camera-native-floor-inverse-01'


def entries(split):
    for row in read(CONTROL/'independent-matched-cases.json.gz'):
        if not row['camera_usable'] and row['split']==split:
            yield row


def diagnose(row):
    frame=SOURCE/row['dataset']/'data'/row['split']/'frames'/row['id']
    pose=read(frame/'camera.json')
    P=truth_camera(pose)
    P[:,3]+=P[:,1]*(1+read(frame/'frame.json')['ground_y'])
    inv=np.linalg.inv(P[:,[0,2,3]])
    with Image.open(frame/'rgb.png') as image:
        samples=patches(image)
    atlas=_atlas('26.3')
    # Work in a nearby integer gauge; wrong scale changes spacing across image.
    origin=np.floor((inv@samples[0][0][:,0])[:2]/(inv@samples[0][0][:,0])[2])
    inv[:2]-=origin[:,None]*inv[2]
    correct=atlas_scan(inv,samples,atlas,np.arange(len(atlas.labels)))
    ids=[atlas.labels.index(correct['tile'])]
    alternatives={}
    for ratio in (.5,.98,1.02,2.):
        chart=inv.copy()
        chart[:2]*=ratio
        alternatives[str(ratio)]=atlas_scan(chart,samples,atlas,ids)
    return dict(id=row['id'],reference_only=True,correct=correct,wrong_scale=alternatives)


if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(3) as pool:
        values=list(pool.map(diagnose,entries('train')))
    save(REVIEW/'train-reference-discrimination.json',values)
