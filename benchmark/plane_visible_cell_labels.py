"""Scoring-only labels giving each visible block face one vote across its pixels."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from benchmark.atlas_plane_round import CACHE, DATA, OUT, read
from benchmark.camera_metrics import registration
from engine.visible_raster import SAMPLE_STRIDE, observed_support


def label(frame):
    inp=read(CACHE/(frame['id']+'.json'))
    observed={r['id']:r for r in inp['parts']}
    root=DATA/'train'/'frames'/frame['id']
    pose=read(root/'camera.json')
    w,h=pose['image_size']
    ids=np.load(root/'labels.npz')['face_id'][2::4,2::4]
    all_counts=np.bincount(ids.ravel())
    faces=read(root/'labels.json')['faces']
    T=registration(np.asarray(inp['camera']['P']),pose)
    planes={}
    for face in faces:
        axis=int(np.argmax(abs(face['normal_world'] @ T[:3,:3])))
        p=(np.asarray(face['corners_world'][0])-T[:3,3]) @ T[:3,:3]
        planes[face['id']]=(axis,int(p[axis]))
    shape=((h+SAMPLE_STRIDE-1)//SAMPLE_STRIDE,(w+SAMPLE_STRIDE-1)//SAMPLE_STRIDE)
    for r in frame['rows']:
        mask=observed_support(observed[r['id']]['support'],shape,(w,h))
        values,counts=np.unique(ids[mask[:ids.shape[0],:ids.shape[1]]],return_counts=True)
        labels=defaultdict(float)
        for v,n in zip(values,counts):
            if v and n>=4 and planes[v][0]==r['axis']:
                labels[planes[v][1]]+=n/all_counts[v]
        r['truth_cells']=labels
    dest=OUT/'cell-labels'/frame['split']/(frame['id']+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(frame))


if __name__=='__main__':
    for f in sorted((OUT/'wide-features'/'train').glob('*.json')):
        if not (OUT/'cell-labels'/'train'/f.name).exists():
            label(read(f))
