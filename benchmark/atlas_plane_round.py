"""Measure integer offset choices inside frozen, single-image surface masks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from engine.atlas_plane_depth import options, choose
from engine.camera_atlas_floor import _atlas
from engine.visible_placement import _estimate_plane_cells
from engine.visible_raster import observed_support, SAMPLE_STRIDE


STABLE=Path(__file__).resolve().parents[3]/'minecraft-image-reconstruction'
CACHE=STABLE/'runs/visible-depth-rounds/inputs'
DATA=STABLE/'runs/visible-builds-v1/data'
OUT=Path(__file__).resolve().parents[1]/'runs/plane-depth'


def read(path):
    return json.loads(path.read_bytes())


def extract(source):
    frame=read(source)
    key=frame['id']
    folder=DATA/frame['split']/'frames'/key
    start=time.perf_counter()
    rows=[]
    camera=frame['camera']['P']
    if camera is not None:
        with Image.open(folder/'rgb.png') as src:
            image=src.convert('RGB')
        rgb=np.asarray(image)
        atlas=_atlas('26.3')
        if atlas is None:
            raise ValueError('Prepare the calibrated 26.3 atlas')
        P=np.asarray(camera)
        P=P/np.linalg.norm(P[2,:3])
        eye=-np.linalg.solve(P[:,:3],P[:,3])
        shape=tuple((n+SAMPLE_STRIDE-1)//SAMPLE_STRIDE for n in image.size[::-1])
        masks=[observed_support(p['support'],shape,image.size) if p.get('support') else np.zeros(shape,bool)
               for p in frame['parts']]
        def collect(part,mask,P,eye,rays,axis,baseline):
            rows.append(dict(id=part['id'],axis=axis,baseline=baseline,n=int(mask.sum()),
                             choices=options(mask,P,eye,rays,axis,baseline,rgb,atlas)))
            return baseline
        if frame['parts']:
            _estimate_plane_cells(image,frame['parts'],masks,P,eye,collect)
        elapsed=time.perf_counter()-start
        # Scoring only from this point: which visible face plane owns each mask sample?
        T=registration(P,read(folder/'camera.json'))
        faces=read(folder/'labels.json')['faces']
        ids=np.load(folder/'labels.npz')['face_id'][2::4,2::4]
        lookup={}
        for f in faces:
            axis=int(np.argmax(abs(np.asarray(f['normal_world']) @ T[:3,:3])))
            p=(np.asarray(f['corners_world'][0])-T[:3,3]) @ T[:3,:3]
            lookup[f['id']]=(axis,int(p[axis]))
        by_id={p['id']:mask for p,mask in zip(frame['parts'],masks)}
        for r in rows:
            mask=by_id[r['id']][:ids.shape[0],:ids.shape[1]]
            values,counts=np.unique(ids[mask],return_counts=True)
            truth={}
            for v,n in zip(values,counts):
                if v and lookup[v][0]==r['axis']:
                    level=lookup[v][1]
                    truth[level]=truth.get(level,0)+int(n)
            r['truth']=truth
    else:
        elapsed=0.
    dest=OUT/'features'/frame['split']/(key+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(id=key,split=frame['split'],family=frame['family'],
        rgb_sha256=frame['rgb_sha256'],seconds=elapsed,rows=rows),allow_nan=False))
    tmp.replace(dest)
    return key


def metrics(files):
    totals=dict.fromkeys(['samples','oracle','baseline','boundary','edge','atlas'],0)
    for f in files:
        for r in read(f)['rows']:
            q=r['choices']
            if not q:
                continue
            truth=r['truth']
            votes=lambda k:truth.get(str(k),0)
            totals['samples']+=r['n']
            totals['baseline']+=votes(r['baseline'])
            totals['oracle']+=max(votes(row['level']) for row in q)
            for method in ('boundary','edge','atlas'):
                totals[method]+=votes(choose(q,r['baseline'],method))
    return totals


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',default='train')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--watch',type=int,default=0)
    a=p.parse_args()
    prefix='legacy' if a.split=='regression' else a.split
    out=OUT/'features'/a.split
    with ProcessPoolExecutor(a.workers) as pool:
        while True:
            jobs=[f for f in sorted(CACHE.glob(prefix+'-*.json')) if not (out/f.name).exists()]
            for key in pool.map(extract,jobs):
                print(key,flush=True)
            completed=list(out.glob('*.json'))
            print(len(completed),metrics(completed),flush=True)
            if len(completed)>=a.watch:
                break
            time.sleep(3)


if __name__=='__main__':
    main()
