"""Fit vertical face acceptance using only the 72 framed training images, grouped by family."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import nearby_face_trial as io
from benchmark.texture_camera_eval import STABLE, read, save
from benchmark.thin_face_fit import fit
from benchmark.vertical_atlas_trial import CURRENT
from engine.camera_atlas_floor import _atlas
from engine import vertical_atlas_faces as faces


def extract(job):
    split,frame,source,_,_=job
    prior=next(r for r in read(CURRENT/split/'cases.json') if r['id']==frame['id'])
    j=split,frame,source,CURRENT/split,prior
    image,parts,camera,cells=io.load_prediction(j)
    _,rows=faces.observations(image,parts,camera,cells,_atlas('26.3',STABLE/'runs/26.3/atlas'))
    transform,truth,occupied=io.labels(j,camera['P'])
    samples=[]
    for (cell,axis),r in rows.items():
        if r['texels']<10 or r['phase']<.04 or r['depth']<.015:
            continue
        xyz=tuple(np.floor((np.asarray(cell)+.5)@transform[:3,:3].T+transform[:3,3]).astype(int))
        samples.append((faces.vector(r),xyz in truth,xyz in occupied and xyz not in truth))
    return dict(id=frame['id'],family=frame['family'],samples=samples)


def train(records):
    X, y, hidden=zip(*(s for row in records for s in row['samples']))
    model=fit(np.asarray(X),np.asarray(y,float))
    model['features']=list(faces.FEATURES)
    return model


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=5)
    opts=cli.parse_args()
    opts.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=opts.workers) as pool:
        rows=list(pool.map(extract,io.cases('train')))
    families=sorted({r['family'] for r in rows})
    for family in families:
        save(opts.out/(family+'.json'),train([r for r in rows if r['family']!=family]))
    save(faces.WEIGHTS,train(rows))
    save(opts.out/'membership.json',[dict(id=r['id'],family=r['family'],candidates=len(r['samples'])) for r in rows])
