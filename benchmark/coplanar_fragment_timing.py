"""Time image-only mask decisions and rebuilding their charts, excluding inference and placement."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark import coplanar_fragment_join as experiment, framed_depth_eval as framed
from benchmark.coplanar_fragment_audit import CONTROL,REVIEW
from benchmark.coplanar_fragment_report import index
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import read,save,checked,digest
from engine.diagonal_texture_join import refine


def measure(job):
    split,frame,prior,record=job
    key=frame['id']
    if split=='shelter_gate':
        root=Path('../framed-depth-transfer/runs/transfer-input')
        f=root/frame['directory']/'captures'/(frame['source_id']+'.png')
        with Image.open(checked(f,frame['files']['rgb.png'])) as src:
            photo=src.convert('RGB')
        upstream=root/'upstream'/key
        parts=read(checked(upstream/'surfaces.json',record['surface_sha256']))['planes']
        counts=read(checked(upstream/'counts.json',record['counts_sha256']))
        camera=frame['camera']
        with ZipFile(Path('benchmark/reviews/visible-error-budget-01/comparisons.zip')) as z:
            content=z.read('production/transfer/'+key+'/reconstruction.zip')
        from hashlib import sha256
        assert sha256(content).hexdigest()==record['control_pack_sha256']
        cells=set(decoded_pack(BytesIO(content)))
    else:
        photo,parts,counts,camera=framed.photo(split,frame,prior)
        cells=set(decoded_pack(checked(CONTROL/split/key/'reconstruction.zip',record['pack_sha256'])))
    elapsed={}
    for name in ('photo','diagonal','transition'):
        with patch.object(experiment,'rebuild',lambda *args:(cells,{})):
            start=perf_counter()
            if name=='diagonal':
                _,groups=refine(photo,parts,camera)
            else:
                fn=experiment.photograph if name=='photo' else experiment.material_transition
                _,result=fn(photo,parts,counts,camera,cells)
                groups=result['groups']
            elapsed[name]={'seconds':perf_counter()-start,'groups':groups}
    return dict(split=split,id=key,methods=elapsed)


def run(args):
    control=index(CONTROL)
    jobs=[(s,f,h,control[f['id']]) for s,f,h in framed.cases(framed.SIZES)]
    extra=index(args.runs/'transfer-photo')
    for frame in read(Path('../framed-depth-transfer/runs/transfer-input/screen.json'))['selected']:
        jobs.append(('shelter_gate',frame,None,extra[frame['id']]))
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(measure,jobs))
    times={}
    for name in rows[0]['methods']:
        variants={k:v for folder in (name,'transfer-'+name) for k,v in index(args.runs/folder).items()}
        assert all(variants[r['id']]['status']['groups']==r['methods'][name]['groups'] for r in rows)
        seconds=[r['methods'][name]['seconds'] for r in rows]
        times[name]=dict(median=np.median(seconds),p95=np.percentile(seconds,95),maximum=max(seconds))
    save(REVIEW/'timing.json',dict(frames=len(rows),workers=args.workers,engine_sha256=digest(Path('engine/diagonal_texture_join.py')),
        scope='wall clock inside each worker with original image, masks, camera and prior already in RAM',seconds=times,rows=rows))
    print(times)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--runs',type=Path,default=Path('runs'))
    run(parser.parse_args())
