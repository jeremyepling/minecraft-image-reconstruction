"""Diagnostic only: supply recorded camera calibration to the previous placer, without supplying geometry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from benchmark import visible_depth_score as scorer, visible_depth_transfer as transfer
from benchmark.camera_metrics import truth_camera, registration
from benchmark.mask_blocks import _write_read
from benchmark.roof_height_transfer_panel import SELECTION
from benchmark.texture_camera_eval import checked, digest, read, save


def prepare(job):
    f,data,out,sources,assets=job
    key=f['id']
    src=data/'upstream'/key
    # Only the recorded pose enters this labeled diagnostic. The local origin uses camera coordinates.
    rgb=sources/f['directory']/'captures'/(f['source_id']+'.png')
    pose=read(checked(rgb.with_suffix('.json'),f['files']['camera.json']))
    transform=np.eye(4)
    transform[:3,3]=[round(pose['eye_world'][0]),65,round(pose['eye_world'][2])]
    P=truth_camera(pose)@transform
    assert np.allclose(registration(P,pose),transform)
    camera=dict(P=P.tolist(),scale_status='supported',camera_source='supplied-native-pose',
                reason='Diagnostic with the recorded camera pose and fixed ground level')
    parts=read(checked(src/'surfaces.json',assets['surfaces.json']))['planes']
    counts=read(checked(src/'counts.json',assets['counts.json']))
    with Image.open(checked(rgb,f['files']['rgb.png'])) as im:
        cells,_=scorer.predict(im.convert('RGB'),'26.3',parts,counts,camera)
    target=out/key
    target.mkdir(parents=True,exist_ok=True)
    _write_read(target/'reconstruction.zip',cells)
    save(target/'camera.json',camera)
    return dict(id=key,camera_sha256=digest(target/'camera.json'),pack_sha256=digest(target/'reconstruction.zip'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--selection',type=Path,default=SELECTION)
    p.add_argument('--previous-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    root=a.previous_root.resolve()
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    assert rev.startswith('54c9fee')
    base=read(a.data/'builds.json')
    frames=read(checked(a.selection,base['selection_sha256']))['selected']
    index={r['id']:r['files'] for r in base['rows']}
    sources=a.data/'sources' if (a.data/'sources').is_dir() else transfer.BUILDS
    with ProcessPoolExecutor(max_workers=a.workers,initializer=scorer.init_candidate,
                             initargs=(root,'engine.atlas_plane_depth:place_ranked')) as pool:
        rows=list(pool.map(prepare,[(f,a.data,a.out,sources,index[f['id']]) for f in frames]))
    save(a.out/'index.json',dict(diagnostic='recorded native calibration supplied; image-derived masks and counts',
        prior_revision=rev,builds_sha256=digest(a.data/'builds.json'),rows=rows))
