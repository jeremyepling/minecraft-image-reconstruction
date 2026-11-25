"""Generate image-only proposals independently of camera and building labels."""
import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait as wait_for_jobs
from pathlib import Path
import hashlib
import json
from tempfile import NamedTemporaryFile
import time

from engine.plane_candidates import candidate
from engine.texel_grid import candidate as texel_candidate
from engine.reconstruction import geometry

METHODS=('baseline','partition','texel','common','remap','ground')


def predict(task):
    image,folder,method=task
    key=hashlib.sha256(image.read_bytes()).hexdigest()
    code=Path(__file__).resolve().parents[1]/'engine'
    files=['geometry.py','reconstruction.py','propose.py']
    if method!='baseline': files.append('plane_candidates.py')
    if method=='texel': files.append('texel_grid.py')
    if method in ('common','remap','ground'): files+=['texel_grid.py','remap.py','local_pitch.py']
    if method=='ground': files+=['depth_layers.py']
    fingerprint=hashlib.sha256(b''.join((code/f).read_bytes() for f in files)).hexdigest()
    output=folder/method/(image.stem+'.json')
    if output.is_file():
        try:
            old=json.loads(output.read_text())
        except json.JSONDecodeError:
            old={}
        if old.get('image_hash')==key and old.get('method_hash')==fingerprint: return
    if method=='baseline':
        _,planes,cells=geometry(image)
    elif method=='partition':
        planes,cells=candidate(image)
    elif method=='texel':
        planes,cells=texel_candidate(image)
    else:
        from PIL import Image
        from engine.remap import candidate as remap
        with Image.open(image) as source:
            planes,cells=remap(source,method!='common',method=='ground')
    output.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(mode='w', dir=output.parent, delete=False) as temp:
        temp_path=Path(temp.name)
    try:
        temp_path.write_text(json.dumps(dict(image_hash=key,method_hash=fingerprint,planes=planes,cells=cells),
                                        separators=(',',':')))
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)


def run(root,splits,workers,wait,methods,prediction_dir=None,capture_active=None):
    views=json.loads((root/'views.json').read_text())
    todo={v['id'] for v in views if v['split'] in splits and v['kind']!='adjacency'}
    if not todo:
        raise ValueError(f'No geometry views in {root} for splits {splits}')
    folder=prediction_dir or root/'geometry-predictions'
    pending=deque()
    running=set()
    deadline=time.monotonic()+wait
    with ProcessPoolExecutor(workers) as pool:
        while todo or pending or running:
            # Check the producer before the scan so its final published frame is seen.
            active=capture_active() if capture_active else time.monotonic()<deadline
            for view in sorted(todo):
                stem=root/'captures'/view
                # The driver closes the PNG before it publishes the pose file.
                if stem.with_suffix('.json').is_file() and stem.with_suffix('.png').is_file():
                    pending.extend((stem.with_suffix('.png'),folder,method) for method in methods)
                    todo.remove(view)
            if todo and not active and not (pending or running):
                raise ValueError(f'{len(todo)} expected screenshots are missing, including {min(todo)}')
            while pending and len(running)<workers:
                running.add(pool.submit(predict,pending.popleft()))
            if running:
                done,running=wait_for_jobs(running,timeout=0.5,return_when=FIRST_COMPLETED)
                for job in done:
                    job.result()
            elif todo:
                time.sleep(0.5)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--splits',nargs='+',choices=['train','dev','test'],required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--wait',type=int,default=0)
    p.add_argument('--prediction-dir',type=Path,help='Private candidate output (default: ROOT/geometry-predictions)')
    args=p.parse_args();run(args.root,args.splits,args.workers,args.wait,args.methods,args.prediction_dir)
