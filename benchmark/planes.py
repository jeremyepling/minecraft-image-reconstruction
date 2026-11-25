"""Generate image-only proposals independently of camera and building labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
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
        old=json.loads(output.read_text())
        if old['image_hash']==key and old['method_hash']==fingerprint: return
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
    output.write_text(json.dumps(dict(image_hash=key,method_hash=fingerprint,planes=planes,cells=cells),
                                 separators=(',',':')))


def run(root,splits,workers,wait,methods):
    views=json.loads((root/'views.json').read_text())
    todo={v['id'] for v in views if v['split'] in splits and v['kind']!='adjacency'}
    deadline=time.monotonic()+wait
    with ProcessPoolExecutor(workers) as pool:
        while todo:
            ready=sorted(v for v in todo if (root/'captures'/f'{v}.json').is_file())
            if ready:
                jobs=[(root/'captures'/f'{v}.png',root/'geometry-predictions',m) for v in ready for m in methods]
                list(pool.map(predict,jobs,chunksize=1))
                todo.difference_update(ready)
            elif time.monotonic()>=deadline:
                raise ValueError(f'{len(todo)} expected screenshots are missing')
            else: time.sleep(2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--splits',nargs='+',choices=['train','dev','test'],required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--wait',type=int,default=0)
    args=p.parse_args();run(args.root,args.splits,args.workers,args.wait,args.methods)
