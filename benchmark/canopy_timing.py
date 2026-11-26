"""Time placement alone on the same frozen image-derived inputs, one method at a time."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

from PIL import Image
from benchmark.texture_camera_eval import DATA, STABLE, checked, read, save
from benchmark.visible_depth_score import init_candidate
from benchmark.visible_depth_transfer import OUT, rgb_path


def measure():
    from benchmark.visible_depth_score import predict
    key = 'shape-benchmark-26.3--train-canopy-09-right'
    native = next(f for f in read(OUT/'screen.json')['selected'] if f['id']==key)
    upstream = OUT/'upstream'/key
    photos = [(key,checked(rgb_path(native),native['files']['rgb.png']),native['camera'],
               read(upstream/'surfaces.json')['planes'],read(upstream/'counts.json'))]
    for split,key in [('validation','validation-cloister1-cool-high-right'),
                      ('regression','legacy-light_gray_concrete-holdout-galleries-plaster-low-left')]:
        entry=read(STABLE/'runs/visible-depth-rounds/inputs'/(key+'.json'))
        path=checked(DATA/'data'/split/'frames'/key/'rgb.png',entry['rgb_sha256'])
        photos.append((key,path,entry['camera'],entry['parts'],entry['counts']))
    times={}
    for key,path,camera,parts,counts in photos:
        with Image.open(path) as source:
            image=source.convert('RGB')
        start=perf_counter()
        cells,_=predict(image,'26.3',parts,counts,camera)
        times[key]=dict(seconds=perf_counter()-start,cells=len(cells))
    return times


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--current',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    pools = {name: ProcessPoolExecutor(max_workers=1,initializer=init_candidate,
                                       initargs=(path.resolve(),'engine.atlas_plane_depth:place_ranked'))
             for name,path in [('previous',args.previous),('current',args.current)]}
    times={name:[] for name in pools}
    try:
        for pool in pools.values():
            pool.submit(measure).result()
        for name in ('previous','current','current','previous'):
            times[name].append(pools[name].submit(measure).result())
    finally:
        for pool in pools.values():
            pool.shutdown()
    save(args.out,times)
