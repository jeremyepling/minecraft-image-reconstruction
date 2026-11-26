"""Freeze the ordinary image-only surface, count and camera inputs to visible placement."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile

from benchmark.texture_camera_eval import DATA, ROOT, STABLE, SPLITS, checked, digest, frames, read, save
from benchmark.texture_camera_placement import mask_sources
from engine.photo_edges import refine_surfaces
from engine.surface_model import _count_regions, DEFAULTS
from PIL import Image


CACHE = STABLE / 'runs/visible-depth-rounds'
ARCHIVE = ROOT / 'benchmark/reviews/texture-camera-01/scoring.zip'


def prepare(job):
    f, camera, mask, cache = job
    folder = DATA/'data'/f['split']/f['directory']
    rgb_path = checked(folder/'rgb.png',f['files']['rgb.png'])
    source = cache/'train-builds'/f['id']
    if f['split']=='train':
        manifest = read(source/'build-manifest.json')
        assert manifest['image_sha256']==f['files']['rgb.png'] and manifest['through']=='counts'
        parts = read(source/'surfaces.json')['planes']
        counts = read(source/'counts.json')
        provenance = {name:digest(source/name) for name in ('surfaces.json','counts.json','build-manifest.json')}
    else:
        mask_path, mask_sha = mask
        original = json.loads(gzip.decompress(checked(mask_path,mask_sha).read_bytes()))
        with Image.open(rgb_path) as src:
            rgb = src.convert('RGB')
        if f['split']=='regression':
            parts = original['edge_detail:native4-stable-counts']
        else:
            assert original['manifest']['image_sha256']==f['files']['rgb.png']
            parts, _ = refine_surfaces(rgb,original['planes'])
        sample_dir = cache/'samples'/f['id']
        sample_dir.mkdir(exist_ok=True,parents=True)
        counts = _count_regions(rgb,parts,'26.3',read(DEFAULTS)['block_count'],None,sample_dir)
        provenance = {'mask_sha256':mask_sha}
    output = dict(id=f['id'], split=f['split'], family=f.get('family',f['layout']), layout=f['layout'],
                  rgb_sha256=f['files']['rgb.png'], camera=camera, parts=parts, counts=counts, source=provenance)
    path = cache/'inputs'/(f['id']+'.json')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(output,separators=(',',':'),allow_nan=False),encoding='utf8')
    tmp.replace(path)
    return f['id']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=(*SPLITS,'all'),default='all')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--watch',action='store_true',help='Wait for ordinary training RGB builds to finish.')
    p.add_argument('--build-train',action='store_true',help='Generate any missing ordinary training outputs on this PC.')
    args=p.parse_args()
    (args.cache/'inputs').mkdir(parents=True,exist_ok=True)
    jobs=[]
    with ZipFile(ARCHIVE) as z:
        for split in SPLITS if args.split=='all' else [args.split]:
            rows=frames(DATA,split)
            base=f'camera/floor/r3-hardened/{split}/'
            cams={r['id']:r for r in json.loads(z.read(base+'cases.json'))}
            masks=list(mask_sources(split,rows)) if split!='train' else [None]*len(rows)
            for f,m in zip(rows,masks,strict=True):
                assert cams[f['id']]['rgb_sha256']==f['files']['rgb.png']
                jobs.append((f,cams[f['id']]['prediction'],m,args.cache))
    selected=[j for j in jobs if not (args.cache/'inputs'/(j[0]['id']+'.json')).exists()]
    if args.build_train:
        for frame,_,_,cache in selected:
            if frame['split']!='train':
                continue
            source=cache/'train-builds'/frame['id']
            if not (source/'build-manifest.json').is_file():
                image=DATA/'data'/'train'/frame['directory']/'rgb.png'
                subprocess.run(['pwsh','-NoProfile','-File',str(ROOT/'Invoke-WallRecovery.ps1'),str(image),
                                '-Through','counts','-Out',str(source)],check=True,cwd=ROOT)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while selected:
            ready=[j for j in selected if j[0]['split']!='train' or
                   (args.cache/'train-builds'/j[0]['id']/'build-manifest.json').exists()]
            for name in pool.map(prepare,ready):
                print(name,flush=True)
            done={j[0]['id'] for j in ready}
            selected=[j for j in selected if j[0]['id'] not in done]
            if selected and args.watch:
                time.sleep(3)
            else:
                break
    paths={j[0]['id']:args.cache/'inputs'/(j[0]['id']+'.json') for j in jobs}
    if all(f.exists() for f in paths.values()):
        save(args.cache/('inputs-'+args.split+'.json'),dict(source='651ef4f',camera_archive_sha256=digest(ARCHIVE),
             input_sha256={k:digest(v) for k,v in paths.items()}))


if __name__=='__main__':
    main()
