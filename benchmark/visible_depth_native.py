"""Prepare the independent six-layout, 12-image supported camera suite for placement."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from benchmark.texture_camera_eval import checked, digest, read, save, STABLE, ROOT
from engine.surface_model import _count_regions, DEFAULTS
from training.visible_labels import visible_labels


BUILDS = STABLE.parent/'workspace/builds'
CAMERAS = ROOT/'benchmark/reviews/texture-camera-01/scoring.zip'
GEOMETRY = ROOT/'benchmark/reviews/combined-default-01/evidence.zip'
CACHE = STABLE/'runs/visible-depth-rounds'


def rows():
    with ZipFile(CAMERAS) as archive:
        index=json.loads(archive.read('native-inputs/index.json'))
        cameras={r['rgb_sha256']:r for r in json.loads(archive.read('native70/fb9d5b4/cases.json'))}
    selected=[]
    for row in index:
        result=cameras[row['sha256']]
        if not result['candidate']['metric']:
            continue
        family=result['candidate']['layout']
        selected.append(dict(id=row['id'],split='native',layout=family,family=family,
                        directory=row['fixture_directory'], camera=result['candidate']['prediction'],
                        files={'rgb.png':row['sha256'],'camera.json':result['pose_sha256'],
                               'world-readback.json':result['states_sha256']}))
    assert len(selected)==12
    return selected


def rgb_path(frame,fixtures=BUILDS):
    return fixtures/frame['directory']/'captures'/(frame['id']+'.png')


def truth(frame,fixtures=BUILDS):
    path=rgb_path(frame,fixtures)
    pose=read(checked(path.with_suffix('.json'),frame['files']['camera.json']))
    raw=read(checked(path.with_name(frame['id']+'-blocks.json'),frame['files']['world-readback.json']))
    states={tuple(b[k] for k in 'xyz'):b for b in raw}
    visible,_,_=visible_labels(pose,states,(0,0,0))
    target={tuple(b['xyz_world']):b['block_id'] for b in visible['blocks']}
    occupied={p for p,b in states.items() if b['block_id'].removeprefix('minecraft:')!='air'}
    return pose,target,occupied


def prepare(job):
    f, cache=job
    key=f['id']
    path=checked(rgb_path(f),f['files']['rgb.png'])
    with Image.open(path) as opened:
        rgb=opened.convert('RGB')
    source=BUILDS/'seam-review/promoted-inputs'
    frozen=read(source/'manifest.json')[key]
    assert frozen['rgb']==f['files']['rgb.png']
    with ZipFile(GEOMETRY) as archive:
        scored=json.loads(archive.read(f'geometry/frames/{key}/scores.json'))
        assert scored['masks_sha256']==frozen['masks'] and scored['image_sha256']==frozen['rgb']
    parts=json.loads(gzip.decompress(checked(source/(key+'.json.gz'),frozen['masks']).read_bytes()))
    sample=cache/'samples'/key
    sample.mkdir(exist_ok=True,parents=True)
    counts=_count_regions(rgb,parts,'26.3',read(DEFAULTS)['block_count'],None,sample)
    target=cache/'inputs'/(key+'.json')
    data=dict(id=key,split='native',family=f['family'],layout=f['layout'],rgb_sha256=frozen['rgb'],
              camera=f['camera'],parts=parts,counts=counts,source={'mask_sha256':frozen['masks']})
    target.write_text(json.dumps(data,separators=(',',':'),allow_nan=False),encoding='utf8')
    return key,digest(target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    (args.cache/'inputs').mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fingerprints=dict(pool.map(prepare,[(f,args.cache) for f in rows()]))
    save(args.cache/'inputs-native.json',dict(source='651ef4f',camera_archive_sha256=digest(CAMERAS),
         input_sha256=fingerprints))


if __name__=='__main__':
    main()
