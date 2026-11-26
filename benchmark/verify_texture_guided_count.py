"""Verify ordinary Block Counter records against the frozen, approved repeat-only results."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.detail_transfer import ARCHIVE as FAMILIES, member, shards
from benchmark.normal_round import groups
from benchmark.texture_count_round import FINDERS
from benchmark.wall_join_test import checked, read_gzip
from engine.photo_edges import refine_surfaces
from engine.surface_model import DEFAULTS, _count_regions
from engine.surface_sample import sample_surface
from engine.visible_placement import _charts
from runtime.mode import image_only, mark_image_only


REVIEWS=ROOT/'benchmark/reviews'
METHOD='texture-guided-repeat'
EXTERIOR=REVIEWS/'phase-corners-01'
TRANSFER=REVIEWS/'count-transfer-check-01'


def cases_at(source):
    sha=read(source/'manifest.json')['artifacts']['cases.json']
    return read(checked(source/'cases.json',sha))


def compare(photo,parts,records,frozen,row,mode,folder):
    estimates={s['proposal']:s for s in row['surfaces']}
    charts=_charts(photo,'26.3',parts,{'surfaces':records})
    fallback=changed=0
    for part,got,old,placement in zip(parts,records,frozen,charts,strict=True):
        pid=part['id']
        if pid!=got['id'] or pid!=old['proposal'] or pid!=placement['id']:
            raise ValueError('Ordered proposals differ')
        expected=old['methods']['phase']['evidence'].get('self_repeat')
        if expected:
            fallback+=1
        result=expected or old['methods']['current']
        original=sample_surface(photo,part)
        quad=original['quad'] if original else None
        selected=result.get('quad',quad)
        blocks=estimates[pid]['guess'][mode]
        if (got['counts']!=result['counts'] or got['phase']!=result['phase'] or got.get('quad')!=selected
                or got['dimensions']['blocks']!=blocks):
            raise ValueError(f"Count/phase/chart/dimensions differ on {row['key']} / {pid}")
        if placement['quad']!=selected or placement['pitch']['counts']!=(result['counts'] if None not in result['counts'] else None):
            raise ValueError(f'Placement received another chart or count on {pid}')
        if expected:
            changed+=int(quad!=selected)
            sample=sample_surface(photo,dict(part,quad=selected))
            with Image.open(folder/got['sample']) as emitted:
                image=np.array(emitted.convert('RGBA'))
            if not (np.array_equal(image[:,:,:3],sample['rgb']) and np.array_equal(image[:,:,3],sample['mask']*255)):
                raise ValueError('Written PNG differs from approved masked crop on '+pid)
    scores=row['scores'][mode]
    return dict(key=row['key'],sha256=row['sha256'],proposals=len(parts),fallback_surfaces=fallback,
                changed_crop_quads=changed,**{k:scores[k] for k in ('eligible','matched','correct','incorrect','missing','unmatched_returned')})


def exterior_data(photo_path,row,pred_hash):
    source,method=FINDERS['photo']
    with Image.open(checked(photo_path,row['sha256'])) as image:
        photo=image.convert('RGB')
    return photo,read_gzip(checked(source/row['predictions'],pred_hash))[method]


def process(job):
    row,location,frame,pred_hash,approved_hash=job
    if frame is None:
        source,mode=EXTERIOR,'self_repeat'
        photo,parts=exterior_data(location,row,pred_hash)
    else:
        source,mode=TRANSFER,'phase_repeat'
        data=read_gzip(checked(FAMILIES/'ordinary'/(row['key']+'.json.gz'),pred_hash))
        with ZipFile(location) as archive:
            photo=Image.open(io.BytesIO(member(archive,frame,'rgb.png'))).convert('RGB')
        parts,_=refine_surfaces(photo,data['planes'])
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)
        records=_count_regions(photo,parts,'26.3',read(DEFAULTS)['block_count'],None,folder)['surfaces']
        approved=read_gzip(checked(source/row['predictions'],approved_hash))
        return compare(photo,parts,records,approved,row,mode,folder)


@image_only()
def run(dataset,out,workers):
    if read(DEFAULTS)['block_count']!=METHOD:
        raise ValueError('The repeat-only fallback must be the production default')
    ext=cases_at(EXTERIOR)
    transfer={r['key']:r for r in cases_at(TRANSFER)}
    captures={c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
    source=FINDERS['photo'][0]
    source_hashes=read(source/'manifest.json')['artifacts']
    photo_hashes=read(EXTERIOR/'manifest.json')['artifacts']
    transfer_hashes=read(TRANSFER/'manifest.json')['artifacts']
    ordinary_hashes=read(FAMILIES/'manifest.json')['artifacts']
    jobs=[(r,captures[r['key']].path('image'),None,source_hashes[r['predictions']],photo_hashes[r['predictions']]) for r in ext]
    for split,path,shard in shards(dataset):
        with ZipFile(path) as z:
            frames=[json.loads(line) for line in z.read('frames.jsonl').splitlines()]
        for frame in frames:
            r=transfer[frame['id']]
            if r['sha256']!=frame['image_sha256']:
                raise ValueError('Transfer image changed')
            jobs.append((r,path,frame,ordinary_hashes['ordinary/'+frame['id']+'.json.gz'],transfer_hashes[r['predictions']]))
    if len(jobs)!=80 or len({job[0]['key'] for job in jobs})!=80 or len(ext)!=56:
        raise ValueError('Fixture inventory changed')
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        records=list(pool.map(process,jobs))
    scores={}
    for name,subset in [('exterior56',records[:len(ext)]),('additional24',records[len(ext):])]:
        scores[name]={k:sum(r[k] for r in subset) for k in subset[0] if k not in ('key','sha256')}
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,dict(code=revision(),method=METHOD,all_counts_phases_quads_dimensions_equal=True,
                  written_fallback_sample_pixels_equal=True,placement_chart_inputs_equal=True,
                  settings=read(DEFAULTS),sources={p.name:digest(p/'manifest.json') for p in (source,EXTERIOR,TRANSFER,FAMILIES)},
                  scores=scores,frames=records))
    print(scores)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--workers',default=4,type=int)
    args=parser.parse_args()
    run(args.dataset,args.out,args.workers)
