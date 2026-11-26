"""Compare local patch extrapolation to the frozen texture-guided fallback on 56 + 24 views."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT,digest,read,revision,save
from benchmark.count_metrics import reference_planes
from benchmark.detail_transfer import ARCHIVE,member,shards
from benchmark.normal_round import groups
from benchmark.surface_reference import recorded
from benchmark.surface_metrics import truth
from benchmark.texture_count_round import FINDERS,summarize
from benchmark.wall_join_test import checked,read_gzip
from benchmark.within_wall_grid import predict_surface,SETTINGS
from engine import photo_edges
from engine.geometry import project,transform
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from runtime.mode import mark_image_only,image_only


REVIEW=ROOT/'benchmark/reviews'
PHASE=REVIEW/'phase-corners-01'
EXTRA=REVIEW/'count-transfer-check-01'
OUTLINES,METHOD=FINDERS['photo']


def pixel_materials(ref,rid,faces,sample,patch):
    ids=ref['id_map']
    results={}
    for key in ('donor_box','far_box'):
        x,y,r,b=patch[key]
        yy,xx=np.nonzero(sample['mask'][y:b:2,x:r:2])
        uv=(np.stack([x+2*xx+.5,y+2*yy+.5],axis=1))/patch['chart_wh']
        pixels=project(transform(sample['quad']),uv)
        size=np.array(ref['image_size'])
        coords=np.clip((pixels*np.array(ids.shape[::-1])/size).astype(int),0,np.array(ids.shape[::-1])-1)
        visible=pixels[(ids[coords[:,1],coords[:,0]]==rid)&np.all((pixels>=0)&(pixels<size),axis=1)]
        cells=np.floor(project(np.linalg.inv(ref['charts'][rid]),visible)).astype(int)
        results[key]=dict(Counter(faces[tuple(c)]['block_id'] for c in cells if tuple(c) in faces))
    return results


def evaluate(job):
    cap,archive,frame,row,hashes=job
    if cap:
        with Image.open(checked(cap.path('image'),row['sha256'])) as img:
            photo=img.convert('RGB')
        parts=read_gzip(checked(OUTLINES/row['predictions'],hashes['outlines']))[METHOD]
        source=PHASE
    else:
        with ZipFile(archive) as z:
            photo=Image.open(io.BytesIO(member(z,frame,'rgb.png'))).convert('RGB')
        ordinary=read_gzip(checked(ARCHIVE/'ordinary'/(frame['id']+'.json.gz'),hashes['ordinary']))
        parts,_=photo_edges.refine_surfaces(photo,ordinary['planes'])
        source=EXTRA
    predicted=read_gzip(checked(source/row['predictions'],hashes['baseline']))
    samples={}
    records=[]
    start=perf_counter()
    for part,previous in zip(parts,predicted,strict=True):
        if part['id']!=previous['proposal']:
            raise ValueError('Predictor order changed')
        methods=previous['methods']
        baseline=(methods['phase']['evidence'].get('self_repeat',methods['current']) if cap else methods['phase_repeat'])
        chart=baseline.get('quad') or part.get('quad')
        sample=sample_surface(photo,dict(part,quad=chart))
        quad=sample['quad'] if sample else None
        result=predict_surface(photo,part,sample,baseline)
        guesses={k:dimensions(part,v.get('quad',quad),photo.size,v['counts'])['blocks']
                 for k,v in (('current',baseline),('patch',result))}
        record=dict(proposal=part['id'],guess=guesses)
        if 'patch' in result.get('evidence',{}):
            record['patch']=result['evidence']['patch']
            record['quad']=quad
            samples[part['id']]=sample
        records.append(record)
    elapsed=perf_counter()-start
    # Pair and grade only after all image-derived estimates are finished.
    if cap:
        _,ref=recorded(cap)
        labels=read(cap.path('labels'))['faces']
    else:
        with ZipFile(archive) as z:
            labels=json.loads(member(z,frame,'reference-faces.json'))['faces']
            ref=truth(json.loads(member(z,frame,'camera.json')),labels)
    planes=reference_planes(ref,labels)
    for record,old in zip(records,row['surfaces'],strict=True):
        if record['proposal']!=old['proposal'] or record['guess']['current']!=old['guess']['self_repeat' if cap else 'phase_repeat']:
            raise ValueError('Stored baseline differs: '+str([row['key'],record,old]))
        if 'reference' not in old:
            continue
        rid=old['reference']; target=planes[rid]
        record.update(reference=rid,actual=target['size'],materials=sorted({f['block_id'] for f in target['faces'].values()}))
        if 'patch' in record:
            record['patch_materials']=pixel_materials(ref,rid,target['faces'],samples[record['proposal']],record['patch'])
    return dict(key=row['key'],layout=row['layout'],cohort=row.get('cohort','additional'),sha256=row['sha256'],
                eligible=row['eligible'],surfaces=records,processing_s=elapsed)


def cases(source):
    m=read(source/'manifest.json')
    return m,read(checked(source/'cases.json',m['artifacts']['cases.json']))


@image_only()
def run(out,split,dataset,workers):
    out.mkdir(parents=True,exist_ok=False)
    a,first=cases(PHASE)
    b,second=cases(EXTRA)
    o=read(OUTLINES/'manifest.json')
    d=read(ARCHIVE/'manifest.json')
    jobs=[]
    if split in ('all','development','regression'):
        captures={c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
        for r in first:
            if split=='development' and r['cohort']!='development' or split=='regression' and r['cohort']=='development':
                continue
            rel=r['predictions']
            jobs.append((captures[r['key']],None,None,r,dict(baseline=a['artifacts'][rel],outlines=o['artifacts'][rel])))
    if split in ('all','additional'):
        lookup={r['key']:r for r in second}
        for _,path,_ in shards(dataset):
            with ZipFile(path) as z:
                frames=[json.loads(line) for line in z.read('frames.jsonl').splitlines()]
            for frame in frames:
                r=lookup[frame['id']]
                if r['sha256']!=frame['image_sha256']:
                    raise ValueError('Extra frame changed')
                jobs.append((None,path,frame,r,dict(baseline=b['artifacts'][r['predictions']],
                                                    ordinary=d['artifacts']['ordinary/'+r['key']+'.json.gz'])))
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        rows=list(pool.map(evaluate,jobs))
    groups_for=lambda r:['all',r['cohort'],r['layout'],'additional' if r['cohort']=='additional' else 'fixed']
    scores={k:summarize([r for r in rows if k in groups_for(r)],['current','patch'])
            for k in sorted({v for r in rows for v in groups_for(r)})}
    save(out/'cases.json',rows)
    save(out/'scores.json',scores)
    save(out/'manifest.json',dict(run_id=out.name,code=revision(),split=split,version='26.3',settings=SETTINGS,
         file_sha256=digest(ROOT/'benchmark/within_wall_grid.py'),source_manifests={f.name:digest(f/'manifest.json') for f in [PHASE,EXTRA,ARCHIVE,OUTLINES]},
         fixture_sha256=a['fixtures'],reference_sha256=a['reference_sha256'],additional_shards=b['shards'],
         methods={'current':'Approved held-strip self-repeat fallback applied to incomplete masked-repeat results','patch':'Within one predicted surface; extrapolate a textured patch and inspect its opposite end'},
         frame_ids=[dict(key=r['key'],image_sha256=r['sha256'],layout=r['layout'],split=r['cohort']) for r in rows],
         timing='Per-frame processing from cached model and baseline outputs; no new model inference. Includes resampling and dimension calculation.',
         artifacts={f:digest(out/f) for f in ('cases.json','scores.json')},
         reproduce=f'python -m benchmark.within_wall_round --split {split} --out runs/my-within-wall --dataset runs/visible-builds-v1'))
    print(scores['all'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--split',choices=['development','regression','additional','all'],default='all')
    p.add_argument('--dataset',type=Path,default=ROOT/'runs/visible-builds-v1')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    run(a.out,a.split,a.dataset,a.workers)
