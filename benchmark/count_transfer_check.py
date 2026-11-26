"""Evaluate frozen single-image block counters on Cloister and Twin Spires."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import importlib
import io
import json
from pathlib import Path
import subprocess
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_metrics import reference_planes
from benchmark.detail_transfer import ARCHIVE, member, shards
from benchmark.mask_metrics import measure
from benchmark.surface_metrics import truth
from benchmark.texture_count_round import scalar, summarize
from benchmark.wall_join_test import checked, read_gzip
from engine import photo_edges
from engine.geometry import UNIT, project
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from runtime.mode import image_only, mark_image_only


MODULES = {'refit':'repeat_rectify','phase':'phase_corners','coplanar':'coplanar_count'}


def predict(photo, parts, methods):
    samples = [sample_surface(photo, part) for part in parts]
    start = perf_counter()
    base = [count(sample) for sample in samples]
    predictions = {'current':base}
    timing = {'current':perf_counter()-start}
    for method in methods:
        module = importlib.import_module('benchmark.'+MODULES[method])
        start = perf_counter()
        if method=='coplanar':
            results = module.predict_frame(photo,parts,samples,deepcopy(base))
        else:
            results = [module.predict_surface(photo,part,sample,deepcopy(baseline))
                       for part,sample,baseline in zip(parts,samples,base)]
        if len(results)!=len(parts):
            raise ValueError(f'{method} returned a different surface inventory')
        if method=='phase':
            predictions['phase_repeat'] = [r['evidence'].get('self_repeat',baseline)
                                           for r,baseline in zip(results,base)]
        predictions[method] = results
        timing[method] = perf_counter()-start
    rows = []
    for index,(part,sample) in enumerate(zip(parts,samples)):
        guesses = {}
        for method,results in predictions.items():
            result = results[index]
            quad = result.get('quad',sample['quad'] if sample else None)
            guesses[method] = dimensions(part,quad,photo.size,result['counts'])['blocks']
        rows.append(dict(proposal=part['id'],guess=guesses))
    return rows,predictions,timing



def grade_transfer(transfer,rid,did,ref,targets):
    data = dict(donor=transfer['donor'],predicted_offset=transfer['recipient_offset'],
                recipient_reference=rid,donor_reference=did,correct_plane=None)
    if did is None or rid is None:
        return data
    source,target = (ref['patches'][i] for i in (did,rid))
    data['correct_plane'] = source['normal']==target['normal'] and source['offset']==target['offset']
    data['donor_and_recipient_materials'] = [
        sorted({f['block_id'] for f in targets[i]['faces'].values()}) for i in (did,rid)]
    if data['correct_plane']:
        world_grid = np.linalg.inv(ref['charts'][did])@transfer['donor_grid_to_image']
        basis = project(world_grid,UNIT)
        vectors = basis[[1,3]]-basis[0]
        axes = np.argmax(abs(vectors),axis=1)
        if axes[0]!=axes[1]:
            orientation = np.sign(vectors[np.arange(2),axes])
            origins = []
            for i in (did,rid):
                cells = np.asarray(list(targets[i]['faces']))[:,axes]
                origins.append(np.where(orientation>0,cells.min(0),-cells.max(0)-1))
            data['actual_offset'] = (origins[1]-origins[0]).tolist()
            data['exact_offset'] = data['actual_offset']==data['predicted_offset']
    return data


def evaluate(job):
    path,frame,ordinary_hash,methods,out = job
    frame_id = frame['id']
    ordinary = read_gzip(checked(ARCHIVE/'ordinary'/(frame_id+'.json.gz'),ordinary_hash))
    with ZipFile(path) as source:
        photo = Image.open(io.BytesIO(member(source,frame,'rgb.png'))).convert('RGB')
        if ordinary['manifest']['image_sha256']!=frame['image_sha256']:
            raise ValueError('Archived regions belong to another screenshot')
        parts,_ = photo_edges.refine_surfaces(photo,ordinary['planes'])
        records,predictions,timing = predict(photo,parts,methods)
        # Save all predictor outputs before looking at the evaluation camera or geometry.
        file = 'predictions/'+frame_id+'.json.gz'
        evidence = [dict(proposal=part['id'],methods={m:p[i] for m,p in predictions.items()})
                    for i,part in enumerate(parts)]
        (out/file).write_bytes(gzip.compress(json.dumps(evidence,separators=(',',':'),allow_nan=False,
                                                       default=scalar).encode(),mtime=0))
        pose = json.loads(member(source,frame,'camera.json'))
        faces = json.loads(member(source,frame,'reference-faces.json'))['faces']
    ref = truth(pose,faces)
    targets = reference_planes(ref,faces)
    metrics,pairs = measure(parts,ref,use_mask=True)
    for index,rid in pairs:
        records[index].update(reference=rid,actual=targets[rid]['size'])
    if 'coplanar' in predictions:
        paired = {records[i]['proposal']:rid for i,rid in pairs}
        for record,result in zip(records,predictions['coplanar']):
            transfer = result.get('evidence',{}).get('transfer')
            if transfer:
                record['transfer_score'] = grade_transfer(transfer,record.get('reference'),
                                                          paired.get(transfer['donor']),ref,targets)
    row = dict(key=frame_id,sha256=frame['image_sha256'],family=frame['family'],layout=frame['layout'],
               split=frame['split'],eligible=metrics['eligible'],surfaces=records,predictions=file,
               processing_s=timing,ordinary_sha256=ordinary_hash)
    row['scores'] = summarize([row],list(predictions))
    return row


def provenance(module_name):
    module = importlib.import_module('benchmark.'+module_name)
    path = Path(module.__file__).resolve()
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=path.parent,text=True).strip()
    frozen = subprocess.check_output(['git','status','--porcelain','--',path.name],cwd=path.parent,text=True)
    if frozen.strip():
        raise ValueError('Commit candidate code before transfer evaluation: '+str(path))
    return dict(module=module_name,file_sha256=digest(path),commit=commit,settings=module.SETTINGS)


@image_only()
def run(dataset,out,workers,methods):
    archive = read(ARCHIVE/'manifest.json')
    previous = {r['id']:r for r in read(checked(ARCHIVE/'cases.json',archive['artifacts']['cases.json']))}
    code = revision()
    sources = {m:provenance(MODULES[m]) for m in methods}
    jobs = []
    for split,path,shard in shards(dataset):
        with ZipFile(path) as source:
            frames = [json.loads(s) for s in source.read('frames.jsonl').splitlines()]
        if len(frames)!=shard['frames'] or any(f['split']!=split for f in frames):
            raise ValueError('Unexpected transfer split')
        for frame in frames:
            frame_id = frame['id']
            if frame['image_sha256']!=previous[frame_id]['sha256']:
                raise ValueError('Screenshot differs from verified frame')
            file = 'ordinary/'+frame_id+'.json.gz'
            jobs.append((path,frame,archive['artifacts'][file],methods,out))
    if {j[1]['id'] for j in jobs}!=previous.keys() or len(jobs)!=len(previous):
        raise ValueError('Frame set changed')
    out.mkdir(parents=True,exist_ok=False)
    (out/'predictions').mkdir()
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        rows = list(pool.map(evaluate,jobs))
    for r in rows:
        score = r['scores']['current']
        old = previous[r['key']]
        if (any(score[k]!=old['sizes']['photo'][k] for k in ('correct','incorrect','missing'))
                or any(score[k]!=old['scores']['photo'][k] for k in ('eligible','matched'))
                or score['unmatched_returned']!=old['sizes']['photo']['unmatched_complete']):
            raise ValueError('Per-picture shipped baseline differs: '+r['key'])
    names = list(rows[0]['scores'])
    groups = sorted({r['family'] for r in rows}|{r['layout'] for r in rows})
    scores = {g:summarize([r for r in rows if g in ('all',r['family'],r['layout'])],names)
              for g in ['all',*groups]}
    save(out/'scores.json',scores)
    save(out/'cases.json',rows)
    atlas = {}
    if 'phase' in methods:
        from benchmark.portico_atlas import atlas as vanilla_atlas
        atlas = vanilla_atlas().provenance
    save(out/'manifest.json',dict(run_id=out.name,version='26.3',code=code,
        algorithm='No tuning. Frozen predictors receive only one RGB and image-derived surfaces, with normals, relative inverse depth and baseline count evidence.',
        methods=names,sources=sources,atlas=atlas,photo_rule=photo_edges.METHOD,
        inputs='Saved ordinary predictions and two Visible Builds v1 released shards; all original inference reused.',
        transfer_archive_sha256=digest(ARCHIVE/'manifest.json'),shards=archive['shards'],
        fixture_ids=[dict(id=r['key'],layout=r['layout'],image_sha256=r['sha256']) for r in rows],
        workers=workers,summary=scores['all'],
        command='python -m benchmark.count_transfer_check --dataset <visible-builds-v1-shards> --out runs/count-transfer-replay --methods '+' '.join(methods),
        artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(scores['all'])


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--methods',choices=MODULES,nargs='*',default=list(MODULES))
    cli.add_argument('--workers',type=int,default=4)
    args = cli.parse_args()
    run(args.dataset,args.out,args.workers,args.methods)
