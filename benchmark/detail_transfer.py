"""Frozen photo-outline transfer check on the two new Visible Builds v1 families."""
import argparse
import gzip
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

from benchmark import edge_detail
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_metrics import reference_planes
from benchmark.crop_fit_sheet import size_text
from benchmark.detail_round import native_metrics, support
from benchmark.mask_metrics import PROTOCOL, aggregate, measure
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_metrics import truth
from benchmark.surface_pipeline_sheet import _picture, _straight_photo
from benchmark.strip_review_sheet import overlay
from benchmark.camera_sheet import label
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface
from runtime.mode import image_only


RULE = 'native4-stable-counts'
ARCHIVE = ROOT/'benchmark/reviews/detail-transfer-01'
METHODS = ('current','photo')
SPLITS = ('validation','test')


def signature(path):
    return hashlib.sha256(path).hexdigest()


def pack(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(obj,separators=(',',':'),allow_nan=False).encode(),mtime=0))


def shards(dataset):
    data = read(ROOT/'training/visible-builds-v1.json')
    for split in SPLITS:
        shard = data['shards'][split]
        path = dataset/shard['file']
        if digest(path)!=shard['sha256']:
            raise ValueError('Incorrect shard: '+str(path))
        yield split,path,shard


def member(z,row,name):
    data=z.read(row['directory']+'/'+name)
    if signature(data)!=row['files'][name]:
        raise ValueError('Dataset member changed: '+row['id']+'/'+name)
    return data


def export_inputs(dataset,out):
    out.mkdir(parents=True,exist_ok=False)
    for _,path,_ in shards(dataset):
        with ZipFile(path) as z:
            for line in z.read('frames.jsonl').splitlines():
                row=json.loads(line)
                (out/(row['id']+'.png')).write_bytes(member(z,row,'rgb.png'))


def size_for(photo, part):
    sample=sample_surface(photo,part)
    counter=count(sample)
    return dimensions(part,sample['quad'] if sample else None,photo.size,counter['counts'])['blocks']


def compact_build(row,builds):
    folders=[p/row['id'] for p in builds if (p/row['id']/'build-manifest.json').is_file()]
    if len(folders)!=1:
        raise ValueError(f"Expected one finished ordinary build for {row['id']}; got {folders}")
    folder=folders[0]
    manifest=read(folder/'build-manifest.json')
    if manifest['image_sha256']!=row['image_sha256'] or manifest['stages']['strip_cleanup']!='neighbor-supported-strip':
        raise ValueError('Unexpected ordinary build input or method')
    parts=read(folder/'surfaces.json')['planes']
    counts=read(folder/'counts.json')['surfaces']
    if [p['id'] for p in parts]!=[c['id'] for c in counts]:
        raise ValueError('Invalid ordinary counts')
    return dict(planes=parts,blocks=[c['dimensions']['blocks'] for c in counts],
                manifest={k:manifest[k] for k in ('version','stages','image_sha256','model_sha256')},
                full_build_manifest_sha256=digest(folder/'build-manifest.json'))


def evaluate(z,row,input_build):
    photo=Image.open(io.BytesIO(member(z,row,'rgb.png'))).convert('RGB')
    if input_build['manifest']['image_sha256']!=row['image_sha256']:
        raise ValueError('Frozen baseline screenshot mismatch')
    base=input_build['planes']
    parts=dict(current=base,photo=edge_detail.predict(photo,base,row['image_sha256'],RULE))
    estimates={m:[size_for(photo,p) for p in regs] for m,regs in parts.items()}
    if estimates['current']!=input_build['blocks']:
        raise ValueError('Did not reproduce dimensions from ordinary build: '+row['id'])
    pose=json.loads(member(z,row,'camera.json'))
    faces=json.loads(member(z,row,'reference-faces.json'))['faces']
    ref=truth(pose,faces)
    native=_native_reference(pose,ref,faces)
    real=reference_planes(ref,faces)
    index={(tuple(face['xyz']),face['direction']):rid for rid,plane in real.items() for face in plane['faces'].values()}
    exported=json.loads(member(z,row,'labels.json'))['faces']
    mapping=np.zeros(len(exported)+1,dtype=np.int32)
    for face in exported:
        mapping[face['id']]=index[(tuple(face['xyz_world']),face['direction'])]
    with np.load(io.BytesIO(member(z,row,'labels.npz')),allow_pickle=False) as maps:
        if not np.array_equal(mapping[maps['face_id']],native):
            raise ValueError('Reprojected visible geometry differs from native release target')
    scores,matched,sizes={}, {}, {}
    for method, regs in parts.items():
        scores[method],pairs=measure(regs,ref,use_mask=True)
        matched[method]={rid:i for i,rid in pairs}
        estimates_for_pairs=[(estimates[method][i],real[rid]['size']) for i,rid in pairs]
        right=sum(guess==expected for guess,expected in estimates_for_pairs)
        wrong=sum(None not in guess and guess!=expected for guess,expected in estimates_for_pairs)
        used={i for i,rid in pairs}
        sizes[method]=dict(correct=right,incorrect=wrong,missing=scores[method]['eligible']-right-wrong,
                           unmatched_complete=sum(i not in used and None not in guess for i,guess in enumerate(estimates[method])))
    paired,counts=native_metrics(parts,native,[(i,rid) for rid,i in matched['current'].items()],photo.size)
    changed=set(matched['photo'])-set(matched['current'])
    exactsets={m:{rid for rid,i in pairs.items() if estimates[m][i]==real[rid]['size']} for m,pairs in matched.items()}
    lost=set(matched['current'])-set(matched['photo'])
    return dict(id=row['id'],split=row['split'],family=row['family'],layout=row.get('layout',row['id']),
                sha256=row['image_sha256'],scores=scores,native=counts,sizes=sizes,gained=sorted(changed),
                lost=sorted(lost),exact_sizes_gained=sorted(exactsets['photo']-exactsets['current']),
                exact_sizes_lost=sorted(exactsets['current']-exactsets['photo']),paired=paired),parts,real,native,photo


def totals(rows):
    report={}
    for group in ('all',*SPLITS,*sorted({r['layout'] for r in rows})):
        subset=[r for r in rows if group in ('all',r['split'],r['layout'])]
        report[group]={}
        for method in METHODS:
            metrics=aggregate([r['scores'][method] for r in subset])
            counts=sum((Counter(r['native'][method]) for r in subset),Counter())
            sizes=sum((Counter(r['sizes'][method]) for r in subset),Counter())
            report[group][method]=dict(frames=len(subset),**metrics,sizes=dict(sizes),
                native=dict(counts,edge_px=counts['edge_sum']/counts['edge_n'],iou=counts['intersection']/counts['union']),
                gained=sum(len(r['gained']) for r in subset) if method=='photo' else 0,
                exact_gained=sum(len(r['exact_sizes_gained']) for r in subset) if method=='photo' else 0,
                exact_lost=sum(len(r['exact_sizes_lost']) for r in subset) if method=='photo' else 0,
                lost=sum(len(r['lost']) for r in subset) if method=='photo' else 0)
    return report


def review_candidate(reviews,row,rec,parts,real,native,photo):
    """Select by number of changed pixels on a paired surface, with stable ties."""
    for face in rec['paired']:
        delta=face['methods']['photo']
        n=sum(delta[k] for k in ('added_correct','added_wrong','removed_correct','removed_wrong'))
        prior=reviews.get(row['layout'])
        if prior and n<=prior[0]:
            continue
        rid=face['reference']
        truth_mask=native==rid
        own={m:next(p for p in pieces if p['id']==face['proposal']) for m,pieces in parts.items()}
        masks={m:support(p,photo.size) for m,p in own.items()}
        l,t,r,b=Image.fromarray(truth_mask|masks['current']|masks['photo']).getbbox()
        detail=(max(0,l-12),max(0,t-12),min(photo.width,r+12),min(photo.height,b+12))
        locator=photo.copy()
        ImageDraw.Draw(locator).rectangle(detail,outline='white',width=4)
        actual=real[rid]['size']
        cells=[(locator.crop(Image.fromarray(native>0).getbbox()),row['angle']+' / '+row['palette']),
               (overlay(photo,detail,[(truth_mask,(245,187,65),.72)]),f'R{rid} | Actual {size_text(actual)}')]
        for m in METHODS:
            mask=masks[m]
            cells.append((overlay(photo,detail,[(mask,(25,220,218),.53)]),f'{int((mask & ~truth_mask).sum())} extra | {int((truth_mask & ~mask).sum())} missing'))
        change=masks['current']^masks['photo']
        good=change & (masks['photo']==truth_mask)
        cells.append((overlay(photo,detail,[(good,(33,230,115),1.),(change&~good,(255,53,124),1.)]),f'{int(good.sum())} fixed | {int((change&~good).sum())} worsened'))
        for m in METHODS:
            sample=sample_surface(photo,own[m])
            counter=count(sample)
            clean,grid=_straight_photo(sample,counter)[:2] if sample else (None,None)
            cells.extend([(clean,'Supported pixels only'),(grid,'Counter '+size_text(size_for(photo,own[m]))+' | Actual '+size_text(actual))])
        reviews[row['layout']]=(n,dict(frame=row['id'],sha256=row['image_sha256'],reference=rid,proposal=face['proposal'],box=detail,actual=actual),cells)


def render(out,reviews):
    names=['Source','Recorded surface','Default outline','Photo outline','Photo changes','Default crop','Default count','Photo crop','Photo count']
    width,height,left,top=320,279,190,120
    page=Image.new('RGB',(left+width*len(reviews),top+height*len(names)+7),'#f8f7f1')
    pen=ImageDraw.Draw(page)
    label(pen,(14,12),'Photo edges on two held-out building families',35,bold=True)
    label(pen,(14,60),'Most changed matched surface per layout. Gold = recorded. Cyan = predicted. Changes show fixed pixels in green and worsened pixels in pink.',21)
    for i,name in enumerate(names):
        y=top+height*i
        pen.line((8,y,page.width-8,y),fill='#acbcb7')
        label(pen,(12,y+15),name,19,bold=True)
    index={}
    for col,(layout,(_,record,cells)) in enumerate(sorted(reviews.items())):
        x=left+col*width
        index[layout]=record
        label(pen,(x,top-27),layout,21,bold=True)
        for i,(im,caption) in enumerate(cells):
            y=top+i*height
            if im is not None: _picture(page,im,x+2,y+5,width-6,height-42)
            label(pen,(x+2,y+height-30),caption,15)
    page.save(out/'comparison.jpg',quality=91,subsampling=0)
    save(out/'sheet-views.json',index)


def frame_job(job):
    path,row,builds,out=job
    filename='ordinary/'+row['id']+'.json.gz'
    if builds:
        build=compact_build(row,builds)
    else:
        manifest=read(ARCHIVE/'manifest.json')
        archived=ARCHIVE/filename
        if digest(archived)!=manifest['artifacts'][filename]:
            raise ValueError('Changed saved build '+filename)
        build=json.loads(gzip.decompress(archived.read_bytes()))
    pack(out/filename,build)
    with ZipFile(path) as z:
        rec,parts,real,native,photo=evaluate(z,row,build)
    local={}
    review_candidate(local,row,rec,parts,real,native,photo)
    info=dict(sha256=row['image_sha256'],split=row['split'],layout=row['layout'],
              build_manifest_sha256=build['full_build_manifest_sha256'],ordinary=filename)
    return rec,local.get(row['layout']),info


@image_only()
def run(dataset,out,builds,workers):
    out.mkdir(parents=True,exist_ok=False)
    jobs=[]
    rows=[]
    index={}
    reviews={}
    for split,path,shard in shards(dataset):
        with ZipFile(path) as z:
            inputs=[json.loads(line) for line in z.read('frames.jsonl').splitlines()]
            if len(inputs)!=shard['frames'] or {r['split'] for r in inputs}!={split}:
                raise ValueError('Frame inventory differs')
            jobs.extend((path,row,builds,out) for row in inputs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rec,review,info in pool.map(frame_job,jobs):
            rows.append(rec)
            index[rec['id']]=info
            previous=reviews.get(rec['layout'])
            if review and (previous is None or review[0]>previous[0]):
                reviews[rec['layout']]=review
    render(out,reviews)
    summary=totals(rows)
    save(out/'cases.json',rows)
    save(out/'scores.json',summary)
    save(out/'manifest.json',dict(run_id=out.name,code=revision(),frozen_method_revision='eefbe59',
        setting=RULE,settings=edge_detail.SETTINGS[RULE],protocol=PROTOCOL,
        expected='Improve paired edge distance for both families, retain existing matches and exact block sizes, avoid extra incorrect guesses or covered openings.',
        models='Ordinary single-image default; per-build model hashes and stages in ordinary/*.json.gz.',
        reference='Scores use reference-faces.json and the recorded camera; every projected native surface pixel verified against the released face_id array.',
        dataset_manifest_sha256=digest(ROOT/'training/visible-builds-v1.json'),
        shards={split:entry for split,_,entry in shards(dataset)},frames=index,
        reproduce='python -m benchmark.detail_transfer --dataset <visible-builds-v1-shards> --out runs/transfer-replay',
        artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    for split,v in summary.items():
        for method,r in v.items():
            print(split,method,r['matched'],r['native']['edge_px'],r['opening_filled'],r['sizes'])


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=4)
    cli.add_argument('--builds',action='append',type=Path,default=[])
    cli.add_argument('--inputs-only',action='store_true')
    args=cli.parse_args()
    if args.inputs_only: export_inputs(args.dataset,args.out)
    else: run(args.dataset,args.out,args.builds,args.workers)
