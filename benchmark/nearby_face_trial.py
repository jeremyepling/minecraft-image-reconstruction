"""Develop and compare local face decisions on frozen single-image placement inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmark import visible_depth_native as native
from benchmark import visible_depth_transfer as transfer
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack, registered, _write_read
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import DATA, TRAINING_DATA, TRAINING_RELEASE, ROOT, STABLE, read, save, checked, frames, digest
from engine import nearby_face_decision as method
from training.visible_labels import visible_labels

BASE=STABLE.parent/(STABLE.name+'-worktrees')/'combined-visible-recovery/runs/triple'
EXTRA=BASE.parents[2]/'visible-recovery-extra-views/runs/extra-views'
RUN=ROOT/'runs/nearby-face'
SPLITS=('validation','test','regression','native','old','fresh','extra')
PUBLIC=ROOT/'benchmark/reviews/nearby-face-01'


def cases(split):
    if split in ('old','fresh','extra'):
        source={'old':transfer.OUT,'fresh':STABLE/'runs/canopy-placement/confirmation','extra':EXTRA}[split]
        ff=read(source/'screen.json')['selected']
    else:
        source=STABLE/'runs/visible-depth-rounds'
        if split == 'train':
            ff = frames(TRAINING_DATA, split, TRAINING_RELEASE)
        else:
            ff=native.rows() if split=='native' else frames(DATA,split)
    base=EXTRA/'candidate' if split=='extra' else BASE/split
    rows={r['id']:r for r in read(base/'cases.json')}
    return [(split,f,source,base,rows[f['id']]) for f in ff]


def load_prediction(job):
    split,f,source,base,row=job
    key=f['id']
    if split in ('old','fresh','extra'):
        parts=read(checked(source/'upstream'/key/'surfaces.json',row['surface_sha256']))['planes']
        camera=f['camera']
        path=transfer.rgb_path(f)
    else:
        data=read(checked(source/'inputs'/(key+'.json'),row['input_sha256']))
        parts,camera=data['parts'],data['camera']
        path=native.rgb_path(f,native.BUILDS) if split=='native' else (TRAINING_DATA if split == 'train' else DATA)/'data'/split/f['directory']/'rgb.png'
    with Image.open(checked(path,f['files']['rgb.png'])) as im:
        image=im.convert('RGB')
    prior=set(decoded_pack(checked(base/key/'reconstruction.zip',row['pack_sha256'])))
    return image,parts,camera,prior


def labels(job,P):
    split,f,_,_,_=job
    if split in ('old','fresh','extra','native'):
        stem=f.get('source_id',f['id'])
        folder=native.BUILDS/f['directory']/'captures'
        pose=read(checked(folder/(stem+'.json'),f['files']['camera.json']))
        raw=read(checked(folder/(stem+'-blocks.json'),f['files']['world-readback.json']))
        states={tuple(b[k] for k in 'xyz'):b for b in raw}
        visible,_,_=visible_labels(pose,states,(0,0,0))
        occupied=set(transfer.structure(raw,64))
    else:
        folder=(TRAINING_DATA if split == 'train' else DATA)/'data'/split/f['directory']
        pose=read(checked(folder/'camera.json',f['files']['camera.json']))
        visible=read(checked(folder/'labels.json',f['files']['labels.json']))
        raw=read(checked(folder/'world-readback.json',f['files']['world-readback.json']))
        occupied=set(transfer.structure(raw,f['ground_y']))
    truth={tuple(b['xyz_world']):b['block_id'] for b in visible['blocks']}
    T=registration(np.asarray(P),pose) if P is not None else None
    return T,truth,occupied


def features(job):
    split,f,_,_,_=job
    out=RUN/'samples'/split/(f['id']+'.npz')
    out.parent.mkdir(parents=True,exist_ok=True)
    image,parts,camera,prior=load_prediction(job)
    data=method.observations(image,parts,camera,prior)
    if data is None:
        data=dict(points=np.array(sorted(prior),dtype=int).reshape(-1,3),edges=np.empty((0,3),int),
                  features=np.empty((0,56),np.float32),base_count=len(prior))
    if split=='train':
        T,truth,_=labels(job,camera['P'])
        pts=data['points']+.5
        world=np.floor(pts@T[:3,:3].T+T[:3,3]).astype(int) if len(pts) else []
        data['correct']=np.array([tuple(p) in truth for p in world],bool)
    np.savez_compressed(out,**data)
    return len(data['edges'])


def frozen(job):
    split,f,*_=job
    with np.load(RUN/'samples'/split/(f['id']+'.npz')) as data:
        return dict(data)


def art(job):
    from engine.nearby_face_art import enrich
    split,f,*_=job
    image,parts,camera,_=load_prediction(job)
    data=frozen(job)
    result=enrich(image,parts,camera,data)
    if result is None:
        raise ValueError('The calibrated atlas is required for the art-feature trial')
    path=RUN/'art'/split/(f['id']+'.npz')
    path.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(path,columns=result['features'][:,56:])
    return len(data['edges'])


def art_sample(job):
    split,f,*_=job
    data=frozen(job)
    with np.load(RUN/'art'/split/(f['id']+'.npz')) as sample:
        data['features']=np.column_stack((data['features'],sample['columns']))
    return data


def evaluate(job):
    split,f,_,_,base=job
    model=read(method.MODEL)
    data=art_sample(job) if model.get('atlas') else frozen(job)
    image,parts,camera,prior=load_prediction(job)
    decisions=(set(),set(),set()) if not len(data['edges']) else method.choose(data,model)
    placed=method.apply(image,parts,camera,prior,data,decisions)
    out=RUN/'predictions'/split/f['id']
    out.mkdir(exist_ok=True,parents=True)
    pack=out/'reconstruction.zip'
    local=_write_read(pack,placed)
    T,truth,occupied=labels(job,camera['P'])
    world,before=registered(local,T),registered(prior,T)
    scores=placement_scores(world,before,truth,occupied)
    save(out/'cells.json',dict(id=f['id'],local=sorted(local),world=sorted(world),input_sha256=base.get('input_sha256')))
    row={k:v for k,v in base.items() if k not in ('seconds','metrics','relative_geometry','translation','baseline_translation','pack_sha256')}
    return dict(row,**scores,pack_sha256=digest(pack))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('features','art','evaluate'))
    p.add_argument('--sets',nargs='+',required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--workers',type=int,default=5)
    args=p.parse_args()
    from benchmark.visible_depth_score import aggregate
    for split in args.sets:
        jobs=cases(split)[:args.limit]
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows=list(pool.map({'features':features,'art':art,'evaluate':evaluate}[args.stage],jobs))
        if args.stage!='evaluate':
            print(split,'pairs',sum(rows),flush=True)
        else:
            out=RUN/'predictions'/split
            save(out/'cases.json',rows)
            summary=aggregate(rows)['relative_geometry']
            save(out/'manifest.json',dict(summary=summary,model_sha256=digest(method.MODEL),cases_sha256=digest(out/'cases.json')))
            print(split,summary,flush=True)
        print('seconds',round(time.perf_counter()-start,2),flush=True)


if __name__=='__main__':
    main()

