"""Rescore frozen decoded builds after one integer translation per screenshot."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import best_translation
from benchmark.texture_camera_eval import DATA, STABLE, SPLITS, checked, digest, frames, read
from benchmark import visible_depth_native as native
from benchmark import visible_depth_transfer as transfer
from training.visible_labels import visible_labels


SIZES = dict(validation=12, test=12, regression=56, native=12, old=12, fresh=8, train=72)
FIELDS = ('target','placed','correct','extras','missed','hidden_extras')


def load_suite(split, runs, data, chooser):
    depth = runs/'visible-depth-rounds'
    canopy = runs/'canopy-placement'
    fixture = depth/'transfer' if split == 'old' else canopy/'confirmation'
    if split in ('old','fresh'):
        samples = read(fixture/'screen.json')['selected']
        prior = fixture/'results'/('production' if split=='old' else 'baseline')
    else:
        samples = native.rows() if split=='native' else frames(data,split)
        prior = depth/'shipped'/split
    folders = dict(previous=prior, current=canopy/'release'/split,
                   before_edges=(fixture/'results/combo' if split=='fresh' else canopy/'combo'/split))
    if chooser is not None and split!='fresh':
        for method,folder in [('guarded','guarded'),('unrestricted','learned')]:
            transfer_folder = 'guarded-native-transfer' if method=='guarded' else 'place-native-transfer'
            folders[method] = chooser/(transfer_folder if split=='old' else folder+'/'+split)
    ids = {f['id'] for f in samples}
    assert len(ids)==len(samples)==SIZES[split], split
    outputs = {method:{} for method in folders}
    manifests = {}
    for method,folder in folders.items():
        manifest = read(folder/'manifest.json')
        rows = read(checked(folder/'cases.json',manifest['cases_sha256']))
        outputs[method] = {r['id']:r for r in rows}
        assert outputs[method].keys()==ids and len(rows)==len(ids)
        manifests[method]=dict(manifest_sha256=digest(folder/'manifest.json'),
                               cases_sha256=manifest['cases_sha256'],
                               revision=manifest.get('revision',manifest.get('candidate_revision')))
    return [(split, f, fixture, data, depth, {m:(p,outputs[m][f['id']]) for m,p in folders.items()})
            for f in samples], manifests


def references(split,f,fixture,data,depth,fixture_roots):
    sha=f['files']
    key=f['id']
    if split in ('old','fresh'):
        base=fixture_roots[split]
        image=base/f['directory']/'captures'/(f['source_id']+'.png')
        checked(image,sha['rgb.png'])
        pose=read(checked(image.with_suffix('.json'),sha['camera.json']))
        raw=read(checked(image.with_name(f['source_id']+'-blocks.json'),sha['world-readback.json']))
        states={tuple(b[k] for k in 'xyz'):b for b in raw}
        labels,_,_=visible_labels(pose,states,(0,0,0))
        truth={tuple(b['xyz_world']) for b in labels['blocks']}
        occupied=set(transfer.structure(raw,64))
        upstream=fixture/'upstream'/key
        hashes=dict(surface_sha256=digest(upstream/'surfaces.json'),counts_sha256=digest(upstream/'counts.json'))
        projection=f['camera']['P']
    else:
        path=depth/'inputs'/(key+'.json')
        source=read(path)
        hashes=dict(input_sha256=digest(path))
        assert source['id']==key and source['rgb_sha256']==sha['rgb.png']
        projection=source['camera']['P']
        if split=='native':
            checked(native.rgb_path(f,fixture_roots['native']),sha['rgb.png'])
            pose,labels,occupied=native.truth(f,fixture_roots['native'])
            truth=set(labels)
        else:
            folder=data/'data'/split/f['directory']
            checked(folder/'rgb.png',sha['rgb.png'])
            pose=read(checked(folder/'camera.json',sha['camera.json']))
            labels=read(checked(folder/'labels.json',sha['labels.json']))
            truth={tuple(b['xyz_world']) for b in labels['blocks']}
            states=read(checked(folder/'world-readback.json',sha['world-readback.json']))
            occupied=set(transfer.structure(states,f['ground_y']))
    assert truth<=occupied
    return truth,occupied,registration(np.asarray(projection),pose) if projection is not None else None,hashes


def metrics(cells,truth,occupied):
    extras=cells-truth
    correct=len(cells)-len(extras)
    return dict(target=len(truth),placed=len(cells),correct=correct,extras=len(extras),
                missed=len(truth)-correct,hidden_extras=len(extras&occupied))


def evaluate(job):
    split,f,fixture,data,depth,methods,fixture_roots=job
    key=f['id']
    truth,occupied,transform,hashes=references(split,f,fixture,data,depth,fixture_roots)
    row=dict(split=split,id=key,rgb_sha256=f['files']['rgb.png'],
             pose_sha256=f['files']['camera.json'],readback_sha256=f['files']['world-readback.json'],
             visible_coordinates_sha256=hashlib.sha256(json.dumps(sorted(truth),separators=(',',':')).encode()).hexdigest(),
             **hashes, methods={})
    cache={}
    for name,(root,saved) in methods.items():
        assert row['rgb_sha256']==saved['rgb_sha256']
        assert all(saved[k]==v for k,v in hashes.items())
        folder=root/key
        emitted=decoded_pack(checked(folder/'reconstruction.zip',saved['pack_sha256']))
        archived=read(folder/'cells.json')
        assert set(emitted)==set(map(tuple,archived['local']))
        assert transform is not None or not emitted
        world=registered(emitted,transform)
        assert world==set(map(tuple,archived['world']))
        original=metrics(world,truth,occupied)
        assert all(original[k]==saved['metrics'][k] for k in FIELDS),(split,key,name)
        identity=frozenset(world)
        if identity not in cache:
            shift,n=best_translation(world,truth)
            x,y,z=shift
            aligned=metrics({(a+x,b+y,c+z) for a,b,c in world},truth,occupied)
            assert aligned['correct']==n and n>=original['correct']
            cache[identity]=tuple(map(int,shift)),aligned
        shift,aligned=cache[identity]
        row['methods'][name]=dict(pack_sha256=saved['pack_sha256'],world_cells_sha256=digest(folder/'cells.json'),
                                  offset=shift,original=original,aligned=aligned)
    return row


def summary(rows):
    methods=set.intersection(*(set(r['methods']) for r in rows))
    result={}
    for method in sorted(methods):
        cases=[r['methods'][method] for r in rows]
        scores={}
        for system in ('original','aligned'):
            counts=Counter()
            for case in cases:
                counts.update(case[system])
            scores[system]=dict(counts,f1=2*counts['correct']/max(1,counts['target']+counts['placed']))
        result[method]=dict(frames=len(rows),improved_frames=sum(c['aligned']['correct']>c['original']['correct'] for c in cases),
                            **scores)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=STABLE/'runs')
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--chooser-runs',type=Path,help='Include guarded and unrestricted model ablations on their common frames.')
    for split in ('native','old','fresh'):
        p.add_argument('--'+split+'-fixtures',type=Path,default=native.BUILDS)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--out',type=Path,required=True,help='ZIP holding scores.json')
    a=p.parse_args()
    result=dict(offset_direction='add to decoded world coordinates after camera registration',manifests={},cohorts={},cases=[])
    roots={s:getattr(a,s+'_fixtures') for s in ('native','old','fresh')}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for split in SIZES:
            jobs,result['manifests'][split]=load_suite(split,a.runs,a.data,a.chooser_runs)
            rows=list(pool.map(evaluate,[(*job,roots) for job in jobs]))
            result['cases'].extend(rows)
            result['cohorts'][split]=summary(rows)
            print(split,[(m,s['original']['correct'],s['aligned']['correct'],s['improved_frames'])
                         for m,s in result['cohorts'][split].items()],flush=True)
    result['evaluation']=summary([r for r in result['cases'] if r['split']!='train'])
    if a.chooser_runs:
        result['chooser_evaluation']=summary([r for r in result['cases'] if r['split'] not in ('train','fresh')])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(a.out,'w',ZIP_DEFLATED,compresslevel=9) as z:
        z.writestr('scores.json',json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
