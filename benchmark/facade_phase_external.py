"""Screen separate saved facade photos, then replay ordinary masks and the frozen inset rule."""
import argparse
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.facade_phase_evidence import REVIEW,common
from benchmark import framed_depth_eval as framed
from benchmark.texture_camera_eval import ROOT,STABLE,digest,read,save
from benchmark.visible_depth_transfer import shape_key,structure
from engine.camera_metric import predict
from engine.facade_inset_phase import refine
from training.framing import measure_framing
from training.visible_labels import visible_labels


BUILDS=STABLE.parent/'workspace/builds'
SUITES=('recess-control-26.3','recess-positive-control-26.3','entrance-holdout-26.3/positive',
        'entrance-holdout-26.3/controls','signed-relief-26.3/test','signed-relief-26.3/dev','shape-benchmark-26.3')


def native(source):
    pose=read(source.with_suffix('.json'))
    raw=read(source.with_name(source.stem+'-blocks.json'))
    labels,ids,_=visible_labels(pose,{tuple(b[k] for k in 'xyz'):b for b in raw},(0,0,0))
    return pose,raw,labels,ids


def screen():
    used=set()
    seen_shapes=set()
    for split,frame,_,_ in common.jobs([*framed.SIZES,'shelter']):
        used.add(frame['files']['rgb.png'])
        if split in ('native','shelter',*framed.NATIVE_INPUTS):
            if split=='native':
                rgb=framed.native.rgb_path(frame,framed.native.BUILDS)
            else:
                root=common.SHELTER if split=='shelter' else framed.NEW_INPUTS if split=='new' else framed.transfer.BUILDS
                rgb=root/frame['directory']/'captures'/(frame['source_id']+'.png')
            states=read(rgb.with_name(rgb.stem+'-blocks.json'))
            ground=64
        else:
            states=read(framed.TRAINING_DATA/'data'/split/frame['directory']/'world-readback.json')
            ground=frame['ground_y']
        seen_shapes.add(shape_key(structure(states,ground)))
    all_rows=[]
    for suite in SUITES:
        for png in sorted((BUILDS/suite/'captures').glob('*.png')):
            if suite=='shape-benchmark-26.3' and '-recess-' not in png.stem:
                continue
            hashed=digest(png)
            info=dict(suite=suite,frame=png.stem,rgb_sha256=hashed)
            if hashed in used:
                all_rows.append(dict(info,reason='previously_seen_pixels'))
                continue
            used.add(hashed)
            pose,raw,labels,ids=native(png)
            framing=measure_framing(pose,raw,64,ids)
            shape=shape_key(structure(raw,64))
            condition=('dark-inlay' if 'dark-inlay' in png.stem else 'physical-shade' if 'physical-shade' in png.stem else 'relief')
            info.update(shape=shape,existing_layout=shape in seen_shapes,condition=condition,
                        camera_sha256=digest(png.with_suffix('.json')),world_sha256=digest(png.with_name(png.stem+'-blocks.json')),
                        framing=framing)
            if not framing['accepted']:
                all_rows.append(dict(info,reason='incomplete_framing'))
                continue
            with Image.open(png) as shot:
                camera=predict(shot.convert('RGB'),atlas_path=common.ATLAS)
            all_rows.append(dict(info,reason='eligible' if camera.get('P') is not None else 'camera_unavailable',camera=camera))
    # Keep a representative flag for layout-level reporting while evaluating
    # every eligible image. Selection uses only framing and input availability.
    chosen=set()
    for row in sorted(all_rows,key=lambda r:(r.get('existing_layout',True),r['rgb_sha256'])):
        if row['reason']=='eligible':
            row['selected_for_ordinary_replay']=True
            group=(row['shape'],row['condition'])
            if group not in chosen:
                chosen.add(group)
                row['layout_condition_representative']=True
    return all_rows


def run(args):
    REVIEW.mkdir(parents=True,exist_ok=True)
    path=REVIEW/'external-screen.json'
    rows=screen() if not path.exists() else read(path)
    save(path,rows)
    eligible=[r for r in rows if r['reason']=='eligible']
    print('screen',len(rows),Counter(r['reason'] for r in rows),'selected',[(r['suite'],r['frame'],r['existing_layout']) for r in eligible],flush=True)
    if args.screen_only:return
    cases=[]
    for r in eligible:
        png=BUILDS/r['suite']/'captures'/(r['frame']+'.png')
        assert digest(png)==r['rgb_sha256']
        folder=args.out/r['rgb_sha256'][:12]
        if not (folder/'build-manifest.json').exists():
            folder.mkdir(parents=True,exist_ok=True)
            print('Building',png.stem,flush=True)
            with (folder/'build.log').open('w',encoding='utf-8') as log:
                subprocess.run(['pwsh','-NoProfile','-File',str(ROOT/'Invoke-WallRecovery.ps1'),str(png),'-Through','surfaces',
                                '-Out',str(folder.resolve())],check=True,stdout=log,stderr=subprocess.STDOUT)
        manifest=read(folder/'build-manifest.json')
        assert manifest['image_sha256']==r['rgb_sha256'] and manifest['through']=='surfaces'
        with Image.open(png) as shot:photo=shot.convert('RGB')
        parts=read(folder/'surfaces.json')['planes']
        log=[]
        proposed,events=refine(photo,parts,r['camera'],candidate_log=log)
        _,_,labels,ids=native(png)
        lookup=np.zeros(len(labels['faces'])+1,int)
        for face in labels['faces']:lookup[face['id']]=face['surface_id']
        truth=lookup[ids]
        before=common.surface_stats(parts,truth,photo.size)
        after=common.surface_stats(proposed,truth,photo.size)
        case=dict(frame=r['frame'],suite=r['suite'],shape=r['shape'],condition=r['condition'],existing_layout=r['existing_layout'],
            rgb_sha256=r['rgb_sha256'],baseline_surface_sha256=digest(folder/'surfaces.json'),events=events,candidates=log,
            before=before,after=after,gained=sorted(set(after['matched'])-set(before['matched'])),
            lost=sorted(set(before['matched'])-set(after['matched'])),
            paired_delta=after['paired_pixels']-before['paired_pixels'],unmatched_delta=after['unmatched_predictions']-before['unmatched_predictions'])
        cases.append(case)
        save(REVIEW/'external-cases.json',cases)
        print(case['frame'],'changed',len(events),'gained',case['gained'],'lost',case['lost'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-only',action='store_true')
    parser.add_argument('--out',type=Path,default=Path('runs/external'))
    run(parser.parse_args())
