"""Paired floating placement replay with ordinary and native ground-truth surface masks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import os

import cv2
import numpy as np
from PIL import Image, ImageFilter

from benchmark.expanded_baseline import ROOT, SOURCE, WORKTREES, digest, read, save
from benchmark import floating_cloud_replay as paths, floating_build_local_run as build
from benchmark import shared_corner_rows as shared, native_surface_counter as counter
from benchmark import native_corner_registration as registration, garden_anchor_extension as atlas
from benchmark import relative_face_round as assembly, expanded_scores
from benchmark.relative_source_charts import extract
from engine import floating_sky, surface_fields
from engine.mask_first import _image_depth
from engine.normal_regions import _field
from engine.plane_consensus import with_mask
from engine.surface_perimeter import _native_alpha

OUT=ROOT/'runs/native-crop-ceiling'
DATASETS=('visible-builds-v2','visible-builds-v3','decorative-builds-v1')


def native_truth(image, case, cache):
    frame=Path(case['frame'])
    labels=read(frame/'labels.json')
    surfaces={f['id']:f['surface_id'] for f in labels['faces']}
    with np.load(frame/'labels.npz') as r:
        owners=r['face_id']
    lut=np.zeros(int(owners.max())+1,np.int32)
    for fid,sid in surfaces.items():
        assert sid>0
        lut[fid]=sid
    surface=lut[owners]
    size=image.size
    with np.load(cache/'normal.npz') as f:
        normal=_field(f['value'],size)
    with np.load(cache/'depth.npz') as f:
        depth=_image_depth(f['value'],size)
    gray=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),dtype=np.float32)[...,None]
    parts=[]
    for sid in sorted(set(surfaces.values())):
        mask=surface==sid
        direction=np.median(normal[mask],axis=0)
        length=np.linalg.norm(direction)
        direction/=max(length,1e-9)
        p=dict(id=f'plane-{sid}',normal_camera=direction.tolist(),
               relative_inverse_depth=float(np.median(depth[mask])))
        part=with_mask(p,mask,size,gray,np.ones(2))
        alpha,(x,y,x1,y1)=_native_alpha(part['support'],size)
        assert np.array_equal(np.asarray(alpha)>0,mask[y:y1,x:x1]) and int(mask.sum())==part['support_area']
        parts.append(part)
    assert sum(p['support_area'] for p in parts)==np.count_nonzero(owners)
    return sorted(parts,key=lambda p:-p['support_area'])


def fields(case):
    name=case['id']
    cache=WORKTREES/'full-expanded-controls/runs/expanded-baseline-01'/case['dataset']/name/'handoffs'
    if cache.is_dir():
        rec=read(cache.parent/'receipt.json')
        assert rec['source_sha256']==case['rgb_sha256']
        assert all(rec['model_sha256'][k]==surface_fields.MODELS[k][2] for k in ('normal','depth'))
        return cache
    cache=OUT/'fields'/case['dataset']/name
    cache.mkdir(parents=True,exist_ok=True)
    image=Image.open(Path(case['frame'])/'rgb.png').convert('RGB')
    for k,(env,file,_) in surface_fields.MODELS.items():
        dest=cache/(k+'.npz')
        if not dest.is_file():
            os.environ[env]=str(WORKTREES/'normal-round/.deps'/file)
            np.savez_compressed(dest,value=surface_fields.infer(image,k))
    return cache


def paired_case(case, *, arms=('source','native'), source_name='source'):
    cv2.setNumThreads(1)
    name=case['id']
    src=Path(case['frame'])/'rgb.png'
    assert digest(src)==case['rgb_sha256']
    spec=dict(image=str(src),source_sha256=case['rgb_sha256'])
    case=dict(case,variants={'sky':spec})  # Internal floating replay key; pixels are the original screenshot.
    metadata=read(src.with_name('frame.json'))
    assert all(digest(src.with_name(n))==metadata['files'][n] for n in ('camera.json','labels.npz','labels.json'))
    cache=fields(case)
    answers={}
    for arm in arms:
        root=OUT/(source_name if arm=='source' else arm)
        folder=root/'sky'/name/'source-handoff'
        rows=root/'shared'
        native=root/'counter'
        stage=root/'atlas'
        export=root/'assembly'
        output=export/'connected-atlas/sky'/name/'score.json'
        def dense(c,h,image,provenance):
            out=rows/'dense'/name
            extract((spec,name,'sky',h,out))
            packet=out/'dense-charts.json'
            return read(packet)|provenance|dict(baseline_chart_sha256=digest(packet))
        if not output.is_file():
            with ExitStack() as stack:
                def bind(mod,**kwargs):stack.enter_context(patch.multiple(mod,**kwargs))
                bind(paths,RUN=root)
                bind(shared,OUTPUT=rows,dense_packet=dense)
                bind(counter,OUTPUT=native,BASE=rows/'sky')
                bind(registration,OUTPUT=native,BASE=rows/'sky')
                bind(atlas,OUTPUT=stage,NATIVE_COUNTER=native)
                bind(assembly,ASSEMBLY=export,MODES=assembly.MODES|{'connected-atlas':(stage/'connected','charts.json',False,1,2.5,.5,False)})
                if arm=='native':
                    def replace(image,parts):
                        truth=native_truth(image,case,cache)
                        return truth,dict(intervention='exact frontmost surface IDs, image-derived directions/depth',kept_count=len(truth))
                    bind(floating_sky,filter_proposals=replace)
                if case['dataset']=='visible-builds-v2':
                    audit=read(ROOT/'training/framing-audit-v2.json')
                    proof=next(x for x in audit['frames'] if x['id']==name)
                    assert proof['accepted'] and proof['rgb_sha256']==case['rgb_sha256']
                    def frame_read(path):
                        data=read(path)
                        return data|dict(framing=proof) if path==src.with_name('frame.json') else data
                    bind(expanded_scores,read=frame_read)
                build.build_handoff(src,cache,folder)
                shared.predict(case)
                counter.predict(case)
                registration.predict(case)
                atlas.all_walls((case,read(atlas.baseline_file(name))))
                assembly.evaluate((case,'sky','connected-atlas'))
        score=read(output)
        answers[arm]=dict(score=score['score']['fixed'] or score['score']['relaxed_yaw'],
                          planes=len(read(folder/'surfaces.json')['planes']),charts=score['charts_seen'],
                          pack_sha256=score['pack_sha256'])
    save(OUT/'pairs'/case['dataset']/(name+'.json'),dict(id=name,dataset=case['dataset'],source_sha256=case['rgb_sha256'],**answers))
    print('PAIRED',name,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--workers',type=int,default=4)
    cli.add_argument('--dataset',choices=DATASETS,nargs='+',default=DATASETS)
    args=cli.parse_args()
    cases=[]
    for ds in args.dataset:
        base=SOURCE/ds
        for r in read(base/'manifest.json')['frames']:
            if r['split']=='test':
                cases.append(dict(r,dataset=ds,frame=str(base/'data/test/frames'/r['id'])))
    with ProcessPoolExecutor(args.workers) as pool:
        futures={pool.submit(paired_case,c):c for c in cases}
        errors=[]
        for f in as_completed(futures):
            try:f.result()
            except Exception as exc:
                errors.append(futures[f]['id'])
                print('FAILED',futures[f]['id'],repr(exc),str(exc.__cause__),flush=True)
        if errors:raise RuntimeError(errors)
