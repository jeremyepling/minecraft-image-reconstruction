"""Recount the supported wall component and pass the new level to ordinary placement."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import wraps
from hashlib import sha256
from io import BytesIO
from unittest.mock import patch
from zipfile import ZipFile
from benchmark.return_face_round import CONTROL, RUN, PC, source
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.expanded_baseline import SOURCE, COHORTS, frames, read, save
from benchmark.camera_horizon_replay import replay
from benchmark.expanded_scores import evaluate_pack, load_reference
from benchmark.mask_blocks import decoded_pack
from engine import surface_model, foreground_conflicts
from engine.plane_consensus import with_mask
from engine.surface_perimeter import with_outline
from PIL import Image, ImageFilter
import numpy as np


def incumbent(row):
    r=row['pack']
    if 'member' in r:
        with ZipFile(PC/r['path']) as z:
            return z.read(r['member'])
    return (PC/r['path']).read_bytes()


def install(stack, row, choice, report):
    if choice is None:
        return
    folder=RUN/row['id']
    inventory=read(folder/'source.json')
    runs=read(folder/'masks.json')[str(choice['group'])]
    parent=inventory['parent']
    child=parent+'-return-component'
    def wrap(fn):
        @wraps(fn)
        def edited(image,parts,*args,**kwargs):
            for i,p in enumerate(parts):
                if p['id']!=parent or p.get('return_recounted') or any(q['id']==child for q in parts):
                    continue
                sw,sh=p['support']['size']
                mask=np.zeros((sh,sw),bool)
                for y,a,b in p['support']['runs']:
                    mask[y,a:b]=True
                small=np.zeros((image.height//4,image.width//4),bool)
                for y,a,b in runs:
                    small[y,a:b]=True
                # Clip the observed component to the original support at its own resolution.
                childmask=mask & np.asarray(Image.fromarray(small).resize((sw,sh),Image.Resampling.NEAREST))
                gray=np.asarray(image.convert('L').resize((sw,sh),Image.Resampling.BOX)
                    .filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
                pieces=[]
                for name,support in ((parent,mask&~childmask),(child,childmask)):
                    if support.any():
                        part=with_mask(p,support,(sw,sh),gray,np.array(image.size)/[sw,sh])
                        pieces.append(with_outline(dict(part,id=name,return_recounted=True)))
                if not any(q['id']==child for q in pieces):
                    raise ValueError('component missing before counting')
                parts[i:i+1]=pieces
                report['counts']+=1
                break
            return fn(image,parts,*args,**kwargs)
        return edited
    stack.enter_context(patch.object(fixed,'recount',wrap(fixed.recount)))
    stack.enter_context(patch.object(surface_model,'_count_regions',wrap(surface_model._count_regions)))
    # An inset host is reconstructed from its original mask and counted again.
    # Apply the identical component partition at that count entry as well.
    inset_count=surface_model._inset_layout_counts
    def final_host_count(image,version,parts,*args,**kwargs):
        return wrap(lambda image,parts: inset_count(image,version,parts,*args,**kwargs))(image,parts)
    stack.enter_context(patch.object(surface_model,'_inset_layout_counts',final_host_count))
    original=foreground_conflicts.ForegroundConflicts.resolve
    def resolve(self):
        result=original(self)
        if child in self.planes:
            result[child]=choice['target_level']
            self.levels[child]=choice['target_level']
            report['placement']+=1
        return result
    stack.enter_context(patch.object(foreground_conflicts.ForegroundConflicts,'resolve',resolve))


def export(task):
    row,mode=task
    key=row['id']
    plan=read(RUN/key/'source.json')
    choice=plan.get(mode)
    out=RUN/key/mode
    out.mkdir(parents=True,exist_ok=True)
    report=dict(counts=0,placement=0)
    payload=incumbent(row)
    prior_sha=sha256(payload).hexdigest()
    assert prior_sha==row['pack'].get('member_sha256',row['pack']['sha256'])
    with ExitStack() as stack:
        install(stack,row,choice,report)
        if row['dataset']=='fixed-104':
            e=next(e for e in fixed.fixed_frames() if e['id']==key)
            inputs=fixed.inference_inputs(e)
            result=fixed.replay_placement(inputs)
            candidate=result.pack
            (out/'reconstruction.zip').write_bytes(candidate)
            current,pred=fixed.score_result(e,result)
            old=decoded_pack(BytesIO(payload))
            _,prior=fixed.score_result(e,fixed.ReplayResult(payload,dict.fromkeys(old,'fallback'),inputs.camera))
            truth={tuple(b['xyz_world']) for b in fixed.fixed_reference(e)[0]['blocks']}
        else:
            e=next(e for e in frames(SOURCE,COHORTS) if e['id']==key)
            _,_,camera,_=source(row)
            current=replay(e,camera,PC/row['replay']['root'],out)
            status=read(out/'build-manifest.json')['placement']
            candidate=(out/'reconstruction.zip').read_bytes()
            _,pred=evaluate_pack(e,candidate,status)
            _,prior=evaluate_pack(e,payload,status)
            truth=load_reference(e).truth.keys()
    if choice and not all(report.values()):
        raise ValueError(f'{key}: selected component missed count or placement: {report}')
    if not choice and decoded_pack(BytesIO(payload))!=decoded_pack(BytesIO(candidate)):
        raise ValueError(f'{key}: no-op differs from released export')
    assert len(truth & prior.keys())==row['baseline']['geometry']['strict']['correct']
    paired=dict(gained=len((pred.keys()-prior.keys())&truth),lost=len((prior.keys()-pred.keys())&truth),
                extras_added=len((pred.keys()-prior.keys())-truth),extras_removed=len((prior.keys()-pred.keys())-truth))
    record=dict(id=key,mode=mode,source_sha256=row['rgb']['sha256'],incumbent_sha256=prior_sha,
                pack_sha256=sha256(candidate).hexdigest(),choice=choice,recount=report,paired=paired,case=current)
    save(out/'score.json',record)
    print(key,mode,paired,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--mode',choices=['chosen','forced','control'],default='chosen')
    p.add_argument('--ids',nargs='+')
    args=p.parse_args()
    tasks=[]
    for row in read(CONTROL)['pilot']['frames']:
        plan=read(RUN/row['id']/'source.json')
        if args.ids and row['id'] not in args.ids:
            continue
        if args.mode=='forced' and (plan['chosen'] or not plan['forced']):
            continue
        tasks.append((row,args.mode))
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(export,tasks))

if __name__=='__main__':
    main()
