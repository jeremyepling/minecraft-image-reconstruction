"""Screen MobileSAM mask proposals with automatic image-space prompts on the frozen exterior suite."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, read, save, digest
from benchmark.mask_metrics import footprint
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.surface_metrics import DOWNSAMPLE, _one_to_one
from runtime.mode import image_only


WALLS = ROOT/'benchmark/reviews/separation-01'
METHOD = 'wall-depth-anything'
SOURCE = 'f706ad9c4eb7f219c00d9050e46328518ffb65d2'
WEIGHTS = '6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f'
SETTINGS = dict(grid_step_native_px=32, near_centroid_per_existing_region=True, batch=12,
                minimum_mask_area_samples=24, predicted_iou_cutoff=.88, stability_cutoff=.95,
                max_mask_iou_after_nms=.8, minimum_building_fraction=.8)


def get_predictor(source,device):
    import torch
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE:
        raise ValueError('Unexpected MobileSAM source revision')
    weight=source/'weights/mobile_sam.pt'
    if digest(weight)!=WEIGHTS:
        raise ValueError('Unexpected MobileSAM TinyViT checkpoint')
    sys.path.insert(0,str(source))
    from mobile_sam import SamPredictor,sam_model_registry
    net=sam_model_registry['vit_t']()
    net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
    return SamPredictor(net.eval().to(device))


def prompt_grid(parts,size):
    w,h=size
    shape=(h//DOWNSAMPLE,w//DOWNSAMPLE)
    union=np.zeros(shape,bool)
    points=set()
    for p in parts:
        mask=footprint(p,shape,size,True)
        union|=mask
        yy,xx=np.where(mask)
        if len(xx):
            near=int(np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2))
            points.add((int(xx[near]),int(yy[near])))
    step=SETTINGS['grid_step_native_px']//DOWNSAMPLE
    y,x=np.nonzero(union[step//2::step,step//2::step])
    points.update(zip((x*step+step//2).tolist(),(y*step+step//2).tolist()))
    return np.asarray(sorted(points),dtype=np.float32).reshape(-1,2)*DOWNSAMPLE+.5*DOWNSAMPLE,union


def make_proposals(predictor, image, parts):
    import torch
    points,union=prompt_grid(parts,image.size)
    model_start=perf_counter()
    predictor.set_image(np.asarray(image))
    if predictor.device.type=='cuda':
        torch.cuda.synchronize(predictor.device)
    model_time=perf_counter()-model_start
    xy=predictor.transform.apply_coords(points,predictor.original_size)
    candidates=[]
    kept_pixels=[]
    seen=set()
    for start in range(0,len(points),SETTINGS['batch']):
        prompts=torch.as_tensor(xy[start:start+SETTINGS['batch']],device=predictor.device,dtype=torch.float32)[:,None,:]
        labels=torch.ones(prompts.shape[:2],device=predictor.device,dtype=torch.int64)
        with torch.inference_mode():
            model_start=perf_counter()
            native,qualities,_=predictor.predict_torch(prompts,labels,multimask_output=True,return_logits=True)
            if predictor.device.type=='cuda':
                torch.cuda.synchronize(predictor.device)
            model_time+=perf_counter()-model_start
            samples=native[:,:,DOWNSAMPLE//2::DOWNSAMPLE,DOWNSAMPLE//2::DOWNSAMPLE].contiguous()
            stable=(samples>1).sum((-1,-2))/(samples>-1).sum((-1,-2)).clamp_min(1)
            grid=(samples>0).cpu().numpy()
            qualities=qualities.cpu().numpy()
            stable=stable.cpu().numpy()
        for i in range(len(prompts)):
            for slot,mask in enumerate(grid[i]):
                area=int(mask.sum())
                if area<SETTINGS['minimum_mask_area_samples']:
                    continue
                packed=np.packbits(mask).tobytes()
                if packed in seen:
                    continue
                seen.add(packed)
                yy,xx=np.where(mask)
                record=dict(point=points[start+i].astype(int).tolist(),slot=slot,area=area,
                    quality=float(qualities[i,slot]),stability=float(stable[i,slot]),
                    building_fraction=float(union[mask].mean()),
                    bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)])
                candidates.append(record)
                kept_pixels.append(mask)
    rank=sorted(range(len(candidates)),key=lambda i:-candidates[i]['quality'])
    chosen=[]
    for i in rank:
        m=candidates[i]
        if (m['quality']<SETTINGS['predicted_iou_cutoff'] or m['stability']<SETTINGS['stability_cutoff'] or
                m['building_fraction']<SETTINGS['minimum_building_fraction']):
            continue
        duplicate=False
        for j in chosen:
            n=candidates[j]
            if min(m['area'],n['area'])<=SETTINGS['max_mask_iou_after_nms']*max(m['area'],n['area']):
                continue
            a,b=m['bbox'],n['bbox']
            x=max(a[0],b[0])
            y=max(a[1],b[1])
            xx=min(a[2],b[2])
            yy=min(a[3],b[3])
            if x<xx and y<yy:
                intersection=int(np.count_nonzero(kept_pixels[i][y:yy,x:xx]&kept_pixels[j][y:yy,x:xx]))
                if intersection>SETTINGS['max_mask_iou_after_nms']*(m['area']+n['area']-intersection):
                    duplicate=True
                    break
        if not duplicate:
            chosen.append(i)
    masks=np.asarray(kept_pixels,bool).reshape(-1,*union.shape)
    return points,candidates,chosen,masks,model_time


def score(masks, indices, ref):
    ids=ref['id_map']
    eligible=[pid for pid,p in ref['patches'].items() if p['eligible']]
    areas=np.bincount(ids.ravel(),minlength=max(ref['patches'])+1)
    pairs=[]
    for i in indices:
        hits=np.bincount(ids[masks[i]],minlength=len(areas))
        size=hits.sum()
        for pid in eligible:
            iou=hits[pid]/(size+areas[pid]-hits[pid])
            if iou>=.5:
                pairs.append((iou,int(i),pid))
    assignments=_one_to_one(pairs)
    pair_ious={(i,pid):v for v,i,pid in pairs}
    return {pid:dict(proposal=i,iou=round(float(pair_ious[i,pid]),6)) for i,pid in assignments}


def summarize(rows):
    total=Counter()
    for r in rows:
        total.update(r['counts'])
    return dict(total)


@image_only()
def run(source,out,device):
    import torch, torchvision, timm, PIL
    predictor=get_predictor(source,device)
    cohort,_,fixtures=groups()
    manifest=read(WALLS/'manifest.json')
    if digest(ARCHIVE)!=manifest['reference_sha256'] or fixtures!=manifest['fixtures']:
        raise ValueError('Wrong reference set')
    if digest(WALLS/'cases.json')!=manifest['artifacts']['cases.json']:
        raise ValueError('Wrong control cases')
    rows=read(WALLS/'cases.json')
    captures={f'{c.floor}/{c.id}':c for g in cohort.values() for c in g}
    if set(captures)!={r['key'] for r in rows} or len(rows)!=56:
        raise ValueError('Expected fixed 56 pictures')
    out.mkdir(parents=True,exist_ok=False)
    (out/'proposals').mkdir()
    save(out/'protocol.json',dict(settings=SETTINGS,source=SOURCE,weights=WEIGHTS,
         purpose='Model screen; point prompts use a fixed grid inside existing foreground and a point per region. No reference prompt or ground-truth crop.',
         scoring='Best possible one-to-one match among all distinct generated masks at IoU >= .50, plus reference-free score/stability and mask-overlap suppression. Outputs may overlap. A match is an evaluation observation, not permission to replace the baseline.'))
    infer_time=0.
    model_time=0.
    scoring_time=0.
    generated={}
    results=[]
    for development in (True,False):
        for r in rows:
            if (r['cohort']=='development')!=development:
                continue
            key=r['key']
            sha=r['sha256']
            case=captures[key]
            src=WALLS/r['predictions']
            if digest(src)!=manifest['artifacts'][r['predictions']] or digest(case.path('image'))!=sha:
                raise ValueError(f'Input mismatch: {key}')
            with gzip.open(src,'rt',encoding='utf8') as f:
                parts=json.load(f)[METHOD]
            fingerprint=json.dumps([p['support'] for p in parts],separators=(',',':'),sort_keys=True)
            if sha in generated and generated[sha]!=fingerprint:
                raise ValueError('Identical pixels have different prompting inputs')
            stem=out/'proposals'/sha
            if sha not in generated:
                with Image.open(case.path('image')) as f:
                    image=f.convert('RGB')
                start=perf_counter()
                pts,proposals,selected,masks,model_elapsed=make_proposals(predictor,image,parts)
                elapsed=perf_counter()-start
                infer_time+=elapsed
                model_time+=model_elapsed
                np.savez_compressed(stem.with_suffix('.npz'),shape=np.asarray(masks.shape),
                    packed=np.packbits(masks,axis=2))
                stem.with_suffix('.json.gz').write_bytes(gzip.compress(json.dumps(dict(points=pts.tolist(),
                    candidates=proposals,selected=selected),allow_nan=False,
                    separators=(',',':')).encode(),mtime=0))
                generated[sha]=fingerprint
            else:
                with gzip.open(stem.with_suffix('.json.gz'),'rt',encoding='utf8') as f:
                    selected=json.load(f)['selected']
                with np.load(stem.with_suffix('.npz'),allow_pickle=False) as f:
                    masks=np.unpackbits(f['packed'],count=int(f['shape'][2]),axis=2).astype(bool)
            # Score only after masks and the image-only filtered set are independently persisted.
            score_start=perf_counter()
            _,ref=recorded(case)
            if tuple(ref['id_map'].shape)!=masks.shape[1:]:
                raise ValueError('Scoring lattice differs from stored masks')
            control=set(r['scopes'][METHOD]['matched'])
            eligible={pid for pid,p in ref['patches'].items() if p['eligible']}
            down={pid for pid in eligible if ref['patches'][pid]['normal'][1]==-1}
            pool=score(masks,range(len(masks)),ref)
            auto=score(masks,selected,ref)
            extra=set(pool)-control
            chosen_extra=set(auto)-control
            scoring_time+=perf_counter()-score_start
            counts=dict(frames=1,eligible=len(eligible),baseline=len(control),pool_proposals=len(masks),
                pool_matched=len(pool),pool_additional=len(extra),pool_extra_undersides=len(extra&down),
                selected_proposals=len(selected),selected_matched=len(auto),selected_unmatched=len(selected)-len(auto),
                selected_additional=len(chosen_extra),selected_extra_undersides=len(chosen_extra&down),
                selected_lost_if_used_alone=len(control-set(auto)))
            results.append(dict(key=key,layout=r['layout'],cohort=r['cohort'],sha256=sha,
                                counts=counts,baseline_matches=sorted(control),pool_matches=pool,selected_matches=auto,
                                additional=sorted(extra),selected_additional=sorted(chosen_extra)))
        print('development' if development else 'regression',summarize([r for r in results if (r['cohort']=='development')==development]),flush=True)
    scores=dict(all=summarize(results),development=summarize([r for r in results if r['cohort']=='development']),
        regression=summarize([r for r in results if r['cohort']!='development']),
        layouts={k:summarize([r for r in results if r['layout']==k]) for k in sorted({r['layout'] for r in results})})
    save(out/'cases.json',results)
    save(out/'scores.json',scores)
    save(out/'manifest.json',dict(run_id=out.name,source_url='https://github.com/ChaoningZhang/MobileSAM',
        source_revision=SOURCE,checkpoint_sha256=WEIGHTS,runner_sha256=digest(Path(__file__)),
        reconstruction_code_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        reproduction="& '../mask-scalelsd/runs/env/Scripts/python.exe' -B -m benchmark.mobile_region_probe --source ../surface-recovery-round/runs/mobile-sam-source --out runs/mobile-regions-replay",
        artifact_location=out.resolve().relative_to(ROOT.resolve(),walk_up=True).as_posix(),
        model='MobileSAM TinyViT, vit_t',device=torch.cuda.get_device_name() if device.startswith('cuda') else device,
        versions=dict(torch=torch.__version__,torchvision=torchvision.__version__,timm=timm.__version__,Pillow=PIL.__version__,numpy=np.__version__),
        runtime=dict(unique_images=len(generated),inference_and_postprocessing_s=infer_time,model_s=model_time,
                     prompt_and_mask_processing_s=infer_time-model_time,reference_load_and_scoring_s=scoring_time),
        baseline_manifest=digest(WALLS/'manifest.json'),reference_sha256=digest(ARCHIVE),fixtures=fixtures,
        protocol='protocol.json',scores='scores.json',cases='cases.json',inputs='Saved original RGB plus current Wall Separator masks for point placement.',
        artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,required=True,help='Checkout of the official MobileSAM release including weights/mobile_sam.pt')
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--device',default='cuda')
    args=cli.parse_args()
    run(args.source,args.out,args.device)
