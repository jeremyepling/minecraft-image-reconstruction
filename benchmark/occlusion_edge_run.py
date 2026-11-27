"""Freeze single-image choices, then replay one real placement alternate for each informative view."""
import argparse
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext, redirect_stdout
from io import BytesIO
import os
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
from PIL import Image
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.expanded_baseline import ROOT, STABLE, WORKTREES, read, save, digest
from benchmark.occlusion_edge_depth import propose
from engine import visible_placement as vp
from engine.visible_raster import observed_support

CONTROL = WORKTREES/'surface-depth-round-controls/benchmark/reviews/surface-depth-round-controls-01/controls.json'
OUT = ROOT/'runs/oe'


def plan(f):
    key = f['id']
    src = STABLE.parent/f['rgb']['path']
    assert digest(src)==f['rgb']['sha256']
    with Image.open(src) as opened:
        image = opened.convert('RGB')
    dest = f['destination']
    if f['dataset']=='fixed-104':
        entry = next(e for e in fixed.fixed_frames() if e['id']==key)
        parts = fixed.inference_inputs(entry).parts
        status = dest['historical_placement']
    else:
        build = STABLE.parent/dest['root']
        for name in ('surfaces.json','build-manifest.json'):
            assert digest(build/name)==dest['artifacts'][name]['sha256']
        parts = read(build/'surfaces.json')['planes']
        status = read(build/'build-manifest.json')['placement']
    P = np.asarray(status['P'])
    eye = np.linalg.solve(P[:,:3],-P[:,3])
    shape = tuple((n+1)//4 for n in image.size[::-1])
    masks = {p['id']:observed_support(p['support'],shape,image.size) for p in parts if p.get('support')}
    rows = [r for r in status['plane_offsets'] if r['axis']!=1 and r['id'] in masks]
    report = dict(id=key, rgb_sha256=f['rgb']['sha256'],pack_sha256=f['pack'].get('member_sha256',f['pack']['sha256']),
                  source_rule_sha256=digest(ROOT/'benchmark/occlusion_edge_depth.py'))
    if rows:
        row = max(rows,key=lambda r:masks[r['id']].sum())
        union = np.logical_or.reduce(list(masks.values()))
        mask = masks[row['id']]
        labels, audit = propose(image,mask,union,P,eye,row['axis'],row['level'])
        report.update(carrier=row['id'],initial=row['level'],axis=row['axis'],audit=audit)
        (OUT/key).mkdir(parents=True,exist_ok=True)
        np.savez_compressed(OUT/key/'labels.npz',labels=labels,mask=mask)
    save(OUT/key/'plan.json',report)
    print('PLAN',key, len(report.get('audit',{}).get('offers',[])),flush=True)


def hook(selection, labels, expected):
    estimate=vp._estimate_plane_cells
    def split(image,parts,masks,P,eye,*args,**kwargs):
        resolved={}
        original=kwargs.get('reconcile')
        def remember(rows):
            rows=original(rows) if original else rows
            resolved.update((i,(a,l)) for i,a,l in rows)
            return rows
        cells=estimate(image,parts,masks,P,eye,*args,**(kwargs|dict(reconcile=remember)))
        for i,(axis,initial) in resolved.items():
            if parts[i]['id']!=selection['carrier']:
                continue
            assert np.array_equal(masks[i],expected) and initial==selection['initial'] and axis==selection['axis']
            offer=selection['audit']['offers'][0]
            replaced=set()
            region=labels==offer['back']
            for subset,level in ((region,offer['level']),(masks[i]&~region,initial)):
                subset_cells,=estimate(image,[parts[i]],[subset],P,eye,lambda *_:level,partial=kwargs.get('partial'))
                replaced.update(subset_cells)
            cells[i]=replaced
        return cells
    return patch.object(vp,'_estimate_plane_cells',split)


def export(f, *, control=False):
    key=f['id']
    root=OUT/key
    source=read(root/'plan.json')
    if not source.get('audit',{}).get('offers'):
        return
    from benchmark.camera_horizon_replay import replay
    from benchmark.expanded_baseline import COHORTS,SOURCE,frames
    from benchmark.expanded_scores import evaluate_pack,load_reference
    from benchmark.mask_blocks import decoded_pack
    with np.load(root/'labels.npz') as archive:
        labels,mask=archive['labels'],archive['mask']
    print('EXPORT',key,flush=True)
    intervention=nullcontext() if control else hook(source,labels,mask)
    label='control' if control else 'alt'
    with intervention, (root/(label+'.log')).open('w') as log, redirect_stdout(log):
        if f['dataset']=='fixed-104':
            entry=next(e for e in fixed.fixed_frames() if e['id']==key)
            inputs=fixed.inference_inputs(entry)
            result=fixed.replay_placement(inputs)
            pack=root/(label+'.zip')
            pack.write_bytes(result.pack)
            case,after=fixed.score_result(entry,result)
            receipt=f['pack']
            with ZipFile(STABLE.parent/receipt['path']) as archive:
                incumbent=archive.read(receipt['member'])
            local=decoded_pack(BytesIO(incumbent))
            _,before=fixed.score_result(entry,fixed.ReplayResult(incumbent,dict.fromkeys(local,'fallback'),inputs.camera))
            truth={tuple(b['xyz_world']) for b in fixed.fixed_reference(entry)[0]['blocks']}
        else:
            entry=next(e for e in frames(SOURCE,COHORTS) if e['id']==key)
            build=STABLE.parent/f['destination']['root']
            status=read(build/'build-manifest.json')['placement']
            camera=dict(P=status['P'],scale_status=status['camera_status'],reason=status['reason'],camera_source=status.get('camera_source'))
            case=replay(entry,camera,STABLE.parent/f['replay']['root'],root/label)
            pack=root/label/'reconstruction.zip'
            _,before=evaluate_pack(entry,build/'reconstruction.zip',status)
            _,after=evaluate_pack(entry,pack,status)
            truth=load_reference(entry).truth.keys()
    if control:
        from benchmark.camera_model_controls import inventory_hash
        decoded=inventory_hash(decoded_pack(pack))
        assert decoded==f['decoded_sha256']
        save(root/'control.json',dict(decoded_sha256=decoded,pack_sha256=digest(pack)))
        return
    assert len(before.keys()&truth)==f['baseline']['geometry']['strict']['correct']
    plus,minus=after.keys()-before.keys(),before.keys()-after.keys()
    save(root/'alternate.json',dict(geometry=case['geometry'],scores=case['scores'],pack_sha256=digest(pack),
         paired=dict(gained=len(plus&truth),lost=len(minus&truth),extras_added=len(plus-truth),extras_removed=len(minus-truth))))
    print('SCORED',key,case['geometry']['strict']['f1'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',choices=['plan','export','control'])
    p.add_argument('group',choices=['dev','hold'])
    p.add_argument('--workers',default=4,type=int)
    args=p.parse_args()
    fs=read(CONTROL)['pilot']['frames']
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(plan if args.step=='plan' else partial(export,control=args.step=='control'),fs[:4] if args.group=='dev' else fs[4:]))
