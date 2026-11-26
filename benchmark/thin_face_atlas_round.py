"""Probe atlas-supported native horizontal faces on the frozen visible-build datasets."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from benchmark.texture_camera_eval import DATA, ROOT, STABLE, checked, digest, frames, read, save
from benchmark import visible_depth_native as native
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.visible_depth_transfer import structure
from engine.camera_atlas_floor import _atlas
from engine.thin_atlas_faces import StripEvidence, choose, recover
from engine.visible_horizontal_faces import _keep_prior_visible
from training.visible_labels import visible_labels
import numpy as np
from PIL import Image

TREES = STABLE.parent/(STABLE.name+'-worktrees')
PREVIOUS = TREES/'combined-visible-recovery/runs/triple'
EXTRA = TREES/'visible-recovery-extra-views/runs/extra-views'
COHORTS = ('validation','test','regression','native','old','fresh','extra')


def fixture(split):
    if split=='old': return STABLE/'runs/visible-depth-rounds/transfer'
    if split=='fresh': return STABLE/'runs/canopy-placement/confirmation'
    if split=='extra': return EXTRA
    return None


def job_list(split,out,modes,count):
    base = EXTRA/'candidate' if split=='extra' else PREVIOUS/split
    baseline = {r['id']:r for r in read(base/'cases.json')}
    source = fixture(split)
    samples = read(source/'screen.json')['selected'] if source else (native.rows() if split=='native' else frames(DATA,split))
    return [(split,f,source,base,baseline[f['id']],out,modes) for f in samples[:count]]


def evaluate(job):
    split,f,fixture_path,base,ref,out,modes = job
    key,h = f['id'],f['files']
    dest = out/split/key
    dest.mkdir(exist_ok=True,parents=True)
    if fixture_path:
        parts = read(checked(fixture_path/'upstream'/key/'surfaces.json',ref['surface_sha256']))['planes']
        camera = f['camera']
    else:
        src = read(checked(STABLE/'runs/visible-depth-rounds/inputs'/(key+'.json'),ref['input_sha256']))
        parts,camera = src['parts'],src['camera']
    if fixture_path or split=='native':
        stem = f.get('source_id',key)
        folder = native.BUILDS/f['directory']/'captures'
        rgb,pose,world = [folder/(stem+s) for s in ('.png','.json','-blocks.json')]
    else:
        folder = DATA/'data'/split/f['directory']
        rgb,pose,world = [folder/s for s in ('rgb.png','camera.json','world-readback.json')]
    prior = set(decoded_pack(checked(base/key/'reconstruction.zip',ref['pack_sha256'])))
    assert h['rgb.png']==ref['rgb_sha256']
    image = Image.open(checked(rgb,h['rgb.png'])).convert('RGB')
    atlas = _atlas('26.3',STABLE/'runs/26.3/atlas')
    assert atlas is not None
    ctx = StripEvidence(image,parts,camera,prior,atlas) if prior and camera['P'] is not None else None
    pool = {}
    if ctx:
        pool = {c:r for c in ctx.candidates() if (r:=ctx.measure(c)) is not None}
    results = {}
    for mode in modes:
        decision,sep,repeat=mode.partition('x')
        selected = choose(pool,decision)
        if selected:
            joint = ctx.joint(selected)
            verified = {c:r for c in selected if (r:=ctx.measure(c,joint)) is not None}
            selected = choose(verified,decision,pool)
        prediction = _keep_prior_visible(image.size,ctx.P,ctx.eye,prior,selected) if selected else prior
        if selected and sep:
            for _ in range(1,int(repeat)):
                extended=recover(image,parts,camera,prediction,atlas,decision)
                if extended==prediction:
                    break
                prediction=extended
        pack = dest/(mode+'.zip')
        results[mode] = _write_read(pack,prediction)
    # Load scoring data only after serializing every candidate.
    recorded = read(checked(pose,h['camera.json']))
    raw = read(checked(world,h['world-readback.json']))
    if fixture_path or split=='native':
        states = {tuple(b[k] for k in 'xyz'):b for b in raw}
        labels,_,_ = visible_labels(recorded,states,(0,0,0))
        occupied = set(structure(raw,64))
    else:
        labels = read(checked(folder/'labels.json',h['labels.json']))
        occupied = set(structure(raw,f['ground_y']))
    truth = {tuple(b['xyz_world']):b['block_id'] for b in labels['blocks']}
    T = registration(np.asarray(camera['P']),recorded) if camera['P'] is not None else None
    before = registered(prior,T)
    trace = [dict(r,cell=c,target=bool(registered([c],T)&truth.keys()),
                  hidden=bool(registered([c],T)&(occupied-truth.keys()))) for c,r in sorted(pool.items())]
    save(dest/'trace.json',trace)
    row = dict(id=key,split=split,layout=f.get('layout',f.get('shape')),rgb_sha256=ref['rgb_sha256'],
               input_sha256=ref.get('input_sha256',ref.get('surface_sha256')),baseline_pack_sha256=ref['pack_sha256'],
               proposed=len(trace),possible=sum(r['target'] for r in trace),methods={})
    for mode,cells in results.items():
        world = registered(cells,T)
        score=placement_scores(world,before,truth,occupied)
        assert score['baseline_translation']==tuple(ref['translation']) or list(score['baseline_translation'])==list(ref['translation'])
        m=score['relative_geometry']
        assert m['correct']-m['gained']+m['lost']==ref['relative_geometry']['correct']
        row['methods'][mode]=dict(metrics=m,translation=score['translation'],sha256=digest(dest/(mode+'.zip')))
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sets',nargs='+',default=['train'])
    p.add_argument('--modes',nargs='+',default=['strong','moderate','union'])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    jobs=[j for s in args.sets for j in job_list(s,args.out,args.modes,args.limit)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(evaluate,jobs))
    for mode in args.modes:
        scores=Counter()
        for row in rows:
            scores.update(row['methods'][mode]['metrics'])
        print(mode,dict(scores),flush=True)
    print('Proposal headroom',sum(r['possible'] for r in rows),sum(r['proposed'] for r in rows))
    save(args.out/'scores.json',dict(source_sha256=digest(ROOT/'engine/thin_atlas_faces.py'),rows=rows))


if __name__=='__main__':
    main()
