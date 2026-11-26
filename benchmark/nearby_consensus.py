"""Test agreement of two out-of-family scores on unselected, photographed positions."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product

import numpy as np

from benchmark.nearby_face_trial import RUN, PUBLIC, SPLITS, cases, art_sample, labels, load_prediction, method
from benchmark.nearby_thin_overlap import THIN, paths
from benchmark.mask_blocks import decoded_pack, _write_read, registered
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import read, save
from engine.camera_atlas_floor import _atlas

spec=spec_from_file_location('thin_face_peer',THIN/'engine/thin_atlas_faces.py')
thin=module_from_spec(spec)
spec.loader.exec_module(thin)
CONFIGS=list(product((.08,.16,.25,.35),(.35,.55,.70,.80)))


def run(job):
    index,case,config=job
    split,f,*_=case
    key=f['id']
    path=paths(split,key)
    data=art_sample(case)
    image,parts,camera,prior=load_prediction(case)
    raw=read(path/'trace.json')
    rows={tuple(row['cell']):{k:v for k,v in row.items() if k not in ('target','hidden','cell')} for row in raw}
    s,t,axis=data['edges'].T
    model=read(method.MODEL)
    if split=='train':
        art=read(THIN/'runs/folds'/(f['family']+'.json'))
        with np.load(RUN/'held-atlas-3.npz') as file:
            dst,src=file[str(index)]
        accepted=dst>=model['add']
        removed=accepted&(src<=model['drop'])&(dst-src>=model['margin'])
        decision=set(zip(s[removed],t[removed])),set(t[accepted]),set(zip(t[accepted],axis[accepted]))
        depth=method.apply(image,parts,camera,prior,data,decision)
        atlas=set(decoded_pack(THIN/'runs/family-round'/split/key/'rank85.zip'))
    else:
        art=read(thin.WEIGHTS)
        dst=method.predict_forest(model['destination'],data['features'])
        depth=set(decoded_pack(RUN/'predictions'/split/key/'reconstruction.zip'))
        atlas=set(decoded_pack(path/'rank85.zip'))
    base=depth|(atlas-prior)
    atr=thin.probabilities(rows,art)
    pairs={}
    for tg,p,a in zip(t,dst,axis):
        cell=tuple(data['points'][tg].tolist())
        if cell not in rows or cell in base or cell in prior:
            continue
        row=rows[cell]
        if row['split']<.25 or row['phase']<=.02 or row['texels']<8:
            continue
        # Both networks must have witnessed the horizontal face itself.
        if a==1:
            pairs[cell]=max(p,pairs.get(cell,0.))
    candidates={c for c,p in pairs.items() if p>=.08 and atr[c]>=.35}
    atlas_art=_atlas('26.3') if candidates else None
    ctx=thin.StripEvidence(image,parts,camera,base,atlas_art) if atlas_art is not None else None
    thresholds=CONFIGS if config is None else [tuple(config)]
    predictions={}
    cache={}
    for lo,hi in thresholds:
        proposed=frozenset(c for c in candidates if pairs[c]>=lo and atr[c]>=hi)
        if proposed not in cache:
            live=set()
            if proposed:
                ownership=ctx.joint(proposed)
                for c in proposed:
                    r=ctx.measure(c,ownership)
                    if r and r['split']>=.25 and r['phase']>.02 and r['texels']>=8:
                        live.add(c)
            cache[proposed]=thin._keep_prior_visible(image.size,ctx.P,ctx.eye,base,live) if live else base
        predictions[lo,hi]=cache[proposed]
    dest=RUN/'consensus'/split/key
    dest.mkdir(parents=True,exist_ok=True)
    decoded={k:_write_read(dest/(str(i)+'.zip'),cells) for i,(k,cells) in enumerate(predictions.items())}
    T,truth,occupied=labels(case,camera['P'])
    before=registered(base,T)
    return [dict(score=placement_scores(registered(cells,T),before,truth,occupied)['relative_geometry'],
                 lo=lo,hi=hi,family=f.get('family'),id=key,split=split) for (lo,hi),cells in decoded.items()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('train','verify'))
    args=p.parse_args()
    splitset=('train',) if args.stage=='train' else SPLITS
    config=None if args.stage=='train' else read(PUBLIC/'consensus-train.json')['selected']
    rows=[]
    jobs=[(i,j,config) for split in splitset for i,j in enumerate(cases(split))]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for result in pool.map(run,jobs):
            rows.extend(result)
    if config is None:
        variants=[]
        for lo,hi in CONFIGS:
            subset=[r for r in rows if r['lo']==lo and r['hi']==hi]
            family={}
            for r in subset:
                counts=family.setdefault(r['family'],Counter())
                counts.update(r['score'])
            gain=sum(m['gained'] for m in family.values())
            extra=sum(m['extra_added'] for m in family.values())
            utility=gain-extra+2*min(m['gained']-m['extra_added'] for m in family.values())
            variants.append(dict(lo=lo,hi=hi,gained=gain,extras=extra,utility=utility,
                                 families={k:[m['gained'],m['extra_added']] for k,m in family.items()}))
        best=max(variants,key=lambda r:(r['utility'],r['lo']+r['hi']))
        save(PUBLIC/'consensus-train.json',dict(selected=[best['lo'],best['hi']],variants=variants))
        print(best,flush=True)
    else:
        save(PUBLIC/'consensus-evaluation.json',dict(selected=config,rows=rows))
        for s in ('fixed','extra'):
            m=Counter()
            for r in rows:
                if r['split'] in (SPLITS[:-1] if s=='fixed' else (s,)):
                    m.update(r['score'])
            print(s,dict(m),flush=True)


if __name__=='__main__':
    main()

