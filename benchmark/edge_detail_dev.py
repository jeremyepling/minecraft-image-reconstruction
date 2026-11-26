"""Development-only native pixel audit for successive settings of the photographic edge collar."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import read, save, revision
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark import edge_detail
from benchmark.mask_metrics import measure
from benchmark.normal_round import groups
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import recorded
from benchmark.wall_join_test import checked, read_gzip
from engine.strip_neighbor import clean_surfaces
from engine.surface_perimeter import _native_alpha


def pixels(part, size):
    mask=np.zeros(size[::-1],bool)
    alpha,(x,y,r,b)=_native_alpha(part['support'],size)
    mask[y:b,x:r]=np.asarray(alpha,bool)
    return mask


def evaluate(job):
    case,row,digest,settings = job
    with Image.open(checked(case.path('image'),row['sha256'])) as src:
        im=src.convert('RGB')
    parts,_=clean_surfaces(im,read_gzip(checked(BASE/row['predictions'],digest))['gap'])
    runs={s:edge_detail.predict(im,parts,row['sha256'],s) for s in settings}
    pose,ref=recorded(case)
    native=_native_reference(pose,ref,read(case.path('labels'))['faces'])
    _,pairs=measure(parts,ref,use_mask=True)
    results={}
    for method,after in runs.items():
        totals=Counter()
        for i,rid in pairs:
            if after[i] is parts[i]:
                continue
            before=pixels(parts[i],im.size)
            now=pixels(after[i],im.size)
            actual=native==rid
            for movement,domain in [('add',now&~before),('drop',before&~now)]:
                totals[movement+'_target']+=int((domain&actual).sum())
                totals[movement+'_other_plane']+=int((domain&~actual&(native!=0)).sum())
                totals[movement+'_background']+=int((domain&(native==0)).sum())
        results[method]=totals
    return dict(key=row['key'],sha256=row['sha256'],layout=row['layout'],methods=results)


def run(settings,out,workers):
    manifest=read(checked(BASE/'manifest.json',BASE_SHA))
    rows=[r for r in read(checked(BASE/'cases.json',manifest['artifacts']['cases.json'])) if r['cohort']=='development']
    caps={c.floor+'/'+c.id:c for cohort in groups()[0].values() for c in cohort}
    jobs=[(caps[r['key']],r,manifest['artifacts'][r['predictions']],settings) for r in rows]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(evaluate,jobs))
    summary={}
    for method in settings:
        tally=Counter()
        for row in results:
            tally.update(row['methods'][method])
        tally['net_correct_edits']=tally['add_target']+tally['drop_other_plane']+tally['drop_background']-tally['drop_target']-tally['add_other_plane']-tally['add_background']
        summary[method]=tally
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,dict(revision=revision(),source_sha256=BASE_SHA,settings={s:edge_detail.SETTINGS[s] for s in settings},summary=summary,frames=results))
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',nargs='+',required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--out',type=Path,required=True)
    opts=parser.parse_args()
    run(opts.settings,opts.out,opts.workers)
