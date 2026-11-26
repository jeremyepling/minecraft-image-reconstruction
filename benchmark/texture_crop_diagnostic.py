"""Reference-only crop experiment. Predicted block counts never receive exact dimensions."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_metrics import reference_planes
from benchmark.normal_round import groups
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import recorded
from benchmark.texture_count_round import FINDERS
from benchmark.mask_metrics import measure
from benchmark.wall_join_test import checked, read_gzip
from engine.geometry import UNIT,project
from engine.masked_count import count
from engine.plane_consensus import row_runs
from engine.surface_dimensions import dimensions
from engine.surface_model import canonical
from engine.surface_sample import sample_surface
from runtime.mode import image_only,mark_image_only


def recorded_region(ref,rid,target,native):
    coords=np.asarray(list(target['faces']))
    low,high=coords.min(0),coords.max(0)+1
    quad=canonical(project(ref['charts'][rid],low+UNIT*(high-low)))
    return dict(quad=quad,support=dict(size=list(native.shape[::-1]),runs=row_runs(native==rid)))


def evaluate(job):
    cap,row,pred_sha,finder=job
    source,method=FINDERS[finder]
    with Image.open(checked(cap.path('image'),row['sha256'])) as src:
        photo=src.convert('RGB')
    parts=read_gzip(checked(source/row['predictions'],pred_sha))[method]
    pose,ref=recorded(cap)
    faces=read(cap.path('labels'))['faces']
    native=_native_reference(pose,ref,faces)
    truth=reference_planes(ref,faces)
    surfaces=[]
    _,pairs=measure(parts,ref,use_mask=True)
    for index,rid in pairs:
        part=parts[index]
        target=truth[rid]
        material=sorted({f['block_id'] for f in target['faces'].values()})
        clean=recorded_region(ref,rid,target,native)
        guesses={}
        for mode,test in [('shipped',part),('recorded_corners',dict(quad=clean['quad'],support=part['support'])),
                          ('recorded_corners_and_visible_mask',clean)]:
            sample=sample_surface(photo,test)
            estimated=count(sample)
            guesses[mode]=dimensions(test,sample['quad'] if sample else None,photo.size,estimated['counts'])['blocks']
        surfaces.append(dict(reference=rid,proposal=part['id'],actual=target['size'],materials=material,guesses=guesses))
    return dict(key=row['key'],sha256=row['sha256'],layout=row['layout'],surfaces=surfaces)


@image_only()
def run(out,workers,finder):
    source,_=FINDERS[finder]
    manifest=read(source/'manifest.json')
    rows=read(checked(source/'cases.json',manifest['artifacts']['cases.json']))
    cases={c.floor+'/'+c.id:c for g in groups()[0].values() for c in g}
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        frames=list(pool.map(evaluate,[(cases[r['key']],r,manifest['artifacts'][r['predictions']],finder) for r in rows]))
    totals={}
    for group in ('all','pure_cobblestone','contains_cobblestone'):
        counts={}
        for frame in frames:
            for s in frame['surfaces']:
                mats=s['materials']
                if (group=='pure_cobblestone' and mats!=['minecraft:cobblestone'] or
                    group=='contains_cobblestone' and 'minecraft:cobblestone' not in mats):
                    continue
                for method,g in s['guesses'].items():
                    c=counts.setdefault(method,Counter())
                    c['surfaces']+=1
                    c['correct' if g==s['actual'] else 'unknown' if None in g else 'incorrect']+=1
        totals[group]=counts
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,dict(code=revision(),scope='Scoring-only recorded frontmost masks and plane corners; oracle chart uses its own vanishing directions, no predicted normal. Matched surfaces only.',
         finder=finder,source_manifest=digest(source/'manifest.json'),totals=totals,frames=frames))
    print(totals)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--finder',choices=FINDERS,default='neighbor')
    a=p.parse_args()
    run(a.out,a.workers,a.finder)
