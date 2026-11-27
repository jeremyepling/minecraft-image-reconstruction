"""Replace counts.json's numeric width and height on eligible source-predicted wall quads."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import shutil
from unittest.mock import patch

import cv2
import numpy as np

from benchmark import expanded_scores, floating_cloud_replay as paths, relative_face_round as assembly
from benchmark import shared_corner_rows as shared, native_surface_counter as counter
from benchmark import native_corner_registration as registration, garden_anchor_extension as atlas
from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.expanded_wall_rulers import NATIVE, OUT, cases, jacobian, reference_chart
from benchmark.relative_source_charts import extract, wall_quad
from engine.geometry import project, transform


def patch_counts(case, folder):
    frame = Path(case['frame'])
    name=case['id']
    original=NATIVE/'sky'/name/'source-handoff'
    folder.mkdir(parents=True,exist_ok=True)
    for file in ('build-manifest.json','surfaces.json'):
        shutil.copyfile(original/file,folder/file)
    parts={p['id']:p for p in read(original/'surfaces.json')['planes']}
    packet=read(original/'counts.json')
    labels=read(frame/'labels.json')
    assert digest(frame/'labels.json')==read(frame/'frame.json')['files']['labels.json']
    blocks={b['id']:b.get('geometry_kind','full_cube') for b in labels['blocks']}
    camera=np.asarray(labels['camera']['world_to_pixel_homogeneous'])
    faces=defaultdict(list)
    for face in labels['faces']:
        faces[face['surface_id']].append(face)
    rows=[]
    for row in packet['surfaces']:
        sid=row['id']
        quad=wall_quad(row,parts[sid])
        if quad is None:
            continue
        truth=reference_chart(faces[int(sid.removeprefix('plane-'))],blocks,camera)
        if truth is None:
            continue
        # Width and height are fractional counts across the predicted rectifying
        # chart. Calibrate their local pitch at its center; leave perspective,
        # placement of that chart and its sub-block phase entirely untouched.
        known,error=truth
        H=transform(quad)
        center=np.array([.5,.5])
        ref_center=project(np.linalg.inv(known),project(H,center))
        src=jacobian(H,center)
        ref=jacobian(known,ref_center)
        counts=(np.linalg.norm(src,axis=0)/np.linalg.norm(ref,axis=0)).tolist()
        if not np.all(np.isfinite(counts)) or min(counts)<=0:
            raise ValueError(f'Invalid width/height {sid}: {counts}')
        rows.append(dict(id=sid,before=row.get('counts'),after=counts,max_corner_error=error))
        row['counts']=counts
    save(folder/'counts.json',packet)
    save(OUT/'counts/evidence'/(name+'.json'),dict(id=name,source_sha256=case['rgb_sha256'],rows=rows,
        before_sha256=digest(original/'counts.json'),after_sha256=digest(folder/'counts.json')))


def run(case):
    cv2.setNumThreads(1)
    name=case['id']
    root=OUT/'counts'
    folder=root/'sky'/name/'source-handoff'
    patch_counts(case,folder)
    rows=root/'shared'
    native=root/'counter'
    stage=root/'atlas'
    def dense(c,h,image,provenance):
        out=rows/'dense'/name
        extract((case['variants']['sky'],name,'sky',h,out))
        packet=out/'dense-charts.json'
        return read(packet)|provenance|dict(baseline_chart_sha256=digest(packet))
    with ExitStack() as stack:
        def bind(mod,**kwargs):
            stack.enter_context(patch.multiple(mod,**kwargs))
        bind(paths,RUN=root)
        bind(shared,OUTPUT=rows,dense_packet=dense)
        bind(counter,OUTPUT=native,BASE=rows/'sky')
        bind(registration,OUTPUT=native,BASE=rows/'sky')
        bind(atlas,OUTPUT=stage,NATIVE_COUNTER=native)
        mode=(stage/'connected','charts.json',False,1,2.5,.5,False)
        bind(assembly,ASSEMBLY=root/'assembly',MODES=assembly.MODES|{'connected-atlas':mode})
        if case['dataset']=='visible-builds-v2':
            audit=read(ROOT/'training/framing-audit-v2.json')
            proof=next(x for x in audit['frames'] if x['id']==name)
            assert proof['accepted'] and proof['rgb_sha256']==case['rgb_sha256']
            def read_v2(path):
                result=read(path)
                return result|dict(framing=proof) if path==Path(case['frame'])/'frame.json' else result
            bind(expanded_scores,read=read_v2)
        shared.predict(case)
        counter.predict(case)
        registration.predict(case)
        atlas.all_walls((case,read(atlas.baseline_file(name))))
        assembly.evaluate((case,'sky','connected-atlas'))
    print('EXPORTED',name,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--workers',type=int,default=4)
    cli.add_argument('--case',nargs='+')
    args=cli.parse_args()
    selected=[c for c in cases() if not args.case or c['id'] in args.case]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,selected))
