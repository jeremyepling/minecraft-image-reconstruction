"""Apply registered texture depth to physical wall masks already refined by the frozen return rule."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from benchmark import framed_depth_eval as framed
from benchmark.facade_phase_evidence import REVIEW,common
from benchmark.texture_camera_eval import ROOT,checked,digest,read,save
from engine.facade_inset_phase import refine, supplemental
from engine.pillar_seams import native_mask

RETURNS=Path('../facade-inset-round/runs/returns')


@lru_cache(maxsize=1)
def returns():
    meta=read(RETURNS/'summary.json')
    assert meta['baseline']=='c589f1a' and meta['frames']==204
    frames=read(checked(RETURNS/'cases.json',meta['cases_sha256']))
    return {r['id']:r for r in frames}


def case(job):
    s,f,h,control,scoped=job
    im,parts,counts,cam=common.ordinary_inputs(s,f,h,control)
    prior=returns()[f['id']]
    assert prior['rgb_sha256']==f['files']['rgb.png']
    if prior['changed']:
        path=RETURNS/s/f['id']/'surfaces.json'
        parts=read(checked(path,prior['surfaces_sha256']))['planes']
    start=perf_counter()
    if scoped:
        result,events=supplemental(im,parts,cam,prior['events'])
    else:
        result,events=refine(im,parts,cam)
    took=perf_counter()-start
    row=dict(id=f['id'],split=s,rgb_sha256=prior['rgb_sha256'],seconds=took,events=events)
    if not events:
        return row
    if scoped:
        by_id={p['id']:p for p in result}
        for part in parts:
            host=part['id']
            children=[e['output'] for e in events if e['parent']==host]
            if not children:
                assert by_id[host]==part
                continue
            support=native_mask(by_id[host],im.size)
            for name in children:
                assert by_id[name]['parent_id']==host
                back=native_mask(by_id[name],im.size)
                assert not (support&back).any()
                support|=back
            assert np.array_equal(support,native_mask(part,im.size))
    labels=common.visible_map(s,f)
    base=common.surface_stats(parts,labels,im.size)
    out=common.surface_stats(result,labels,im.size)
    changes=dict(gained=sorted(set(out['matched'])-set(base['matched'])),lost=sorted(set(base['matched'])-set(out['matched'])),
                 paired_delta=out['paired_pixels']-base['paired_pixels'],unmatched_delta=out['unmatched_predictions']-base['unmatched_predictions'])
    geometry=common.geometry(s,f,control,im,parts,result,counts,cam)
    folder=Path('runs/scoped' if scoped else 'runs/after-returns')/s/f['id']
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'surfaces.json',dict(image_size=im.size,planes=result))
    if prior['changed']:
        geom=prior['geometry']
        assert (geom['correct'],geom['extras'])==(geometry['correct']-geometry['gained']+geometry['lost'],
                geometry['extras']-geometry['extra_added']+geometry['extra_removed'])
    return dict(row,baseline=base,after=out,surfaces_sha256=digest(folder/'surfaces.json'),difference=changes,geometry=geometry)


def run(args):
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(case,[(*job,args.scoped) for job in common.jobs([*framed.SIZES,'shelter'])]))
    elapsed=[r['seconds'] for r in rows]
    target='supplemental.json' if args.scoped else 'after-returns.json'
    archive=None
    if args.scoped:
        archive=REVIEW/'supplemental-surfaces.zip'
        with ZipFile(archive,'w',ZIP_DEFLATED) as z:
            for r in rows:
                if r['events']:
                    path=Path('runs/scoped')/r['split']/r['id']/'surfaces.json'
                    z.write(checked(path,r['surfaces_sha256']),r['id']+'/surfaces.json')
        with ZipFile(archive) as z:assert z.testzip() is None
    save(REVIEW/target,dict(returns_sha256=digest(RETURNS/'cases.json'),phase_file_sha256=digest(ROOT/'engine/facade_inset_phase.py'),
        output_archive_sha256=digest(archive) if archive else None,median_seconds=np.median(elapsed),
        p95_seconds=np.quantile(elapsed,.95),frames=rows))
    print([(r['id'],r['difference']) for r in rows if r['events']])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--scoped',action='store_true')
    run(p.parse_args())
