"""Scoring-only comparison of recorded chart corners versus the ordinary Finder chart."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import read,save
from benchmark.count_metrics import chart_spans,reference_planes
from benchmark.normal_round import groups
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import recorded
from benchmark.texture_count_round import SOURCE
from benchmark.wall_join_test import checked,read_gzip
from benchmark.portico_repeat import axis_peaks,choose,predict
from engine.geometry import UNIT,project
from engine.masked_count import count
from engine.plane_consensus import row_runs
from engine.surface_dimensions import dimensions
from engine.surface_model import canonical
from engine.surface_sample import sample_surface


def metrics(photo, part, chart):
    sample=sample_surface(photo,part)
    baseline=count(sample)
    prediction=predict(sample,baseline)
    spans=chart_spans(chart,sample['quad'])
    wh=np.asarray(sample['mask'].shape[::-1])
    return dict(quad=sample['quad'],chart_pixels=wh.tolist(),valid_pixels=int(sample['mask'].sum()),
                valid_fraction=float(sample['mask'].mean()),chart_span_in_recorded_blocks=spans,
                expected_mean_pitch_px=(wh/np.asarray(spans)).tolist(),
                baseline=baseline,repeat=prediction,
                baseline_dimensions=dimensions(part,sample['quad'],photo.size,baseline['counts'])['blocks'],
                repeat_dimensions=dimensions(part,sample['quad'],photo.size,prediction['counts'])['blocks'],
                strongest_patch_translations=[choose(axis_peaks(sample,k)) for k in (0,1)])


def run(out):
    requests={'smooth_stone/holdout-portico-plaster-low-left':[20,13,8],
              'smooth_stone/dev-shopfront-plaster-low-left':[11]}
    by_key={c.floor+'/'+c.id:c for g in groups()[0].values() for c in g}
    manifest=read(SOURCE/'manifest.json')
    rows={r['key']:r for r in read(checked(SOURCE/'cases.json',manifest['artifacts']['cases.json']))}
    output=[]
    for key,refs in requests.items():
        cap,row=by_key[key],rows[key]
        with Image.open(checked(cap.path('image'),row['sha256'])) as src:
            photo=src.convert('RGB')
        parts=read_gzip(checked(SOURCE/row['predictions'],manifest['artifacts'][row['predictions']]))['current']
        indexed={p['id']:p for p in parts}
        pose,ref=recorded(cap)
        faces=read(cap.path('labels'))['faces']
        planes=reference_planes(ref,faces)
        native=_native_reference(pose,ref,faces)
        for rid in refs:
            pid=next(p['proposal'] for p in row['paired_surfaces'] if p['reference']==rid)
            plane=planes[rid]
            chart=ref['charts'][rid]
            xy=np.asarray(list(plane['faces']))
            lo,hi=xy.min(0),xy.max(0)+1
            quad=canonical(project(chart,lo+UNIT*(hi-lo)))
            oracle=dict(quad=quad,support=dict(runs=row_runs(native==rid),size=list(photo.size)))
            output.append(dict(key=key,sha256=row['sha256'],reference=rid,proposal=pid,actual=plane['size'],
                               block_ids=sorted({v['block_id'] for v in plane['faces'].values()}),
                               predicted_chart=metrics(photo,indexed[pid],chart),
                               recorded_chart_diagnostic=metrics(photo,oracle,chart)))
    save(out,dict(scope='Recorded chart, reference visibility mask and face IDs are evaluation-only. '
                         'The predicted-chart runs use image-only Finder and model evidence.',examples=output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
