"""Texture controls using recorded surface/face corners for diagnostics only."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, read, save
from benchmark.count_metrics import reference_planes, chart_spans
from benchmark.normal_round import groups
from benchmark.portico_atlas import atlas, predict
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import recorded
from benchmark.mask_metrics import measure
from benchmark.wall_join_test import read_gzip, checked
from engine.surface_dimensions import dimensions
from engine import atlas_count, masked_count
from engine.geometry import UNIT, project
from engine.plane_consensus import row_runs
from engine.surface_model import canonical
from engine.surface_sample import sample_surface


def tile_control():
    source=atlas()
    tile=source.tiles[source.labels.index('cobblestone')]
    image=np.roll(np.tile(np.rot90(tile),(4,6,1)),(3,6),axis=(0,1))
    rgb=(np.repeat(np.repeat(image,2,axis=0),2,axis=1)*.78).astype(np.uint8)
    mask=np.ones(rgb.shape[:2],bool)
    mask[41:72,80:107]=False
    sample=dict(rgb=rgb,mask=mask,normal_aspect=1.5)
    ordinary=masked_count.count(sample)
    # Force the fallback to be evaluated for this positive control, retaining RGB-derived pitches.
    candidate=predict(sample,dict(ordinary,counts=[None,None],phase=[None,None]))
    evidence=candidate['evidence']['atlas_check']
    if evidence['atlas_texture']!='cobblestone' or not evidence['accepted'] or list(np.rint(candidate['counts']))!=[6,4]:
        raise ValueError('Whole-tile positive control failed')
    return dict(constructed_from='Vanilla 26.3 cobblestone tile, quarter turn, cyclic x/y shift, 78% brightness, 2x nearest magnification, rectangular hole',
                ordinary=ordinary['counts'], atlas=candidate['counts'], atlas_check=evidence)


def controls(out):
    cases={c.floor+'/'+c.id:c for g in groups()[0].values() for c in g}
    selected={'smooth_stone/holdout-portico-plaster-low-left':[8,13,20],
              'light_gray_concrete/holdout-workshop-plaster-low-left':[15]}
    rows=[]
    archive=ROOT/'benchmark/reviews/detail-01'
    frame_rows={r['key']:r for r in read(archive/'cases.json')}
    source_manifest=read(archive/'manifest.json')
    for key,ids in selected.items():
        capture=cases[key]
        with Image.open(capture.path('image')) as src:
            photo=src.convert('RGB')
        pose,ref=recorded(capture)
        faces=read(capture.path('labels'))['faces']
        targets=reference_planes(ref,faces)
        rel=frame_rows[key]['predictions']
        predictions=read_gzip(checked(archive/rel,source_manifest['artifacts'][rel]))['current']
        _,pairs=measure(predictions,ref,use_mask=True)
        by_reference={r:predictions[i] for i,r in pairs}
        native=_native_reference(pose,ref,faces)
        for rid in ids:
            surface=targets[rid]
            cells=np.array(list(surface['faces']))
            lo,hi=cells.min(axis=0),cells.max(axis=0)+1
            quad=canonical(project(ref['charts'][rid],lo+UNIT*(hi-lo)))
            support=dict(size=list(photo.size),runs=row_runs(native==rid))
            sample=sample_surface(photo,dict(quad=quad,support=support))
            base=masked_count.count(sample)
            candidate=predict(sample,dict(base,counts=[None,None],phase=[None,None]))
            votes=Counter()
            truth_votes=0
            for f in surface['faces'].values():
                face=sample_surface(photo,dict(quad=canonical(f['image_corners']),support=support))
                if face is not None and face['mask'].sum()>=32:
                    choice=atlas_count.rank_tile(face,atlas())
                    votes[str(choice['block_id'])]+=1
                    truth_votes+=int(choice['block_id']==f['block_id'])
            current=by_reference[rid]
            original=sample_surface(photo,current)
            checks={}
            for chart_name,corners in [('finder',original['quad']),('reference',quad)]:
                for mask_name,mask in [('finder',current['support']),('reference',support)]:
                    # Keep the predicted normal unchanged in this four-way comparison.
                    part=dict(current,quad=corners,support=mask)
                    view=sample_surface(photo,part)
                    estimate=masked_count.count(view)
                    checks[chart_name+'_corners_'+mask_name+'_mask']=dict(
                        blocks=dimensions(part,corners,photo.size,estimate['counts'])['blocks'],
                        measured_chart_spans=estimate['counts'],true_chart_spans=chart_spans(ref['charts'][rid],corners),
                        valid_chart_pixels=int(view['mask'].sum()),chart_wh=list(view['mask'].shape[::-1]))
            rows.append(dict(four_way_corner_mask_check=checks,frame=key,reference=rid,actual=surface['size'],real_corner_chart=quad,
                       chart_wh=list(sample['mask'].shape[::-1]) if sample else None,
                       material=dict(Counter(f['block_id'] for f in surface['faces'].values())),
                       crop_only_estimate=base['counts'],atlas_check=candidate['evidence']['atlas_check'],
                       known_face_matcher=dict(ranked=sum(votes.values()),right=truth_votes,top1=dict(votes))))
    save(out,dict(atlas=atlas().provenance,synthetic=tile_control(),
        diagnostic='Uses recorded plane masks and corners; separately ranks individual true face crops with at least 32 supported pixels. Does not feed ordinary predictions.',
        examples=rows))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',required=True,type=Path)
    controls(cli.parse_args().out)
