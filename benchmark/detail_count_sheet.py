"""Retrospective close-ups of two recovered surfaces and one newly incorrect Portico count."""
import argparse
from pathlib import Path

from PIL import Image

from benchmark.camera_round import read
from benchmark.count_metrics import reference_planes
from benchmark.detail_sheet import render
from benchmark.mask_metrics import measure
from benchmark.normal_round import groups
from benchmark.strip_review_sheet import native_mask
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import recorded
from benchmark.wall_join_test import checked, read_gzip
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface


def guess(photo, part):
    sample = sample_surface(photo,part)
    result = count(sample)
    return dimensions(part,sample['quad'] if sample else None,photo.size,result['counts'])['blocks']


def render_count_review(out):
    ids = [('Shopfront','light_gray_concrete/dev-shopfront-plaster-low-left',3),
           ('Workshop','light_gray_concrete/holdout-workshop-masonry-high-right',19),
           ('Portico','smooth_stone/holdout-portico-plaster-low-left',None)]
    captures = {c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
    rows = {r['key']:r for r in read(out/'cases.json')}
    loose = 'edge_detail:native4-count-guard'
    strict = 'edge_detail:native4-stable-counts'
    examples = []
    for name,key,selected in ids:
        row,case = rows[key],captures[key]
        with Image.open(checked(case.path('image'),row['sha256'])) as src:
            photo = src.convert('RGB')
        pose,ref = recorded(case)
        faces = read(case.path('labels'))['faces']
        native = _native_reference(pose,ref,faces)
        truth = reference_planes(ref,faces)
        parts = read_gzip(out/row['predictions'])
        lookup = {p['id']:p for p in parts['current']}
        _,pairs = measure(parts[loose],ref,use_mask=True)
        for i,rid in sorted(pairs,key=lambda pair:pair[1]):
            part = parts[loose][i]
            if selected is not None:
                if rid!=selected:
                    continue
            else:
                prior,later = guess(photo,lookup[part['id']]),guess(photo,part)
                if None not in prior or None in later or later==truth[rid]['size']:
                    continue
            source = native_mask(part,photo.size)
            l,t,r,b = Image.fromarray(source | (native==rid)).getbbox()
            box = [max(0,l-8),max(0,t-8),min(photo.width,r+8),min(photo.height,b+8)]
            examples.append(dict(id=name,frame=key,reference=rid,proposal=part['id'],sha256=row['sha256'],
                                 detail_xyxy=box,scene_xyxy=Image.fromarray(native>0).getbbox(),
                                 reference_size_blocks=truth[rid]['size']))
            break
        else:
            raise ValueError('Review case unavailable: '+key)
    render(out,examples=examples,titles={'current':'Current Finder',loose:'Photo edges',strict:'Stricter count rule'})


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    render_count_review(cli.parse_args().out)
