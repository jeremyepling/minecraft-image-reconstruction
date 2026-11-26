"""Measure exact roof-to-post displacements in visible, decoded canopy geometry."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_metrics import registration
from benchmark.canopy_fixture import OUT as FRESH
from benchmark.canopy_fixture_metrics import PROBES, gap_width, raster
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.texture_camera_eval import ROOT, STABLE, checked, digest, read, save
from benchmark.texture_camera_sheet import font
from benchmark.visible_depth_transfer import OUT as OLD, rgb_path, structure


def shift(cells, delta):
    dx, dy, dz = delta
    return {(x+dx, y+dy, z+dz) for x,y,z in cells}


def score(cells, visible, spec):
    left, right = spec['left']-1, spec['right']
    z, bottom, top = spec['front_z'], spec['bottom'], spec['roof_y']
    courses = [(left,y,z) for y in range(bottom,top)
               if (left,y,z) in visible and (right,y,z) in visible]
    roof = {p for p in visible if p[1]==top}
    # Every nonzero triple score must include a post pair with this separation.
    offsets = {(lx-x,ly-y,lz-pz) for lx,ly,lz in courses for x,y,pz in cells
               if (x+right-left,y,pz) in cells}
    best = None
    offset = None
    aligned = set()
    for delta in sorted(offsets):
        placed = shift(cells,delta)
        posts = sum(p in placed and (right,p[1],z) in placed for p in courses)
        roofs = len(placed & roof)
        # Use the two posts as a shared origin; among equally good post fits give
        # the roof the most favorable fit. No individual part can move again.
        rank = (posts, roofs, len(placed & visible))
        if best is None or rank > best:
            best,offset,aligned = rank,delta,placed
    # No supported pair makes a width measurement and a shared roof anchor unavailable.
    pairs = best[0] if best else 0
    return dict(offset=offset, target_post_courses=len(courses), matching_post_courses=pairs,
                visible_roof_cells=len(roof), anchored_roof_cells=len(aligned & roof),
                target_triples=len(courses)*len(roof), matching_triples=pairs*len(aligned & roof),
                target_gap_width=right-left-1, observed_gap_width=gap_width(aligned,z,bottom,top) if pairs else None)


def controls():
    spec=dict(left=1,right=5,front_z=0,bottom=0,roof_y=4)
    pillars={(x,y,0) for x in (0,5) for y in range(4)}
    roof={(x,4,0) for x in range(6)}
    truth=pillars | roof
    examples=[('Reference', truth),('Whole build moved', shift(truth,(2,1,0))),
              ('Roof raised one block', pillars | shift(roof,(0,1,0)))]
    rows=[score(cells,truth,spec) for _,cells in examples]
    assert [r['matching_triples'] for r in rows]==[24,24,0]
    assert score(set(),truth,spec)['observed_gap_width'] is None
    for (_,cells),row in zip(examples,rows,strict=True):
        moved=score(shift(cells,(17,-300,41)),truth,spec)
        assert {k:v for k,v in moved.items() if k!='offset'}=={k:v for k,v in row.items() if k!='offset'}
    return examples,rows


def draw(examples,rows,path):
    canvas=Image.new('RGB',(1100,455),'#14232C')
    draw=ImageDraw.Draw(canvas)
    draw.text((24,12),'Does the roof keep its position relative to the posts?',font=font(28),fill='white')
    draw.text((24,64),'A relationship joins one roof block to two post blocks at the same height.',font=font(19),fill='#CBDBE0')
    for n,((name,cells),row) in enumerate(zip(examples,rows,strict=True)):
        x0=25+360*n
        draw.text((x0,115),name,font=font(21),fill='white')
        for x in range(10):
            for y in range(7):
                a,b=x0+28*x,341-28*y
                color='#45B7AD' if (x,y,0) in cells else '#1B2E39'
                draw.rectangle((a,b,a+26,b+26),fill=color)
        draw.text((x0,391),f"{row['matching_triples']} / 24 relationships",font=font(20),fill='white')
    canvas.save(path)


def verify_pack(frame,folder,pose):
    key=frame['id']
    row=next(r for r in read(folder/'cases.json') if r['id']==key)
    assert row['rgb_sha256']==frame['files']['rgb.png']
    pack=checked(folder/key/'reconstruction.zip',row['pack_sha256'])
    states=decoded_pack(pack)
    cells=registered(states,registration(np.asarray(frame['camera']['P']),pose))
    saved=read(folder/key/'cells.json')
    assert cells==set(map(tuple,saved['world'])) and set(states)==set(map(tuple,saved['local']))
    return cells,row['pack_sha256']


def run(out):
    specs=read(PROBES)
    release=STABLE/'runs/canopy-placement/release'
    cases=[]
    for cohort,inputs,old in [('old',OLD,OLD/'results/production'),('fresh',FRESH,FRESH/'results/baseline')]:
        for f in read(inputs/'screen.json')['selected']:
            key=f['id']
            if key not in specs:
                continue
            png=rgb_path(f)
            pose=read(checked(png.with_suffix('.json'),f['files']['camera.json']))
            source=read(checked(png.with_name(f['source_id']+'-blocks.json'),f['files']['world-readback.json']))
            faces,_=raster(pose,structure(source,64))
            visible={tuple(face['xyz_world']) for face in faces}
            row=dict(id=key,cohort=cohort,rgb_sha256=f['files']['rgb.png'],world_sha256=f['files']['world-readback.json'])
            for name,folder in [('previous',old),('current',release/cohort)]:
                cells,sha=verify_pack(f,folder,pose)
                result=score(cells,visible,specs[key])
                delta=(42,-17,31)
                moved=score(shift(cells,delta),visible,specs[key])
                expected=dict(result)
                if result['offset'] is not None:
                    expected['offset']=tuple(v-d for v,d in zip(result['offset'],delta,strict=True))
                assert moved==expected, (key,name)
                row[name]=dict(result,pack_sha256=sha)
            cases.append(row)
    assert len(cases)==10 and len({r['id'] for r in cases})==10
    examples,control=controls()
    out.mkdir(parents=True,exist_ok=True)
    save(out/'cases.json',dict(method='shared-translation-roof-post-triples',fixture_sha256=digest(PROBES),
                               translated_replay=[42,-17,31],cases=cases,controls=control))
    draw(examples,control,out/'control.png')
    for cohort in ('old','fresh'):
        for name in ('previous','current'):
            rows=[r[name] for r in cases if r['cohort']==cohort]
            print(cohort,name,{k:sum(v[k] for v in rows) for k in ('target_triples','matching_triples','visible_roof_cells','anchored_roof_cells','matching_post_courses','target_post_courses')},
                  [v['observed_gap_width'] for v in rows])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'benchmark/reviews/relative-spacing-01')
    run(p.parse_args().out)
