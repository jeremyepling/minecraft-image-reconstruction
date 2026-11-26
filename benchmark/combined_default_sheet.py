"""Follow one measured surface per exterior building through the ordinary material run."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_labels import exposed_faces
from benchmark.camera_metrics import registration, truth_camera
from benchmark.camera_round import digest, read, save
from benchmark.normal_round import groups
from benchmark.surface_pipeline_sheet import _picture, _straight_photo, PAPER, MUTED
from benchmark.surface_sheet import text
from benchmark.texture_count_round import FINDERS
from benchmark.wall_join_test import checked, read_gzip
from engine import surface_perimeter
from engine.surface_model import _count_regions, _material_assets
from engine.surface_tiles import tile_sample


def projected_build(photo, record, pose):
    cubes = set(map(tuple, record['registered']))
    correct = set(map(tuple, record['correct']))
    canvas = photo.convert('L').convert('RGB')
    draw = ImageDraw.Draw(canvas)
    projection = truth_camera(pose)
    polygons = []
    for point, normal, _, corners in exposed_faces(cubes, pose['eye_world']):
        xyz = corners @ projection[:, :3].T + projection[:, 3]
        if xyz[:, 2].min() <= 0:
            continue
        xy = xyz[:, :2] / xyz[:, 2:]
        color = np.array([88, 193, 153] if point in correct else [228, 96, 65])
        shade = [0.82, 1., .68][int(np.argmax(abs(normal)))]
        polygons.append((xyz[:, 2].mean(), list(map(tuple, xy)), tuple((color*shade).astype(int))))
    for _, polygon, color in sorted(polygons, key=lambda p: p[0], reverse=True):
        draw.polygon(polygon, fill=color, outline='#243c3e')
    return canvas


def material_overlay(photo, audit, pose, visible):
    transform = registration(np.asarray(audit['projection']), pose)
    canvas = photo.convert('L').convert('RGB')
    draw = ImageDraw.Draw(canvas)
    correct = accepted = 0
    for reading in audit['choices']:
        if reading['accepted']:
            world = tuple(np.floor(transform[:3,:3] @ (np.array(reading['cell']) + .5) + transform[:3,3]).astype(int))
            match = visible.get(world) == reading['predicted_id']
            draw.polygon(list(map(tuple, reading['quad'])), fill='#32aaa1' if match else '#dc6041', outline='#273e41')
            accepted += 1
            correct += int(match)
    return canvas, f'{correct} of {accepted} reads at correct XYZ + ID'


def run(geometry, out):
    cohorts, _, _ = groups()
    captures = {f'{c.floor}/{c.id}': c for cohort in cohorts.values() for c in cohort}
    root, method = FINDERS['photo']
    manifest = read(root/'manifest.json')
    cases = read(checked(root/'cases.json',manifest['artifacts']['cases.json']))
    selected = {}
    for row in cases:
        # One reproducible view per building; prefer the low-left masonry view.
        key = row['key']
        priority = ('masonry' not in key, 'low-left' not in key, 'smooth_stone/' not in key, key)
        if row['layout'] not in selected or priority < selected[row['layout']][0]:
            selected[row['layout']] = (priority, row)
    counter = 'texture-guided-repeat'
    atlas, matcher, _, _ = _material_assets('26.3', counter, 'shape', None)
    if matcher is None:
        raise ValueError('The calibrated Shape matcher is required for the sheet')
    labels = ['Source screenshot','Detected surface','Surface with context','Masked rectification',
              'Measured block grid','Sampled face and ID','Exported coordinates','Projected material trial']
    col, stride, left, header = 292, 382, 235, 162
    sheet = Image.new('RGB',(left+(col+12)*len(selected)+14, header+stride*len(labels)+32),PAPER)
    draw = ImageDraw.Draw(sheet)
    text(draw,18,14,'Visible reconstruction from one image',36,True)
    text(draw,18,66,'Cyan outlines the selected surface. Exported coordinates: green correct, orange extra. Trial material reads: teal correct position and ID, orange incorrect, gray unpainted.',18)
    for i, label in enumerate(labels):
        y = header+i*stride
        draw.line((12,y,sheet.width-12,y),fill='#ccdbd2',width=2)
        text(draw,16,y+26,label,18,True)
    examples=[]
    for j,(layout,(_,row)) in enumerate(selected.items()):
        key=row['key']
        cap=captures[key]
        parts=read_gzip(checked(root/row['predictions'],manifest['artifacts'][row['predictions']]))[method]
        with Image.open(checked(cap.path('image'),row['sha256'])) as source:
            photo=source.convert('RGB')
        with TemporaryDirectory() as tmp:
            counts=_count_regions(photo,parts,'26.3',counter,atlas,Path(tmp),matcher)
            # Illustrate an actual material read when available; otherwise the largest charted surface.
            idx=max(range(len(parts)), key=lambda i: (
                len(counts['surfaces'][i].get('visible_cells',[])), 'sample' in counts['surfaces'][i],
                sum(b-a for _,a,b in parts[i]['support']['runs'])))
            part, result=parts[idx],counts['surfaces'][idx]
            outline=photo.copy()
            surface_perimeter.draw_outline(outline,surface_perimeter.with_outline(part),'#35ffff',4)
            crop=surface_perimeter.context_crop(photo,part,8)
            cells=result.get('visible_cells',[])
            named=next((c for c in cells if c.get('material',{}).get('block_id')),None)
            clean=grid=None
            if 'sample' in result:
                with Image.open(Path(tmp)/result['sample']) as sample:
                    rgba=np.asarray(sample.convert('RGBA'))
                    clean,grid,_=_straight_photo(dict(rgb=rgba[:,:,:3],mask=rgba[:,:,3]>0),result,
                                                 named['cell'] if named else None)
        record=read(geometry/'frames'/key/'registered.json')
        material=named['material']['block_id'] if named else None
        tile=Image.fromarray(tile_sample(photo,named['quad'])['rgb']) if named else None
        texture=named['material'].get('texture') if named else None
        if texture in atlas.labels:
            pair=Image.new('RGB',(264,132),PAPER)
            _picture(pair,tile,0,0,128,132,nearest=True)
            _picture(pair,Image.fromarray(atlas.tiles[atlas.labels.index(texture)]),136,0,128,132,nearest=True)
            tile=pair
        pose=read(cap.path('pose'))
        src=projected_build(photo,record,pose) if record['registered'] else None
        trial=(None,'No placed face to name')
        if record['registered']:
            audit=read(geometry/'projected-material/frames'/key/'predicted.json')
            visible={tuple(r[:3]):r[3] for r in record['visible_states']}
            trial=material_overlay(photo,audit,pose,visible)
        dims=result['dimensions']['blocks']
        desc=' × '.join('?' if n is None else str(n) for n in dims)
        panels=[(photo,cap.id), (outline,f"{len(parts)} surface proposals"),
                (crop[0] if crop else None,f"Surface {part['id']}"),(clean,'Visible pixels used by the counter'),
                (grid,f'{desc} blocks across and down'),(tile,material or 'No supported material ID'),
                (src,f"{len(record['correct'])} correct of {len(record['registered'])} placed" if src else 'No supported metric camera'),trial]
        x=left+j*(col+12)
        text(draw,x,112,layout.replace('_',' ').title(),22,True)
        for i,(image,caption) in enumerate(panels):
            y=header+i*stride+15
            if image is not None:
                _picture(sheet,image,x,y,col,stride-80,nearest=i==5)
            else:
                draw.rectangle((x,y,x+col-1,y+stride-81),fill='#e4e8e3')
            if i==0:
                caption=caption.removeprefix('dev-').removeprefix('holdout-')
            text(draw,x,y+stride-63,caption,15,fill=MUTED)
        examples.append(dict(key=key,source_sha256=digest(cap.path('image')),surface=part['id'],
                             material=material,dimensions=dims,placed=len(record['registered']),correct=len(record['correct'])))
    out.mkdir(exist_ok=True,parents=True)
    sheet.save(out/'comparison.jpg',quality=90,optimize=True)
    save(out/'examples.json',examples)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--geometry',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    args=cli.parse_args()
    run(args.geometry,args.out)
