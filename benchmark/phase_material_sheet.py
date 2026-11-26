"""One visible surface per building through the two grid and material paths."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, read, save
from benchmark.count_materials import true_faces
from benchmark.count_metrics import reference_planes
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.known_spacing_atlas import predict
from benchmark.normal_round import groups
from benchmark.surface_pipeline_sheet import _picture, _straight_photo, PAPER, MUTED
from benchmark.surface_reference import recorded
from benchmark.surface_sheet import text
from benchmark.wall_join_test import checked, read_gzip
from engine.masked_count import count
from engine.strip_neighbor import clean_surfaces
from engine.surface_perimeter import context_crop, draw_outline, with_outline
from engine.surface_sample import sample_surface
from engine.surface_tiles import tile_sample


def select(frames):
    previous = read(ROOT/'benchmark/reviews/block-seam-01/sheet-views.json')
    selected = {r['layout']:(0,r['frame'],r['proposal'],r['reference'],None,None) for r in previous}
    for f in frames:
        methods = f['methods']['shape']['faces']
        before = {tuple(p['reference']):p for p in methods['ordinary'] if p.get('aligned')}
        for p in methods['restricted']:
            if not p.get('aligned'):
                continue
            key = tuple(p['reference'])
            rank = 1 if key in before else 2
            # Show the less frequent recovered material when a building supplies both.
            if key not in before and p['expected']=='minecraft:spruce_planks':
                rank = 3
            if rank>selected[f['layout']][0]:
                selected[f['layout']] = (rank,f['key'],p['proposal'],key[0],p,before.get(key))
    return selected


def short(value):
    return value.removeprefix('minecraft:').replace('_',' ') if value else 'Unresolved'


def tile(image,quad):
    return Image.fromarray(tile_sample(image,quad)['rgb'])


def render(cases,out):
    selected = select(read(cases))
    cohorts,_,_ = groups()
    caps = {c.floor+'/'+c.id:c for group in cohorts.values() for c in group}
    source = read(checked(BASE/'manifest.json',BASE_SHA))
    rows = {r['key']:r for r in read(checked(BASE/'cases.json',source['artifacts']['cases.json']))}
    labels = ('Screenshot','Visible surface','Rectified crop','Ordinary grid','Atlas offset',
              'Ordinary material','Atlas-offset material','Recorded face')
    left,col,head,h = 167,202,118,223
    page = Image.new('RGB',(left+len(selected)*col+12,head+len(labels)*h),PAPER)
    draw = ImageDraw.Draw(page)
    text(draw,16,10,'Does a located seam recover the correct block ID?',32,True)
    text(draw,16,54,'Cyan outlines the detected surface. Gold marks measured seams. Blue highlights the cell used in the material comparison.',17)
    for i,label in enumerate(labels):
        y = head+i*h
        draw.line((12,y,page.width-12,y),fill='#b4c4bf',width=2)
        text(draw,12,y+10,label,16,True)
    index = []
    for j,(layout,(_,key,pid,rid,after,before)) in enumerate(sorted(selected.items())):
        x = left+j*col
        text(draw,x,85,layout.title(),21,True)
        c,r = caps[key],rows[key]
        with Image.open(checked(c.path('image'),r['sha256'])) as src:
            image = src.convert('RGB')
        upstream = read_gzip(checked(BASE/r['predictions'],source['artifacts'][r['predictions']]))['gap']
        part = next(p for p in clean_surfaces(image,upstream)[0] if p['id']==pid)
        marked = image.copy()
        draw_outline(marked,with_outline(part),width=3)
        _,box,_ = context_crop(image,part,margin_px=16)
        sample = sample_surface(image,part)
        base = count(sample)
        new = predict(sample,base)
        rect,grid,_ = _straight_photo(sample,base,before['cell'] if before else None)
        _,updated,_ = _straight_photo(sample,new,after['cell'] if after else None)
        ref = recorded(c)[1]
        planes = reference_planes(ref,read(c.path('labels'))['faces'])
        valid = true_faces(planes,image.size)
        if after:
            _,u,v = after['reference']
            actual = valid[rid,(u,v)]
        else:
            actual = next((f for (p,_),f in valid.items() if p==rid),None)
        def panel_face(row):
            return (tile(image,row['quad']),short(row['guess'])) if row else (None,'No recovered face here')
        def phase_label(result):
            return f"{sum(t is not None for t in result['phase'])} of 2 offsets located"
        panels = [(image,' '.join(key.split('-')[-3:])),(marked.crop(box),pid),(rect,'Supported pixels'),
                  (grid,phase_label(base)),(updated,phase_label(new)),panel_face(before),panel_face(after),
                  (tile(image,actual['image_corners']),short(actual['block_id'])) if actual else (None,'No eligible reference face')]
        for i,(photo,caption) in enumerate(panels):
            y = head+i*h+9
            if photo is not None:
                _picture(page,photo,x,y,col-8,174,nearest=i>=5)
            else:
                draw.rectangle((x,y,x+col-9,y+173),fill='#e6eae5')
            text(draw,x,y+180,caption,14,fill=MUTED)
        index.append(dict(layout=layout,key=key,proposal=pid,reference=rid,cell=None if after is None else after['reference'][1:]))
    out.mkdir(parents=True,exist_ok=True)
    page.save(out/'comparison.jpg',quality=90)
    save(out/'sheet-views.json',index)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('cases',type=Path)
    cli.add_argument('out',type=Path)
    args = cli.parse_args()
    render(args.cases,args.out)
