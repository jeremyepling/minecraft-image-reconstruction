"""Show the photographed structures and horizontally aligned footprints of both exports."""
import argparse
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from benchmark.framed_depth_eval import (DEPTH, NATIVE_INPUTS, NEW_INPUTS, TRAINING_DATA,
    cases, members, native, references, transfer)
from benchmark.relative_geometry import best_translation
from benchmark.texture_camera_eval import checked, read
from benchmark.texture_camera_sheet import font

VIEWS = {
    'shape-benchmark-26.3--train-canopy-06-left': ('Canopy 06', 'Left view'),
    'inverse-voxels-26.3--confirm--test-colonnade-01-oblique-left': ('Colonnade 01', 'Oblique left'),
}


def aligned(cells,truth):
    delta,_=best_translation(cells,truth)
    return {tuple(a+b for a,b in zip(p,delta)) for p in cells}


def draw_footprint(image, cells, truth, bounds, x, y):
    pen=ImageDraw.Draw(image)
    x0,z0,x1,z1=bounds
    scale=min(18, 390/(x1-x0+1), 250/(z1-z0+1))
    origin_x=x+(396-scale*(x1-x0+1))/2
    origin_y=y+(256-scale*(z1-z0+1))/2
    high={}
    for p in cells|truth:
        key=(p[0],p[2])
        if key not in high or p[1]>high[key][1]:
            high[key]=p
    for (cx,cz),p in high.items():
        u=origin_x+(cx-x0)*scale
        v=origin_y+(cz-z0)*scale
        color='#43c997' if p in cells and p in truth else '#f26363' if p in cells else '#658ff6'
        pen.rectangle((int(u),int(v),int(u+scale)-1,int(v+scale)-1),fill=color)


def sheet(args):
    measured=read(args.predictions/'cases.json')
    by_id={r['id']:r for r in measured}
    keys=dict(VIEWS)
    switchback=next(r['id'] for r in measured if r['split']=='new' and 'switchback' in r['id'] and 'high' in r['id'])
    keys[switchback]=('Switchback landing', 'High right')
    canvas=Image.new('RGB',(1272,1055),'#13232c')
    pen=ImageDraw.Draw(canvas)
    def label(x,y,text,size=19,fill='#eef4f6'):
        pen.text((x,y),text,font=font(size),fill=fill)
    label(22,14,'Depth seen from above',33)
    label(22,58,'Green: correct     Red: extra     Blue: missing visible block',20)
    label(22,93,'Top-down panels share the same grid and origin within each column. Each square shows the highest scored block.',17)
    inputs={f['id']:(split,f,prior) for split,f,prior in cases(['old','extra','new']) if f['id'] in keys}
    for i,(key,(name,angle)) in enumerate(keys.items()):
        split,f,_=inputs[key]
        x=20+419*i
        label(x,137,name,25)
        label(x,171,angle,17,'#b5c8d0')
        builds=NEW_INPUTS if split=='new' else transfer.BUILDS
        rgb=builds/f['directory']/'captures'/(f['source_id']+'.png')
        with Image.open(checked(rgb,f['files']['rgb.png'])) as im:
            thumb=ImageOps.contain(im.convert('RGB'),(396,228),Image.Resampling.LANCZOS)
            canvas.paste(thumb,(x+(396-thumb.width)//2,202))
        truth,_,_,_=references('old',f,NATIVE_INPUTS[split],TRAINING_DATA,DEPTH,
                               dict(native=native.BUILDS,old=builds,fresh=builds))
        z,folder=members(split)
        before=aligned(set(map(tuple,json.loads(z.read(f'{folder}/{key}/cells.json'))['world'])),truth)
        after=aligned(set(map(tuple,read(args.predictions/split/key/'cells.json')['world'])),truth)
        all_cells=truth|before|after
        bounds=min(p[0] for p in all_cells),min(p[2] for p in all_cells),max(p[0] for p in all_cells),max(p[2] for p in all_cells)
        row=by_id[key]
        for y,title,cells,metric in [(440,'Before',before,row['previous_f1']),(740,'After',after,row['f1'])]:
            label(x,y,f'{title}   {metric:.1%} F1',21)
            label(x,y+30,f'{len(cells&truth)}/{len(truth)} visible blocks   {len(cells-truth)} extra',17,'#b5c8d0')
            draw_footprint(canvas,cells,truth,bounds,x,y+52)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(args.out,quality=91)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    sheet(p.parse_args())
