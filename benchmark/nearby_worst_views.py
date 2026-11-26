"""Rank frozen evaluation views and draw their source and scored visible geometry."""
import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import visible_depth_native as native, visible_depth_transfer as transfer
from benchmark.nearby_thin_comparison import EVALUATION, EXTRA_INPUTS, decode, records
from benchmark.relative_geometry_replay import references
from benchmark.texture_camera_eval import DATA, STABLE, checked, frames, read, save
from benchmark.texture_camera_sheet import font
from training.visible_labels import visible_labels


BG = '#14212c'
PANEL = '#223543'
INK = '#f2f6f8'
MUTED = '#b6c8d4'
COLORS = ((71,212,150),(253,83,82),(94,164,255))


def f1(row):
    m = row['relative_geometry']
    return Fraction(2*m['correct'],m['placed']+m['target'])


def samples(split, extra_inputs):
    folder = {'old':transfer.OUT,'fresh':STABLE/'runs/canopy-placement/confirmation','extra':extra_inputs}.get(split)
    if folder is not None:
        return read(folder/'screen.json')['selected'], folder
    return (native.rows() if split=='native' else frames(DATA,split)), None


def image_paths(split, frame):
    if split in ('old','fresh','extra','native'):
        path = native.rgb_path(frame) if split=='native' else transfer.rgb_path(frame)
        return path,path.with_suffix('.json')
    folder = DATA/'data'/split/frame['directory']
    return folder/'rgb.png',folder/'camera.json'


def title(key):
    if key.startswith('legacy-'):
        ground,_,parts=key.removeprefix('legacy-').split('-',2)
        return parts.replace('-',' ').title()+'  ·  '+ground.replace('_',' ')
    return key.split('--')[-1].removeprefix('train-').removeprefix('test-').replace('-',' ').title()


def overlay(image, pose, truth, placed):
    # Jointly rasterize the colored cells. A ray gets the color of its nearest face.
    states=dict.fromkeys(truth|placed,dict(block_id='smooth_stone',state='Block{minecraft:smooth_stone}'))
    labels, ids, _ = visible_labels(pose,states,(0,0,0))
    lut=np.zeros((len(labels['faces'])+1,3))
    for face in labels['faces']:
        pos=tuple(face['xyz_world'])
        color=COLORS[0] if pos in placed and pos in truth else COLORS[1] if pos in placed else COLORS[2]
        axis=np.argmax(np.abs(face['normal_world']))
        lut[face['id']]=np.array(color)*(.90,1.,.81)[axis]
    pixels=np.array(image)
    keep=ids>0
    pixels[keep]=.76*lut[ids[keep]]+.24*pixels[keep]
    edge=np.zeros_like(keep)
    edge[:,1:] |= ids[:,1:]!=ids[:,:-1]
    edge[1:] |= ids[1:]!=ids[:-1]
    pixels[keep&edge]=(pixels[keep&edge]*.61).astype(np.uint8)
    x0,y0,x1,y1=Image.fromarray(keep).getbbox()
    margin=max(10,round(.08*max(x1-x0,y1-y0)))
    bounds=(max(0,x0-margin),max(0,y0-margin),min(image.width,x1+margin),min(image.height,y1+margin))
    return image.crop(bounds),Image.fromarray(pixels).crop(bounds)


def picture(canvas, image, x, y, width, height):
    ImageDraw.Draw(canvas).rectangle((x,y,x+width-1,y+height-1),fill=PANEL)
    tile=ImageOps.contain(image,(width,height),Image.Resampling.LANCZOS)
    canvas.paste(tile,(x+(width-tile.width)//2,y+(height-tile.height)//2))


def sheet(rows, headline, destination):
    width,pad,gap=1440,32,18
    col=(width-2*pad-gap)//2
    top,h=161,374
    canvas=Image.new('RGB',(width,top+h*len(rows)+40),BG)
    pen=ImageDraw.Draw(canvas)
    pen.text((pad,17),headline,font=font(34),fill=INK)
    for x,color,label in zip((pad,359,736),COLORS,('Correct block','Extra block','Missing visible block')):
        pen.rectangle((x,79,x+21,100),fill=color)
        pen.text((x+33,72),label,font=font(23),fill=INK)
    pen.text((pad,124),'Source screenshot',font=font(21),fill=MUTED)
    pen.text((pad+col+gap,124),'Scored geometry over the same crop',font=font(21),fill=MUTED)
    for i,(name,detail,images) in enumerate(rows):
        y=top+i*h
        pen.text((pad,y),name,font=font(24),fill=INK)
        pen.text((width-pad,y),detail,font=font(21),fill=INK,anchor='ra')
        for j,tile in enumerate(images):
            picture(canvas,tile,pad+j*(col+gap),y+43,col,h-61)
    pen.text((pad,top+h*len(rows)),'Colors identify the nearest projected face. The counts include blocks behind other colored faces.',font=font(18),fill=MUTED)
    canvas.save(destination,quality=90)


def empty_sheet(rows,destination):
    canvas=Image.new('RGB',(1360,112+196*((len(rows)+1)//2)),BG)
    pen=ImageDraw.Draw(canvas)
    pen.text((26,16),'Lowest-scoring views · camera abstentions',font=font(32),fill=INK)
    distinct=len({r['rgb_sha256'] for r,_ in rows})
    pen.text((26,68),f'{len(rows)} builds at 0% F1 · {distinct} distinct source images across the photographed floor settings',font=font(21),fill=MUTED)
    for i,(entry,image) in enumerate(rows):
        x=26+(i%2)*664
        y=110+(i//2)*196
        picture(canvas,image,x,y,245,167)
        floor,_,parts=entry['id'].removeprefix('legacy-').split('-',2)
        pen.text((x+262,y+16),parts.replace('-close-front','').replace('-',' ').title(),font=font(23),fill=INK)
        pen.text((x+262,y+58),floor.replace('_',' ')+' floor',font=font(21),fill=MUTED)
        pen.text((x+262,y+94),f"0 / {entry['target']} visible · 0 extra",font=font(21),fill=MUTED)
    canvas.save(destination,quality=90)


def footprint(name, truth, placed, destination):
    canvas=Image.new('RGB',(1380,644),BG)
    pen=ImageDraw.Draw(canvas)
    pen.text((30,20),name+' · depth seen from above',font=font(32),fill=INK)
    pen.text((30,79),'Same origin and scale in both panels. Each colored square is the highest scored block in its column.',font=font(21),fill=MUTED)
    low=np.min(list(truth|placed),axis=0)
    high=np.max(list(truth|placed),axis=0)
    nx,_,nz=high-low+1
    matches=truth&placed
    unit=min(10,604//int(nz),335//int(nx))
    for i,(label,cells) in enumerate([('Visible target',truth),('Exported geometry',placed)]):
        x,y=34+i*689,184
        pen.text((x,y-43),label,font=font(25),fill=INK)
        for xx in range(int(nx)):
            for zz in range(int(nz)):
                u,v=x+zz*unit,y+xx*unit
                pen.rectangle((u,v,u+unit-2,v+unit-2),fill=PANEL)
        columns={}
        for cx,cy,cz in cells:
            columns[cx,cz]=max(cy,columns.get((cx,cz),cy))
        for (cx,cz),cy in columns.items():
            u,v=x+int(cz-low[2])*unit,y+int(cx-low[0])*unit
            correct=(cx,cy,cz) in matches
            pen.rectangle((u,v,u+unit-2,v+unit-2),fill=COLORS[0] if correct else COLORS[2 if i==0 else 1])
        span=max(c[2] for c in cells)-min(c[2] for c in cells)+1
        pen.text((x,536),f'{span} blocks across the Z axis',font=font(24),fill=INK)
    for x,c,label in zip((31,365,810),COLORS,('Correct block','Extra block','Missing visible block')):
        pen.rectangle((x,604,x+21,625),fill=c)
        pen.text((x+33,597),label,font=font(21),fill=INK)
    canvas.save(destination,quality=93)


def build(args):
    args.out.mkdir(parents=True,exist_ok=True)
    output, selected, empty = {}, {}, []
    for name,splits,count in [('fixed',EVALUATION,3),('alternate',('extra',),2)]:
        indexed=[]
        for split in splits:
            frameset,fixture=samples(split,args.extra_inputs)
            source=args.predictions/split
            rows,_=records(source,{f['id'] for f in frameset})
            indexed.extend((split,source,f,fixture,rows[f['id']]) for f in frameset)
        indexed.sort(key=lambda item:(f1(item[-1]),item[2]['id']))
        output[name]=[]
        cards=[]
        for split,source,f,fixture,row in indexed:
            key=f['id']
            m=row['relative_geometry']
            record=dict(id=key,split=split,f1=float(f1(row)),rgb_sha256=row['rgb_sha256'],
                        pack_sha256=row['pack_sha256'],**{k:m[k] for k in ('target','placed','correct','extras','missed')})
            output[name].append(record)
            if m['placed'] and len(cards)>=count:
                continue
            rgb,pose_path=image_paths(split,f)
            with Image.open(checked(rgb,f['files']['rgb.png'])) as src:
                image=src.convert('RGB')
            if not m['placed']:
                assert not row['camera_supported'] and f1(row)==0
                empty.append((record,image))
                continue
            pose=read(checked(pose_path,f['files']['camera.json']))
            truth,_,transform,hashes=references('old' if split=='extra' else split,f,fixture,DATA,
                STABLE/'runs/visible-depth-rounds',dict(native=native.BUILDS,old=transfer.BUILDS,fresh=transfer.BUILDS))
            cells=decode(source,row,dict(hashes,rgb_sha256=f['files']['rgb.png']),transform)
            dx,dy,dz=row['translation']
            placed={(x+dx,y+dy,z+dz) for x,y,z in cells}
            assert len(placed&truth)==m['correct'] and len(placed-truth)==m['extras'] and len(truth-placed)==m['missed']
            if name=='fixed' and not cards:
                footprint(title(key),truth,placed,args.out/'worst-footprint.png')
            detail=f"{float(f1(row)):.2%} F1     {m['correct']}/{m['target']} correct     {m['extras']} extra     {m['missed']} missing"
            cards.append((title(key),detail,overlay(image,pose,truth,placed)))
        selected[name]=[entry['id'] for entry in output[name] if entry['placed']][:count]
        sheet(cards,f"Weakest nonempty builds · {len(indexed)} {name} views",args.out/(name+'.jpg'))
    if empty:
        empty.sort(key=lambda pair:(pair[0]['id'].split('-',2)[-1],pair[0]['id']))
        empty_sheet(empty,args.out/'camera-abstentions.jpg')
    save(args.out/'ranking.json',dict(sort='ascending exact relative F1, then frame ID',selected=selected,
         abstentions=[r['id'] for r,_ in empty],rankings=output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True,help='Evaluated run with fixed and extra cohort folders')
    parser.add_argument('--extra-inputs',type=Path,default=EXTRA_INPUTS)
    parser.add_argument('--out',type=Path,required=True)
    build(parser.parse_args())
