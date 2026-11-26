"""Draw numbered material errors from a Blocker Picker results file."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from engine.geometry import face


def read(path):
    return json.loads(path.read_text())


def short(name):
    return name.removeprefix('minecraft:')


def pretty(name):
    return short(name).replace('_', ' ')


def select(rows):
    wrong = [r for r in rows if short(r['block_id']) != short(r['truth'])]
    chosen = []
    for kind, limit in [('predicted', 8), ('fixed', 12)]:
        pool = [r for r in wrong if r['kind'] == kind]
        pairs, buildings, frames, truths = set(), set(), set(), set()
        while pool and len(pairs) < limit:
            def priority(r):
                pair = (r['truth'], r['block_id'])
                building = r['key'].split('/')[-1].split('-')[1]
                return (pair not in pairs, r["truth"] not in truths, building not in buildings, r['key'] not in frames)
            item = max(pool, key=priority)
            pair = (item['truth'], item['block_id'])
            if pair in pairs:
                break
            chosen.append(item)
            pairs.add(pair)
            truths.add(item["truth"])
            frames.add(item['key'])
            buildings.add(item['key'].split('/')[-1].split('-')[1])
            pool.remove(item)
    return chosen


def render(result, atlas, out):
    rows = read(result)['rows']
    cases = select(rows)
    materials = read(Path('benchmark/reviews/count-01/materials.json'))
    refs = {(p['key'], tuple(k['xyz']), k['direction']): k for p in materials['pictures'] for k in p['known']}
    sheets = read(atlas/'atlas.json')
    art = Image.open(atlas/sheets['atlas']).convert('RGB')
    sprites = {t['id'].split('/')[-1]: t['rect'] for t in sheets['textures']}
    choices = read(atlas/'placement-index.json')['choices']
    out.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (1280, 118 + 283*((len(cases)+1)//2)), '#eef2f2')
    pen = ImageDraw.Draw(canvas)
    regular = 'C:/Windows/Fonts/segoeui.ttf'
    bold = 'C:/Windows/Fonts/segoeuib.ttf'
    def label(x, y, value, size=17, weight=False, color='#183333'):
        pen.text((x,y), value, font=ImageFont.truetype(bold if weight else regular, size), fill=color)
    label(20,9,'Blocker Picker  /  Wrong material picks',36,True)
    label(20,64,'Cyan outline = tested crop     •     Gold outline = recorded face     •     Swatches show vanilla atlas art at full brightness.')
    for i,r in enumerate(cases):
        x = 12 + 634*(i%2)
        y = 110 + 283*(i//2)
        pen.rounded_rectangle((x,y,x+620,y+270),radius=9,fill='white')
        key = r['key']
        ref = refs[key,tuple(r['xyz']),r['direction']]
        q = np.array(r['adjusted_quad'])
        actual = np.array(ref['quad'])
        image = Image.open(r['image']).convert('RGB')
        pts = np.concatenate([q, actual])
        center = pts.mean(axis=0)
        span = max(32, np.ptp(pts,axis=0).max()*2.8)
        box = np.rint([*(center-span/2),*(center+span/2)]).astype(int)
        side = 144
        context = image.crop(tuple(box)).resize((side,side),Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(context)
        for quad,col in [(actual,'#ffc84a'),(q,'#21e9ed')]:
            path = [tuple(pt) for pt in (quad-box[:2])/(box[2:]-box[:2])*side]
            draw.line(path+[path[0]],fill=col,width=2)
        label(x+12,y+6,f"{i+1:02}   {key.split('/')[-1]}",19,True)
        label(x+12,y+36,('Predicted grid' if r['kind']=='predicted' else 'Recorded corners')+'   /   '+r['direction']+' face   /   '+pretty(key.split('/')[0]),15)
        canvas.paste(context,(x+12,y+75))
        sample = Image.fromarray(face(np.asarray(image),q,16))
        canvas.paste(sample.resize((112,112),Image.Resampling.NEAREST),(x+172,y+89))
        label(x+12,y+224,'Source area',15)
        label(x+173,y+215,'Sampled face',15)
        label(x+12,y+248,'Before adjustment: '+pretty(r['baseline_id']),14)
        for col,identifier,title,pick in [(305,r['block_id'],'Picked',True),(460,r['truth'],'Actual',False)]:
            block = short(identifier)
            matches = [tex for tex,ids in choices[r['direction']].items() if block in ids]
            texture = short(r['texture']) if pick else next(iter(matches),block)
            if texture in sprites:
                u,v,w,h = sprites[texture]
                tile = art.crop((u,v,u+w,v+h)).resize((108,108),Image.Resampling.NEAREST)
                canvas.paste(tile,(x+col,y+90))
            label(x+col,y+202,title,15)
            import textwrap
            for j,line in enumerate(textwrap.wrap(pretty(identifier),17)):
                label(x+col,y+223+j*17,line,15,True)
    canvas.save(out/'wrong-picks.png')
    (out/'sheet-cases.json').write_text(json.dumps([dict(case=i+1,**r) for i,r in enumerate(cases)],indent=2))
    print(len(cases), canvas.size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ('result','atlas','out'):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    render(args.result,args.atlas,args.out)


