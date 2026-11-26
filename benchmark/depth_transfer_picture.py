"""Show the sealed gate source and both aligned exports from above at one scale."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from benchmark.texture_camera_eval import read
from benchmark.texture_camera_sheet import font


def render(inputs,runs,out):
    key='holdout-offsetgate-quartz-high-right'
    truth={tuple(b['xyz_world']) for b in read(inputs/'truth'/key/'labels.json')['blocks']}
    samples=[truth]
    for method in ('baseline','final'):
        folder=runs/method
        shift=next(r['translation'] for r in read(folder/'cases.json') if r['id']==key)
        samples.append({tuple(v+d for v,d in zip(p,shift)) for p in read(folder/key/'cells.json')['world']})
    all_cells=set.union(*samples)
    z0,z1=min(p[2] for p in all_cells),max(p[2] for p in all_cells)
    x0,x1=min(p[0] for p in all_cells),max(p[0] for p in all_cells)
    raw=read(inputs/'scoring/captures'/(key+'-blocks.json'))
    occupied=[r for r in raw if r['y']>64 and r['block_id'].split(':')[-1]!='air']
    xmin,xmax=min(r['x'] for r in occupied),max(r['x'] for r in occupied)
    zmin,zmax=min(r['z'] for r in occupied),max(r['z'] for r in occupied)
    outside=[y for x,y,z in samples[-1] if x<xmin or x>xmax or z<zmin or z>zmax]
    origin_y=min(r['y'] for r in occupied)
    depths=[max(p[2] for p in cells)-min(p[2] for p in cells)+1 for cells in samples]
    sheet=Image.new('RGB',(1280,695),'#16232e')
    pen=ImageDraw.Draw(sheet)
    pen.text((24,14),'Offset gate, high-right camera',font=font(33),fill='white')
    pen.text((24,62),'The two entrance beams sit at different heights.',font=font(21),fill='#bfced8')
    with Image.open(inputs/'scoring/captures'/(key+'.png')) as source:
        thumb=ImageOps.contain(source.convert('RGB'),(438,257))
        sheet.paste(thumb,(23,105))
    pen.text((489,111),'Reference depth',font=font(23),fill='white')
    pen.text((489,150),f'{depths[0]} blocks',font=font(32),fill='#68dbaa')
    pen.text((735,111),'Previous export',font=font(23),fill='white')
    pen.text((735,150),f'{depths[1]} blocks',font=font(32),fill='#ef968b')
    pen.text((1000,111),'New export',font=font(23),fill='white')
    pen.text((1000,150),f'{depths[2]} blocks',font=font(32),fill='#ef968b')
    pen.text((489,226),f'{len(outside)} exported blocks extend outside the structure footprint.',font=font(22),fill='white')
    pen.text((489,262),f'{outside.count(origin_y+4)} lie at the lower roof height and {outside.count(origin_y+8)} at the high lintel.',
             font=font(21),fill='#bfced8')
    pen.text((489,309),'All three top-down grids share an origin and scale.',font=font(19),fill='#bfced8')
    step=min(9,400//(z1-z0+1),207//(x1-x0+1))
    for i,(cells,label) in enumerate(zip(samples,('Visible reference','Previous export','New export'))):
        u=24+420*i
        pen.text((u,378),label,font=font(24),fill='white')
        for x in range(x0,x1+1):
            for z in range(z0,z1+1):
                a=u+(z-z0)*step
                b=428+(x-x0)*step
                pen.rectangle((a,b,a+step-2,b+step-2),fill='#253849')
        top={}
        for x,y,z in cells:
            top[x,z]=max(y,top.get((x,z),y))
        for (x,z),y in top.items():
            a=u+(z-z0)*step
            b=428+(x-x0)*step
            color='#87bfff' if i==0 else '#53d39b' if (x,y,z) in truth else '#ee645f'
            pen.rectangle((a,b,a+step-2,b+step-2),fill=color)
    pen.text((25,648),'Blue = visible target     Green = correct     Red = extra     Each square shows the highest block in its column.',font=font(19),fill='#d9e5ef')
    sheet.save(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('inputs','runs','out'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    render(a.inputs,a.runs,a.out)
