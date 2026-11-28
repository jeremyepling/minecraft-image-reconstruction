"""Pooled decoded outcomes and source partition overlays for the regional ruler experiment."""
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.floating_mixed_wall_refit import ARMS, OUT, REVIEW, jobs, read, save, project


def summarize_rows(rows):
    result={}
    for arm in ('base',*ARMS):
        totals={}
        for field in ('geometry','scores'):
            for kind in ('strict','facade'):
                agg=Counter()
                for row in rows:
                    score=row['score'][arm][field][kind]
                    agg.update({k:score[k] for k in ('correct','missed','extras')})
                c,m,e=(agg[k] for k in ('correct','missed','extras'))
                totals[field+'_'+kind]=dict(agg,f1=2*c/(2*c+m+e))
        totals['paired']={k:sum(r['paired'][arm][k] for r in rows) for k in ('gained','lost','added','removed')}
        result[arm]=totals
    return result


def publish(rows):
    groups={'all':rows,'fixed_sky':[r for r in rows if r['audit']['panel']=='sky']}
    for dataset in sorted({r['dataset'] for r in rows}):
        groups[dataset]=[r for r in rows if r['dataset']==dataset]
    save(REVIEW/'summary.json',{k:dict(conditions=len(v),arms=summarize_rows(v)) for k,v in groups.items()})
    cases={(group,c['id']):c for group,c in jobs()}
    working=[r for r in rows if r['audit']['regions']]
    working.sort(key=lambda r:('stable-yard' not in r['id'],r['id']))
    sheet=Image.new('RGB',(450*len(working),490),(24,28,30))
    font=ImageFont.truetype('arial.ttf',17)
    labels=ImageDraw.Draw(sheet)
    for i,row in enumerate(working):
        group,name=row['audit']['panel'],row['id'];case=cases[group,name]
        ids=np.asarray(Image.open(OUT/group/'source'/name/'region-ids.png'))
        rgb=np.asarray(Image.open(case['variants']['sky']['image']).convert('RGB'))
        yy,xx=np.nonzero(ids);crop=(xx.min(),yy.min(),xx.max()+1,yy.max()+1)
        original=Image.fromarray(np.where((ids>0)[...,None],rgb,43).astype('uint8'))
        colors=np.random.default_rng(4).integers(30,235,(int(ids.max())+1,3))
        painted=(.45*rgb+.55*colors[ids]).astype('uint8')
        painted[ids==0]=43
        overlay=Image.fromarray(painted)
        draw=ImageDraw.Draw(overlay)
        packet=read(OUT/group/'visible-patches/charts/sky'/name/'charts.json')
        for cid,parts in row['audit']['regions'].items():
            for part in parts:
                if 'visible-patches' not in part.get('evidence',{}):
                    continue
                chart=next(c for c in packet['charts'] if c['id']==cid+'~r'+str(part['part']))
                for cell in chart['cells']:
                    corners=project(np.asarray(chart['H']),np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])+[cell['u'],cell['v']])
                    draw.line(list(map(tuple,corners)),fill='#3ff5f1',width=2)
        labels.text((450*i+6,4),name.replace('train-',''),font=font,fill='white')
        for j,image in enumerate((original,overlay)):
            view=image.crop(crop);view.thumbnail((438,215),Image.Resampling.LANCZOS)
            sheet.paste(view,(450*i+(450-view.width)//2,28+j*231))
    sheet.save(REVIEW/'source-regions.jpg',quality=91)
