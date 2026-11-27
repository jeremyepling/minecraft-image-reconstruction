"""Grade native wall textures and boundaries against recorded surfaces after image-only proposals."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from benchmark import framed_depth_eval as f
from benchmark.roof_span_worst_stages import read_view
from benchmark.texture_camera_eval import checked,digest,read,save
from benchmark.texture_camera_sheet import font
from benchmark.wall_texture_masks import rewrite
from engine.surface_perimeter import _native_alpha,draw_outline


RUN=Path('../facade-placement-round/runs/production')
OUT=Path('benchmark/reviews/wall-texture-seams-01')
SOURCE=Path('benchmark/reviews/facade-placement-01/worst/ranking.json')
MODES=('split','join','shade','prune')
NAMES={'Canopy 04':'split','Shopfront':'shade','Facade':'join'}


@lru_cache(maxsize=1)
def names():
    return {r['representative']['id']:r['name'] for r in read(SOURCE)['layouts'] if r['name'] in NAMES}


def metrics(parts,truth,size):
    sizes=np.bincount(truth.ravel()); overlaps=[]; partrows=[]
    for part in parts:
        alpha,(x,y,r,b)=_native_alpha(part['support'],size)
        mask=np.asarray(alpha,bool); area=int(mask.sum())
        n=np.bincount(truth[y:b,x:r][mask],minlength=len(sizes))
        winners=np.argsort(n[1:])[::-1]+1
        iou=n[1:]/(area+sizes[1:]-n[1:]).clip(1)
        overlaps.extend((float(iou[k-1]),part['id'],int(k),int(n[k])) for k in np.flatnonzero(iou>=.5)+1)
        partrows.append(dict(id=part['id'],pixels=area,background=int(n[0]),
                             dominant=int(winners[0]),second=int(winners[1]) if len(winners)>1 else 0,
                             correct_if_dominant=int(n[winners[0]]),intersections={int(k):int(n[k]) for k in winners if n[k]>0}))
    matched={};matched_part=set();paired=0
    for iou,key,sid,pixels in sorted(overlaps,reverse=True):
        if sid not in matched and key not in matched_part:
            matched[sid]=key;matched_part.add(key);paired+=pixels
    return dict(matched=sorted(matched),predicted=len(parts),target=len(sizes)-1,
                paired_pixels=paired,unmatched_predictions=len(parts)-len(matched),
                background_pixels=sum(p['background'] for p in partrows),parts=partrows)


def draw_view(image,before,after,truth,name,out):
    ys,xs=np.nonzero(truth);box=(max(0,xs.min()-9),max(0,ys.min()-9),min(image.width,xs.max()+10),min(image.height,ys.max()+10))
    ref=np.array(image)
    edge=np.zeros(truth.shape,bool)
    edge[:,1:]|=(truth[:,1:]!=truth[:,:-1])&((truth[:,1:]>0)|(truth[:,:-1]>0))
    edge[1:,:]|=(truth[1:,:]!=truth[:-1,:])&((truth[1:,:]>0)|(truth[:-1,:]>0))
    ref[edge]=[83,240,221]
    orig=image.copy();changed=image.copy()
    for part in before:draw_outline(orig,part,'#ff9d46',2)
    previous={p['id'] for p in before}
    for part in after:draw_outline(changed,part,'#ff9d46' if part['id'] in previous else '#52edbf',2)
    pictures=[image,orig.crop(box),changed.crop(box),Image.fromarray(ref).crop(box)]
    w,h=550,340
    panel=Image.new('RGB',(w,h*4+75),'#152530');pen=ImageDraw.Draw(panel)
    pen.text((9,8),name,fill='#ffffff',font=font(23))
    for j,(label,pic) in enumerate(zip(('Full source photograph','Original Surface Finder','Trial on the same source pixels','Recorded visible boundaries'),pictures)):
        top=45+j*h
        pen.text((10,top),label,fill='white',font=font(18));thumb=ImageOps.contain(pic,(w-10,h-34))
        panel.paste(thumb,((w-thumb.width)//2,top+26))
    panel.save(out/(name.replace(' ','-').lower()+'.jpg'),quality=93)


def score(job):
    split,fr,prior,row,out=job
    key=fr['id']
    image,parts,_,cam=f.photo(split,fr,prior)
    candidates={mode:rewrite(image,parts,cam,mode) for mode in MODES}
    # Open captured world geometry only after all predictions have been produced.
    _,_,_,_,_,labels,ids,*_=read_view(RUN,split,fr,prior,row)
    lut=np.zeros(len(labels['faces'])+1,int)
    for face in labels['faces']:lut[face['id']]=face['surface_id']
    truth=lut[ids]
    base=metrics(parts,truth,image.size)
    result={}
    for mode,(pp,ops) in candidates.items():
        m=metrics(pp,truth,image.size) if ops else base
        gained=set(m['matched'])-set(base['matched']);lost=set(base['matched'])-set(m['matched'])
        result[mode]=dict(ops=ops,gained=sorted(gained),lost=sorted(lost),
                          unmatched_predictions=m['unmatched_predictions'],matched=len(m['matched']),
                          paired_pixels=m['paired_pixels'],background_pixels=m['background_pixels'])
        if key in names():
            result[mode]['parts']=m['parts']
            if NAMES[names()[key]]==mode:
                draw_view(image,parts,pp,truth,names()[key],out)
    if key not in names():base.pop('parts')
    return dict(id=key,split=split,rgb_sha256=row['rgb_sha256'],input_hashes=row['input_hashes'],
                reference_hashes=row['reference_hashes'],baseline=base,methods=result)


def aggregate(rows):
    out={}
    for mode in MODES:
        xx=[r['methods'][mode] for r in rows];bb=[r['baseline'] for r in rows]
        out[mode]=dict(frames=len(rows),changed=sum(bool(o['ops']) for o in xx),
                matched=sum(o['matched'] for o in xx),old_matched=sum(len(b['matched']) for b in bb),
                gained=sum(len(o['gained']) for o in xx),lost=sum(len(o['lost']) for o in xx),
                unmatched=sum(o['unmatched_predictions'] for o in xx),
                old_unmatched=sum(b['unmatched_predictions'] for b in bb),
                paired_pixel_change=sum(o['paired_pixels']-b['paired_pixels'] for o,b in zip(xx,bb)),
                old_paired=sum(b['paired_pixels'] for b in bb),recorded_surfaces=sum(b['target'] for b in bb),
                background_pixel_change=sum(o['background_pixels']-b['background_pixels'] for o,b in zip(xx,bb)))
    return out


def main(out,workers,panel):
    out.mkdir(parents=True,exist_ok=True)
    manifest=read(RUN/'summary.json')
    rows={r['id']:r for r in read(checked(RUN/'cases.json',manifest['cases_sha256']))}
    jobs=[]
    for split,frame,prior in f.cases(('train',*f.EVALUATION)):
        if panel and frame['id'] not in names():continue
        jobs.append((split,frame,prior,rows[frame['id']],out))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scored=list(pool.map(score,jobs))
    save(out/'cases.json',scored)
    totals={g:aggregate([r for r in scored if (r['split']=='train')==(g=='development')]) for g in ('development','fixed')}
    save(out/'summary.json',dict(baseline='028819a',input_sha256=manifest['cases_sha256'],cases_sha256=digest(out/'cases.json'),groups=totals))
    imgs=[Image.open(out/(k.replace(' ','-').lower()+'.jpg')) for k in NAMES]
    sheet=Image.new('RGB',(sum(i.width for i in imgs),max(i.height for i in imgs)))
    for j,im in enumerate(imgs):sheet.paste(im,(j*im.width,0))
    sheet.save(out/'texture-seams.jpg',quality=94)
    print(totals)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--workers',type=int,default=4);p.add_argument('--panel',action='store_true')
    a=p.parse_args();main(a.out,a.workers,a.panel)
