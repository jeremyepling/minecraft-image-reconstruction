"""Show the complete source images, proposed shared edge and graded pixel transfers."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from benchmark.edge_vote_outline import adjustments
from benchmark.edge_vote_outline_corroborated import guarded_pixels,witnesses,WITNESSES
from benchmark.edge_vote_outline_eval import OUT,baseline_pack,recount
from benchmark import facade_inset_round as suite
from benchmark.texture_camera_sheet import font
from benchmark.wall_texture_evaluate import metrics
from engine import inset_returns,surface_model,texture_guided_count
from engine.surface_boundary_refine import _retain_topology
from engine.surface_perimeter import with_outline


def label(pc,xy,text,small=False):
    ImageDraw.Draw(pc).text(xy,text,font=font(17 if small else 24),fill='white')


def panel(job,mode,name):
    im,orig,counts,cam=suite.ordinary_inputs(*job)
    parts,strict,phase,extra,_=inset_returns.detect(im,orig,'26.3',camera=cam)
    if parts is not orig:
        parts=[with_outline(p) for p in parts]
        atlas=None
        if 'atlas_phase' in counts:
            from engine.atlas_count import load_atlas
            atlas=load_atlas(suite.ATLAS,'26.3')
        counts=recount(im,parts,atlas)
        surface_model._inset_layout_counts(im,'26.3',[with_outline(p) for p in orig],counts,
            {e['host'] for e in strict+extra},texture_guided_count.METHOD,atlas)
    pack,cells=baseline_pack(job)
    owner,_,options=adjustments(im,parts,counts,cam,cells)
    if mode=='protected':
        options=[e for e in options if e['cells']>=4 and e['margin']>=4 and e['strength']>=10
                 and abs(e['offset'])<=1 and np.any(e['clear'])]
    truth=suite.visible_map(job[0],job[1])
    grades=metrics(parts,truth,im.size)
    byid={p['id']:p['dominant'] for p in grades['parts']}
    reference=np.array([0]+[byid[p['id']] for p in parts])
    best=None
    for e in options:
        new=owner.copy()
        take=e['clear'] if mode=='protected' else slice(None)
        new.flat[e['flat'][take]]=e['after'][take]
        if mode=='protected':new=_retain_topology(owner,new,dict(radius=3))
        moved=new!=owner
        if not moved.any():continue
        good=moved&(reference[new]==truth)&(reference[owner]!=truth)
        bad=moved&(reference[owner]==truth)&(reference[new]!=truth)
        score=int(moved.sum())
        if best is None or score>best[0]:best=(score,e,good,bad,moved)
    if mode=='ray':
        witness=witnesses(WITNESSES,job[1]['id'],job[1]['files']['rgb.png'],pack)
        owner,new,ops=guarded_pixels(im,parts,counts,cam,cells,witness)
        assert len(ops)==1
        moved=new!=owner
        good=moved&(reference[new]==truth)&(reference[owner]!=truth)
        bad=moved&(reference[owner]==truth)&(reference[new]!=truth)
        e=ops[0]
    else:
        _,e,good,bad,moved=best
    x0,y0,x1,y1=e['xyxy'];pad=15
    x0,y0,x1,y1=max(0,x0-pad),max(0,y0-pad),min(im.width,x1+pad),min(im.height,y1+pad)
    region=(x0,y0,x1,y1)
    full=im.copy();pen=ImageDraw.Draw(full);pen.rectangle(region,outline='#ffbf51',width=4)
    rgb=np.array(im)
    rgb[good]=(104,255,160);rgb[bad]=(255,79,100);rgb[moved&~(good|bad)]=(255,209,76)
    seen=Image.fromarray(rgb)
    tile=Image.new('RGB',(560,770),'#1b2730')
    label(tile,(12,8),name)
    label(tile,(12,41),'Full source image and location of candidate edge',True)
    def paste(im,top,height):
        tmp=ImageOps.contain(im,(534,height),Image.Resampling.NEAREST if top>320 else Image.Resampling.LANCZOS)
        tile.paste(tmp,((560-tmp.width)//2,top+(height-tmp.height)//2))
    paste(full,70,255)
    label(tile,(12,335),f"Source pixels along {e['cells']} adjoining blocks",True)
    paste(im.crop(region),365,150)
    label(tile,(12,524),'Detail of reassigned source pixels',True)
    yy,xx=np.nonzero(moved)
    ymid,xmid=int(np.median(yy)),int(np.median(xx))
    detail=(max(0,xmid-22),max(0,ymid-14),min(im.width,xmid+23),min(im.height,ymid+15))
    paste(seen.crop(detail),555,150)
    label(tile,(12,711),f"Green: {good.sum()} fixed     Red: {bad.sum()} previously correct",True)
    label(tile,(12,739),f"Yellow: {(moved&~(good|bad)).sum()} other transfers",True)
    return tile


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cases',nargs='+',help='mode|frame-id|column title');p.add_argument('--out',type=Path,default=OUT/'comparison.jpg')
    a=p.parse_args()
    jobs={j[1]['id']:j for j in suite.jobs(list(suite.framed.SIZES)+['shelter'])}
    sheet=Image.new('RGB',(560*len(a.cases),770),'#1b2730')
    for k,case in enumerate(a.cases):
        mode,key,title=case.split('|')
        sheet.paste(panel(jobs[key],mode,title),(k*560,0))
    sheet.save(a.out,quality=96)
