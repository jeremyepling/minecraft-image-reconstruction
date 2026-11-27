"""Score source-supported outline gaps on the 104 fixed views at native resolution."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark import facade_inset_round as suite
from benchmark.shared_face_error_budget import current_stages, LATEST
from benchmark.surface_sheet import panel, text
from benchmark.texture_camera_eval import read, save
from benchmark.wall_texture_evaluate import metrics
from engine.outline_gaps import candidates
from engine.flat_surface_extent import proposals
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs
from engine.surface_perimeter import draw_outline

FOCUS = 'legacy-light_gray_concrete-dev-shopfront-plaster-low-left'
REGISTRY = 'legacy-light_gray_concrete-holdout-registry-plaster-high-right'
REVIEW = suite.ROOT/'benchmark/reviews/shopfront-outline-gaps-01'


def check(task):
    job, row, output, veto, row_veto, patch_veto = task
    split, frame, _, _ = job
    image, parts, _, _ = current_stages(job,row)
    bracketed = candidates(image,parts)
    flat = proposals(image,parts,exterior_veto=veto,row_veto=row_veto,patch_veto=patch_veto)
    events = bracketed + flat
    result = dict(id=frame['id'], source=frame['files']['rgb.png'], events=[], modes={})
    truth = suite.visible_map(split,frame) if events or frame['id'] in (FOCUS,REGISTRY) else None
    if events:
        base = metrics(parts,truth,image.size)
        original = {p['id']:p for p in parts}
        rows = {r['id']:r for r in base['parts']}
        for e in events:
            target = rows[e['surface']]['dominant']
            indices = np.concatenate([truth[y,a:b] for y,a,b in e['runs']])
            result['events'].append(dict(**{k:v for k,v in e.items() if k!='runs'}, dominant_host=target,
                               on_host=int((indices==target).sum()), background=int((indices==0).sum())))
        for name,selected in [('any',bracketed),('cross',[e for e in bracketed if e['both_axes'] >= e['pixels']*.5]),('flat',flat)]:
            changed = {}
            for e in selected:
                key=e['surface']
                if key not in changed:
                    changed[key]=native_mask(original[key],image.size).copy()
                for y,a,b in e['runs']:
                    changed[key][y,a:b]=True
            after=[dict(p,support=dict(size=list(image.size),runs=row_runs(changed[p['id']]))) if p['id'] in changed else p for p in parts]
            score=metrics(after,truth,image.size) if selected else base
            result['modes'][name]=dict(gained=sorted(set(score['matched'])-set(base['matched'])),
                             lost=sorted(set(base['matched'])-set(score['matched'])),
                             paired=score['paired_pixels']-base['paired_pixels'],background=score['background_pixels']-base['background_pixels'])
    if frame['id'] in (FOCUS,REGISTRY):
        mark=np.asarray(image).copy()
        inspected=np.zeros(truth.shape,bool)
        for e in flat:
            for y,a,b in e['runs']:
                inspected[y,a:b]=True
        mark[inspected]=mark[inspected]*.28+np.array([75,224,223])*.72
        edges=image.copy()
        for p in parts:
            draw_outline(edges,p,'#ff9842',2)
        shop=frame['id']==FOCUS
        box=(645,367,943,582) if shop else (895,250,1202,627)
        sheet=Image.new('RGB',(640,1200),'#f8f7f1')
        pen=ImageDraw.Draw(sheet)
        text(pen,15,12,'Shopfront · plaster, low left' if shop else 'Registry · plaster, high right',27,True)
        for i,(pic,title) in enumerate([(image,'Source image'),(edges,'Current surface outlines'),(Image.fromarray(mark),'Teal shows the proposed additional visible pixels')]):
            text(pen,15,54+382*i,title,20,True)
            panel(sheet,pic,box,16,86+382*i,608,342)
        sheet.save(output/('plaster-gap.jpg' if shop else 'registry-right.jpg'),quality=94)
        region=np.s_[490:545,790:850]
        focus=np.zeros(truth.shape,bool)
        if shop:
            focus[region]=True
        else:
            focus=truth==10
        result['center']=dict(proposed=int(inspected[focus].sum()),area=int(focus.sum()),
                              reference_ids=np.unique(truth[focus]).tolist())
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=REVIEW)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--transfer',action='store_true')
    ap.add_argument('--exterior-veto',action='store_true')
    ap.add_argument('--row-veto',action='store_true')
    ap.add_argument('--patch-veto',action='store_true')
    args=ap.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    expected={r['id']:r for r in read(LATEST)['rows']}
    splits=([s for s in [*suite.framed.SIZES,'shelter'] if s not in suite.framed.EVALUATION]
            if args.transfer else suite.framed.EVALUATION)
    jobs=[(j,expected[j[1]['id']],args.out,args.exterior_veto,args.row_veto,args.patch_veto) for j in suite.jobs(splits)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(check,jobs))
    save(args.out/'cases.json',rows)
    for scope in ('shopfront','registry','other'):
        group=[r for r in rows if (next((k for k in ('shopfront','registry') if k in r['id']), 'other'))==scope]
        for mode in ('any','cross','flat'):
            changes=[r['modes'][mode] for r in group if r['modes']]
            print(scope,mode,'frames',len(group),'matches +/-',sum(len(m['gained']) for m in changes),sum(len(m['lost']) for m in changes),
                  'paired/background',*(sum(m[k] for m in changes) for k in ('paired','background')))
    for row in rows:
        if 'center' in row:
            print(row['id'],row['center'])


if __name__=='__main__':
    main()
