"""Review the current pipeline and compare the three user-flagged surfaces at the original view."""
import gzip
from html import escape
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import digest,read,save
from benchmark.count_review import fit_panel,table
from benchmark.count_round import FINDER,cached_regions
from benchmark.surface_metrics import DOWNSAMPLE
from benchmark.surface_reference import recorded
from benchmark.surface_sheet import text
from engine.surface_perimeter import context_crop,draw_outline,masked_crop


FOCUS=('courtyard','workshop','registry')
NAMES={'regions-first':'Original Regions First','color-edges':'Snap the perimeter to image edges',
       'depth-split':'Separate parallel surfaces using depth'}


def load_parts(out,row,original,frozen,method):
    if method=='regions-first':
        return cached_regions(original,frozen)
    with gzip.open(out/row['predictions'],'rt',encoding='utf8') as stream:
        return json.load(stream)[method]


def reference_part(ids,pid):
    mask=ids==pid
    runs=[]
    for y,line in enumerate(mask):
        borders=np.flatnonzero(np.diff(np.r_[False,line,False]))
        runs.extend([[y,int(a),int(b)] for a,b in borders.reshape(-1,2)])
    return dict(support=dict(size=list(ids.shape[::-1]),runs=runs))


def zoom_box(chart,mask,scene,image_size):
    """Center the detail where a sampling-chart edge strays farthest from the model perimeter."""
    from benchmark.mask_metrics import outline
    edge=(outline(mask)[:,::-1]+.5)*DOWNSAMPLE
    quad=np.asarray(chart)
    t=np.linspace(0,1,81)[:,None,None]
    samples=((1-t)*quad+t*np.roll(quad,-1,axis=0)).reshape(-1,2)
    dist=((samples[:,None]-edge[None])**2).sum(2).min(1)
    cx,cy=samples[dist.argmax()]
    size=min(150,scene[2]-scene[0],scene[3]-scene[1])
    x=int(np.clip(round(cx-size/2),0,image_size[0]-size))
    y=int(np.clip(round(cy-size/2),0,image_size[1]-size))
    return (x,y,x+size,y+size)


def detail_sheet(out,examples,lookup):
    from benchmark.mask_metrics import footprint
    canvas=Image.new('RGB',(1760,122+len(examples)*308),'#f8f7f2')
    draw=ImageDraw.Draw(canvas)
    text(draw,26,14,'Where did the diagonal come from?',33,True)
    text(draw,26,59,'Same screenshot in each column. The last column uses recorded Minecraft geometry for grading.',19)
    for x,title in zip((26,369,715,1062,1407),('Original','Old sampling chart','Surface Finder','Enlarged difference','Recorded surface')):
        text(draw,x,99,title,20,True)
    details=[]
    for index,ex in enumerate(examples):
        y=132+index*308
        image=ex['image']
        view=ex['review']
        _,ref=recorded(lookup[ex['key']])
        actual=reference_part(ref['id_map'],view['recorded_surface'])
        actual_xy=actual['support']
        # The scoring raster consists of four-pixel cells. The suite's native size is an exact multiple.
        assert (np.array(actual_xy['size'])*DOWNSAMPLE==image.size).all()
        pred=image.copy()
        draw_outline(pred,ex['part'],color='#13f5e0',width=3)
        target=image.copy()
        draw_outline(target,actual,color='#ffc543',width=3)
        chart=image.copy()
        q=list(map(tuple,view['old_chart']))
        ImageDraw.Draw(chart).line(q+q[:1],fill='#fb62ba',width=3)
        mask=footprint(ex['part'],ref['id_map'].shape,image.size,True)
        zbox=zoom_box(view['old_chart'],mask,ex['box'],image.size)
        both=chart.copy()
        draw_outline(both,ex['part'],color='#13f5e0',width=2)
        text(draw,26,y,ex['layout'].title(),21,True)
        for x,im,box in [(26,image,ex['box']),(369,chart,ex['box']),(715,pred,ex['box']),
                         (1062,both,zbox),(1407,target,ex['box'])]:
            fit_panel(canvas,im.crop(box),x,y+32,321,201)
        old=view['quad_as_outline']
        proposed=view['methods'][ex['finder']]
        text(draw,369,y+242,f"Overlap  {old['iou']:.0%}",17)
        text(draw,715,y+242,f"Overlap  {proposed['iou']:.0%}",17)
        text(draw,1062,y+242,'Pink chart / cyan perimeter',17)
        text(draw,1407,y+242,'Gold reference perimeter',17)
        text(draw,26,y+273,ex['key'],15)
        details.append(dict(frame=ex['key'],proposal=ex['part']['id'],stage='surface-finder',
                            native_detail_xyxy=list(zbox),native_scene_xyxy=ex['box'],reference_surface=view['recorded_surface']))
        dest=out/'crops'/ex['layout']
        dest.mkdir(parents=True,exist_ok=True)
        cut,tight=masked_crop(image,ex['part'])
        cut.save(dest/'surface.png')
        surroundings,context,local=context_crop(image,ex['part'],ex['margin'])
        surroundings.save(dest/'context.png')
        details[-1].update(tight_xyxy=list(tight),context_xyxy=list(context),tight_xyxy_in_context=list(local),
                           source_png=lookup[ex['key']].path('image').name,margin_px=ex['margin'])
    canvas.save(out/'edge-closeups.jpg',quality=93,subsampling=0)
    save(out/'edge-closeups.json',details)


def review_inputs(run, rows, lookup, originals, frozen, selected, artifacts=None):
    """Use the same eight saved views in the detection review and the pipeline sheet."""
    from benchmark.perimeter_round import COUNT
    index={r['key']:r for r in originals}
    examples=[]
    predictions={}
    for view in read(COUNT/'review-regions.json'):
        key=view['frame']
        row=rows[key]
        case=lookup[key]
        if digest(case.path('image'))!=row['sha256']:
            raise ValueError(f'Screenshot changed for {key}')
        if artifacts is not None and selected!='regions-first':
            path=row['predictions']
            if digest(run/path)!=artifacts[path]:
                raise ValueError(f'Saved predictions changed for {key}')
        with Image.open(case.path('image')) as source:
            image=source.convert('RGB')
        predictions[key]=load_parts(run,row,index.get(key),frozen,selected)
        examples.append(dict(layout=row['layout'],key=key,image=image,box=row['review']['scene_box'],
                             source=case.path('image').name,review=row['review']))
    return examples,predictions


def make_sheet(out,results,lookup,originals,frozen,selected):
    from benchmark.surface_pipeline_sheet import stage_sheet
    from benchmark.surface_detection_sheet import diagnostics
    from engine.atlas_count import load_atlas
    atlas=load_atlas(version='26.3')
    rows={r['key']:r for r in results}
    examples,predictions=review_inputs(out,rows,lookup,originals,frozen,selected)
    detection=diagnostics(out,examples,predictions,lookup,rows,selected)
    for ex in examples:
        review=ex['review']
        focus=review['methods'][selected]
        part=next(p for p in predictions[ex['key']] if p['id']==focus['id'])
        ex.update(part=part,expected=review['expected'] if focus['matched'] else None,
                  margin=8,finder=selected,extra=None)
    detail_sheet(out,[next(ex for ex in examples if ex['layout']==k) for k in FOCUS],lookup)
    # Show one supported grid as well as abstentions on the pre-selected comparison surfaces.
    from engine.masked_count import count
    from engine.surface_sample import sample_surface
    from engine.surface_tiles import visible_tiles
    for ex in examples.copy():
        key=ex['key']
        if rows[key]['cohort']!='development':
            continue
        best=None
        for part in predictions[key]:
            sample=sample_surface(ex['image'],part)
            reading=count(sample)
            n=sum(1 for _ in visible_tiles(sample,reading))
            if n and (best is None or n>best[0]):
                best=n,part
        if best:
            examples.append(dict(ex,part=best[1],extra='Supported block grid',expected=None))
            break
    for ex in examples:
        diagnostic=detection[ex['key']]
        ex['surface_counts']=diagnostic['counts']
        p=next(p for p in diagnostic['predictions'] if p['proposal']==ex['part']['id'])
        ref=next((r for r in diagnostic['references'] if r['id']==p['reference']),None)
        ex['selected_iou']=ref['match_iou'] if ref and ref['eligible'] else None
    stage_sheet(out,examples,atlas)
    stats=read(out/'scores.json')
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           '<title>Surface Finder perimeter review</title><style>body{margin:2rem auto;padding:0 1rem;max-width:1250px;background:#faf9f4;color:#183638;font:16px/1.45 system-ui}img{max-width:100%}table{border-collapse:collapse}td,th{padding:.4rem .6rem;text-align:left;border-bottom:1px solid #cdd6d1}a{color:#106568}summary{cursor:pointer}</style>',
           '<h1>Surface Finder perimeter review</h1><h2>Which surfaces were found or missed?</h2><p>Follow one building down its column. Green marks a scored match, amber a missed surface and pink an extra guess. Blue and gray show small surfaces checked separately. The last two rows enlarge the same spot in the recorded surface and the prediction.</p>',
           '<a href="surface-detection.jpg"><img src="surface-detection.jpg" alt="All building surfaces and close-ups of missed regions in eight pictures"></a>',
           '<p>A scored surface has at least four visible block faces and 24 samples on the four-pixel scoring grid. Small surfaces are counted and drawn from the original-resolution geometry. We check them separately against the remaining predictions. <a href="surface-detection.json">Individual surface IDs, pairings and native close-up coordinates</a></p>',
           '<h2>What happens to a surface after detection?</h2><p>Follow a selected surface through cropping, straightening, counting and texture matching. A missing grid means the counter could not place the lines with confidence.</p>',
           '<a href="comparison.jpg"><img src="comparison.jpg" alt="Eight buildings shown through each stage of reconstruction"></a>',
           '<h2>The three highlighted buildings</h2><p>The older sheet used the four sampling corners as a surface outline. The detail shows where that chart cuts across visible pixels. The gold overlay is the independently recorded surface.</p>',
           '<a href="edge-closeups.jpg"><img src="edge-closeups.jpg" alt="Old chart and actual surface contours on courtyard, workshop and registry"></a>',
           '<p>Full-resolution crops: '+' · '.join(f'{k.title()} <a href="crops/{k}/surface.png">surface</a> / <a href="crops/{k}/context.png">with context</a>' for k in FOCUS)+'</p>',
           '<h2>Which boundaries improved?</h2>',
           table(['Method','Matched / eligible','Mean border error','Correct surface pixels','Opening area covered','Both block counts correct'],
                 [[NAMES[m],f"{v['matched']} / {v['eligible']}",f"{v['edge_error_px']:.2f} px",
                   f"{v['assigned_fraction']:.2%}",f"{v['opening_fill']:.2%}",stats['block_counts']['all'][m]['correct']]
                  for m,v in stats['all'].items()]),
           '<h2>How much context does a wider crop show?</h2><p>Each count is a sample of four by four native pixels, summed across matched regions. The added margin shows real neighboring pixels. The ordinary Block Counter measures the owned surface.</p>']
    margins=read(out/'context-margins.json')['native_pixel_margins']
    parts.append(table(['Margin on each side','Missed target samples visible in RGB context','Other surface or background samples in context'],
                       [[k+' px',v['missed_target_visible_in_context'],v['other_surface_or_background_in_context']] for k,v in margins.items()]))
    context_counts=read(out/'count-context/scores.json')['all']
    caption={'tight':'Tight surface only','context8':'8 px with neighboring texture','context16':'16 px with neighboring texture',
             'padding8':'8 px, extra pixels hidden','padding16':'16 px, extra pixels hidden'}
    parts+=['<h2>Does the extra texture improve counting?</h2><p>In this diagnostic we allowed connected, similar-colored pixels outside the tight crop to help the counter. Holes and other detected surfaces stayed excluded. We also ran the expanded sampling charts with the extra RGB hidden to isolate the effect of seeing those neighbors.</p>',
            table(['Counter input','Both dimensions right','Wrong pair','No complete count','Both grid origins estimated'],
              [[caption.get(k,k),v['correct'],v['incorrect'],v['missing'],v['both_phases_returned_on_matches']] for k,v in context_counts.items()]),
            '<p><a href="count-context/cases.json">All context measurements</a> · <a href="count-context/manifest.json">Context settings and methods</a></p>']
    parts+=['<h2>All 56 original pictures</h2>']
    for row in results:
        case=lookup[row['key']]
        link=os.path.relpath(case.path('image'),out).replace(os.sep,'/')
        parts+=['<details><summary>'+escape(row['key'])+'</summary>',f'<p><a href="{escape(link)}">Original PNG</a></p>',
                table(['Surface Finder','Correct surfaces','Eligible surfaces','Correctly sized surfaces'],
                      [[NAMES[m],s['matched'],s['eligible'],row['block_counts'][m]['correct']] for m,s in row['scores'].items()]),'</details>']
    parts+=['<p><a href="manifest.json">Run manifest</a> · <a href="cases.json">Per-picture numbers</a> · <a href="selection.json">Development choice</a></p></html>']
    (out/'report.html').write_text('\n'.join(parts),encoding='utf8')


def replay_detection(run, out):
    from benchmark.perimeter_round import ARCHIVE, COUNT, groups
    from benchmark.surface_detection_sheet import diagnostics

    manifest=read(run/'manifest.json')
    selected=manifest['selection']['selected']
    cohorts,_,fixtures=groups()
    if fixtures!=manifest['fixtures']:
        raise ValueError('Saved fixture set differs from the run')
    def verify(path, expected):
        if digest(path)!=expected:
            raise ValueError(f'Saved input changed: {path}')
    verify(ARCHIVE,manifest['reference_sha256'])
    verify(run/'cases.json',manifest['artifacts']['cases.json'])
    verify(COUNT/'manifest.json',manifest['review_source_manifest_sha256'])
    verify(COUNT/'review-regions.json',read(COUNT/'manifest.json')['artifacts']['review-regions.json'])
    originals,frozen=[],None
    if selected=='regions-first':
        verify(FINDER/'manifest.json',manifest['baseline_manifest_sha256'])
        frozen=read(FINDER/'manifest.json')
        verify(FINDER/'cases.json',frozen['artifact_sha256']['cases.json'])
        originals=read(FINDER/'cases.json')
    lookup={f'{case.floor}/{case.id}':case for group in cohorts.values() for case in group}
    rows={row['key']:row for row in read(run/'cases.json')}
    examples,predictions=review_inputs(run,rows,lookup,originals,frozen,selected,manifest['artifacts'])
    out.mkdir(parents=True,exist_ok=False)
    diagnostics(out,examples,predictions,lookup,rows,selected)
    return out/'surface-detection.jpg'


if __name__=='__main__':
    import argparse
    from pathlib import Path
    from uuid import uuid4
    from benchmark.camera_round import ROOT
    from runtime.mode import image_only

    parser=argparse.ArgumentParser(description='Rebuild the surface-matching sheet from a saved perimeter run.')
    parser.add_argument('--run',type=Path,default=ROOT/'benchmark/reviews/perimeters-01')
    parser.add_argument('--out',type=Path,help='New directory for the sheet and per-surface pairings')
    args=parser.parse_args()
    out=args.out or ROOT/'runs'/f'surface-detection-{uuid4().hex[:12]}'
    with image_only():
        print(replay_detection(args.run,out).resolve())
