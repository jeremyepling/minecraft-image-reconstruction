"""Create a reading view and one side-by-side sheet using the established eight camera angles."""
from collections import Counter
from html import escape
import os
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from benchmark.camera_round import read, save
from benchmark.normal_building_sheet import selected_views
from benchmark.surface_reference import recorded
from benchmark.surface_sheet import scene_box,text
from engine.surface_perimeter import draw_outline
from engine.surface_tiles import tile_sample


NAMES={'legacy-repeat':'B’s texture counter','masked-repeat':'Masked repeat counter','atlas-repeat':'Atlas-assisted counter'}
PAPER='#f8f7f2'
INK='#183639'


def pair(values):
    if values is None:
        return 'unmatched'
    return ' × '.join('?' if n is None else str(n) for n in values)


def fit_panel(canvas,im,x,y,w,h):
    scaled=ImageOps.contain(im,(w,h),Image.Resampling.NEAREST)
    canvas.paste(scaled,(x+(w-scaled.width)//2,y+(h-scaled.height)//2))


def sheet(out,results,inputs,lookup,atlas,frozen,selected):
    from benchmark.count_round import cached_regions
    from engine.surface_sample import sample_surface
    by_key={r['key']:r for r in results}
    material_file=out/'materials.json'
    materials={r['key']:r for r in read(material_file)['pictures']} if material_file.exists() else {}
    displayed=[r['key'] for r in selected_views() if r['key'] in by_key]
    canvas=Image.new('RGB',(1660,136+len(displayed)*232),PAPER)
    pen=ImageDraw.Draw(canvas)
    text(pen,25,16,'Can we count the blocks in a visible surface?',36,True)
    text(pen,25,66,'Cyan follows the detected perimeter. Dark pixels are excluded from the sample. Counts read across × down. ? means unknown.',21)
    for x,title in [(25,'Source and surface'),(445,'Straightened surface'),(817,'Whole surface in blocks'),(1241,'One block and atlas match')]:
        text(pen,x,105,title,22,True)
    regions=[]
    examples=[]
    for j,key in enumerate(displayed):
        y=145+j*232
        row=by_key[key]
        case=lookup[key]
        with Image.open(case.path('image')) as src:
            image=src.convert('RGB')
        _,ref=recorded(case)
        parts=cached_regions(inputs[key],frozen)
        records={r['id']:r for r in row['surfaces']}
        # Always choose the largest matched, charted surface by image area, independently of the counter scores.
        eligible=[p for p in parts if records[p['id']]['expected'] is not None and records[p['id']]['quad'] is not None]
        text(pen,25,y,row['layout'].title(),20,True)
        if not eligible:
            text(pen,445,y+40,'No matched surface with a sampling chart',18)
            continue
        part=max(eligible,key=lambda p:p['support_area'])
        rec=records[part['id']]
        sample=sample_surface(image,part)
        sw,sh=part['support']['size']
        raw=np.zeros((sh,sw),np.uint8)
        for v,a,b in part['support']['runs']:
            raw[v,a:b]=100
        surface=image.copy()
        surface.paste('#2ce4ed',(0,0,*image.size),Image.fromarray(raw).resize(image.size,Image.Resampling.NEAREST))
        q=np.asarray(rec['quad'])
        draw_outline(surface,part,color='#49ffff',width=3)
        box=scene_box(ref,image)
        examples.append(dict(layout=row['layout'],key=key,image=image,part=part,box=box,margin=8,
                             source=case.path('image').name,expected=rec['expected'],finder='regions-first',counter=selected))
        fit_panel(canvas,surface.crop(box),25,y+28,385,180)
        rgb=np.asarray(sample['rgb']).copy()
        rgb[~sample['mask']]=[42,48,51]
        fit_panel(canvas,Image.fromarray(rgb),445,y+28,338,180)
        text(pen,817,y+34,'Recorded building   '+pair(rec['expected']),20,True)
        for n,(method,r) in enumerate(rec['measures'].items()):
            text(pen,817,y+72+32*n,NAMES[method]+'   '+pair(r['dimensions']['blocks']),17)
        data=materials.get(key,{})
        tiles=[v for v in data.get('predicted',[]) if v['proposal']==part['id'] and 'truth' in v]
        if tiles:
            tile=tiles[0]
            label='Predicted corners'
        else:
            known=[v for v in data.get('known',[]) if v['fixed_sample']]
            tile=known[0] if known else None
            label='Known corners • same picture'
        if tile is not None:
            text(pen,1241,y+26,label,17,True)
            got=Image.fromarray(tile_sample(image,tile['quad'])['rgb'])
            fit_panel(canvas,got,1241,y+54,96,83)
            tex=tile.get('texture')
            if tex in atlas.labels:
                art=Image.fromarray(np.asarray(atlas.tiles[atlas.labels.index(tex)]).astype(np.uint8))
                fit_panel(canvas,art,1360,y+54,96,83)
            text(pen,1241,y+146,'Picture                    Atlas',16)
            text(pen,1241,y+171,'Guess  '+str(tile['block_id']),16)
            text(pen,1241,y+195,'Actual  '+str(tile['truth']),16)
        regions.append(dict(frame=key,stage='block-count',proposal=part['id'],chart_xy=q.tolist(),
                            region_xyxy=list(map(int,box)),sampling_selection='largest matched charted surface'))
        pen.line((25,y+222,1634,y+222),fill='#d4d9d5',width=1)
    canvas.save(out/'counter-comparison.jpg',quality=91,subsampling=0)
    from benchmark.surface_pipeline_sheet import stage_sheet
    stage_sheet(out,examples,atlas)
    save(out/'review-regions.json',regions)


def table(headers, rows):
    head=''.join('<th>'+escape(str(x))+'</th>' for x in headers)
    body=''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)
    return '<table><tr>'+head+'</tr>'+body+'</table>'


def sampling_closeup(out, measurements, lookup, atlas):
    pairs=[]
    for picture in measurements['pictures']:
        for known in picture['known']:
            if known['paired_tile'] is None:
                continue
            prediction=picture['predicted'][known['paired_tile']]
            if (known['block_id']==known['truth'] and prediction['block_id']!=known['truth']
                and known['texture'] in atlas.labels and prediction['texture'] in atlas.labels):
                pairs.append((picture['key'],known,prediction))
    if not pairs:
        return
    # Pick the middle overlap from the most frequent confusion, only for this review illustration.
    def confusion(row):
        return (row[1]['truth'],row[2]['block_id'])
    common=Counter(map(confusion,pairs)).most_common(1)[0][0]
    examples=sorted((row for row in pairs if confusion(row)==common),key=lambda row:row[2]['iou'])
    key,known,pred=examples[len(examples)//2]
    with Image.open(lookup[key].path('image')) as src:
        image=src.convert('RGB')
    canvas=Image.new('RGB',(1120,375),PAPER)
    draw=ImageDraw.Draw(canvas)
    text(draw,24,20,'How a shifted crop changes the block ID',30,True)
    text(draw,24,67,key+'   •   '+pred['proposal'],18)
    overlay=image.copy()
    pixels=np.vstack([known['quad'],pred['quad']])
    lo=np.maximum([0,0],np.floor(pixels.min(axis=0))-14).astype(int)
    hi=np.minimum(image.size,np.ceil(pixels.max(axis=0))+14).astype(int)
    for item,color in [(known,'#ffd055'),(pred,'#00f2ff')]:
        q=list(map(tuple,item['quad']))
        ImageDraw.Draw(overlay).line(q+q[:1],fill=color,width=1)
    fit_panel(canvas,overlay.crop(tuple([*lo,*hi])),24,114,225,163)
    text(draw,24,293,'Gold   recorded corners',16)
    text(draw,24,319,'Cyan   predicted corners',16)
    for x,item,title in [(274,known,'Recorded corners'),(699,pred,'Predicted corners')]:
        text(draw,x,104,title,19,True)
        crop=Image.fromarray(tile_sample(image,item['quad'])['rgb'])
        fit_panel(canvas,crop,x,143,128,124)
        art=Image.fromarray(np.asarray(atlas.tiles[atlas.labels.index(item['texture'])]).astype(np.uint8))
        fit_panel(canvas,art,x+165,143,128,124)
        text(draw,x,278,'Sample',16)
        text(draw,x+165,278,'Top atlas match',16)
        text(draw,x,316,'Guess   '+item['block_id'].replace('_',' '),19,True)
    canvas.save(out/'sampling-closeup.png')
    save(out/'sampling-closeup.json',dict(frame=key,stage='material',proposal=pred['proposal'],xyz=pred['xyz'],
         known_corners=known['quad'],predicted_corners=pred['quad'],crop_overlap=pred['iou'],
         case_selection='Median-overlap example of the most frequent wrong ID corrected by recorded corners.'))


def review(out,results,rows,lookup,atlas,frozen,selected):
    from benchmark.count_round import METHODS
    inputs={r['key']:r for r in rows}
    sheet(out,results,inputs,lookup,atlas,frozen,selected)
    scores=read(out/'scores.json')
    def summary_table(s,mode):
        items=[]
        for name in METHODS:
            v=s[name][mode]
            items.append([NAMES[name],v['correct'],v['incorrect'],v['missing'],v['eligible'],
                          v['across_right'],v['down_right'],v['unmatched_returned']])
        return table(['Counter','Both right','Wrong pair','No pair','Eligible','Across right','Down right','Other proposals with a pair'],items)
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>How many blocks across and down?</title><style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;background:#faf9f4;color:#163238;font:16px/1.5 system-ui}table{border-collapse:collapse}td,th{padding:.35rem .65rem;text-align:left;border-bottom:1px solid #cbd1cd}details{margin:.6rem 0}summary{cursor:pointer}a{color:#126574}h2{margin-top:2rem}</style>',
        '<h1>How many blocks across and down?</h1><p><a href="comparison.jpg">Eight-building pipeline sheet</a> · <a href="counter-comparison.jpg">Three counters side by side</a> · <a href="manifest.json">Run manifest</a> · <a href="cases.json">All surface measurements</a></p>',
        '<p>Each counter received the same Surface Finder masks. A correct pair gives the recorded number of visible block columns and rows. A mask can include a doorway; the dimensions describe its outside span. Cropped surfaces can include partial edge blocks. Across and down follow the two surface directions as seen in the picture. All views count separately.</p>',
        '<h2>Whole visible surfaces</h2>',summary_table(scores['all'],'surface'),
        '<p>The first three result columns add to the eligible surface count. No pair includes missed surfaces and counters that could only read one direction. A wrong pair has a count on both axes with at least one error. Other proposals do not match a scored surface.</p>',
        '<h2>Across the eight buildings</h2><p>Each row sums the views of one building. Seeing a wall in two screenshots gives two opportunities to measure it.</p>',
        table(['Building','Views','Eligible surface appearances',*NAMES.values()],
              [[layout.title(),sum(r['layout']==layout for r in results),arms[selected]['surface']['eligible'],
                *[arms[m]['surface']['correct'] for m in METHODS]] for layout,arms in scores['layout'].items()]),
        '<h2>When given a correct rectangle</h2><p>These diagnostic patches cover fully visible cells on a known plane, with edges deliberately inset by a fraction of a block. Both predicted spans must be within half a block.</p>',
        summary_table(scores['all'],'known_rectangles')]
    if (out/'extent-diagnostic.json').exists():
        audit=read(out/'extent-diagnostic.json')['all']
        parts+=[f"<p>With the recorded surface and camera, rounding the visible span gives the recorded count of touched columns and rows on {audit['equal']} of {audit['eligible']} surfaces. Partial blocks, occlusion and the four-pixel reference raster affect this difference. "
                "<a href='extent-diagnostic.json'>Reference-only diagnostic</a></p>"]
    for name,tables in scores['cohort'].items():
        parts+=['<details><summary>'+escape(name.replace('_',' '))+'</summary>',summary_table(tables,'surface'),'</details>']
    material_path=out/'materials.json'
    if material_path.exists():
        measurements=read(material_path)
        d=measurements['all']
        paired={item['truth'] for frame in measurements['pictures'] for item in frame['predicted'] if 'truth' in item}
        sampling_closeup(out,measurements,lookup,atlas)
        parts+=['<h2>Material Matcher</h2><p>The same vanilla atlas matcher saw each crop. The fixed sample uses up to 12 known faces per picture. The paired rows use the same physical blocks twice.</p>',
           table(['Corners used','Right ID first','Right ID in top five','Faces'],
                 [['Known, fixed sample',d['fixed_top1'],d['fixed_top5'],d['fixed_faces']],
                  ['Known, on recovered faces',d['paired_known_top1'],d['paired_known_top5'],d['matched']],
                  ['Predicted, on recovered faces',d['predicted_top1'],d['predicted_top5'],d['matched']]]),
            f"<p>The chosen counter proposed {d['proposed']} whole block footprints. {d['matched']} matched out of {d['eligible_faces']} eligible visible faces. The paired sample covers {escape(', '.join(sorted(paired)) or 'no materials')}. Sampling requires a measured grid origin and a complete cell inside the straightened mask. The exact-ID test counts visually identical block IDs separately. <a href='materials.json'>Every sampled face and guess</a></p>"]
        if (out/'sampling-closeup.png').exists():
            parts+=["<a href='sampling-closeup.png'><img src='sampling-closeup.png' style='max-width:100%' alt='The same block sampled with recorded and predicted corners'></a>"]
    geometry=out/'placed-geometry.json'
    if geometry.exists():
        placed=read(geometry)
        parts+=['<h2>Did it put more blocks in the right place?</h2><p>Each counter used the same camera estimate, surface masks and 3D exporter. We read the coordinates back from each smooth-stone datapack and aligned the arbitrary origin for scoring. Minecraft stayed closed.</p>',
                table(['Views','Counter','Correct coordinates','Extra coordinates','Visible coordinates','Pictures with blocks'],
                      [[group.replace('_',' '),NAMES[method],a['score']['correct_visible'],a['score']['extra'],
                        a['score']['visible_truth'],a['nonempty_pictures']]
                       for group,arms in [('All',placed['all']),*placed['cohorts'].items()] for method,a in arms.items()]),
                f"<p>The camera established block scale on {placed['camera_status'].get('supported',0)} of {len(results)} images. "
                "<a href='placed-geometry.json'>Per-picture export results</a> · <a href='geometry-packs.zip'>Minecraft datapacks</a></p>"]
    parts+=['<h2>Scores by picture</h2>']
    for row in results:
        link=Path(os.path.relpath(lookup[row['key']].path('image'),out)).as_posix()
        records=[[r['id'],pair(r['expected']),*[pair(r['measures'][name]['dimensions']['blocks']) for name in METHODS]]
                 for r in row['surfaces']]
        matched=sum(r['expected'] is not None for r in row['surfaces'])
        parts+=['<details><summary>'+escape(row['key'])+'</summary>',
                f"<p><a href='{link}'>Original screenshot</a>. Surface Finder recovered {matched} of {row['eligible']} eligible surfaces.</p>",
                table(['Surface','Recorded dimensions',*NAMES.values()],records),'</details>']
    (out/'report.html').write_text('\n'.join(parts)+'</html>',encoding='utf8')
