"""Draw the single-outline experiment from saved masks and paired depth decisions."""
import argparse
import gzip
import json
from pathlib import Path
from textwrap import wrap
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.camera_sheet import label
from benchmark.normal_round import groups
from benchmark.overhang_review import detail_box
from benchmark.recovery_overview import (COL, GAP, GOLD, LEFT, TOP, assessment, checked,
                                         delta_label, tile)
from benchmark.split_sheet import _detail
from benchmark.surface_detection_sheet import COLORS, INK, MUTED, PAPER, focus_box, paint
from benchmark.surface_reference import ARCHIVE, recorded
from engine.surface_perimeter import _native_alpha
from runtime.mode import image_only


REVIEWS = ROOT / 'benchmark/reviews'
METHODS = ('current', 'single-v2', 'single-pro')
NAMES = dict(zip(METHODS, ('Wall Recovery', 'Depth Anything V2', 'Depth Pro')))
INSIDE, OUTSIDE = '#20ddd1', '#e754ad'


def load_gzip(root, name, hashes):
    path = root / name
    checked(path, hashes[name])
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return json.load(f)


def visual_inputs(case, row, source, hashes):
    checked(case.path('image'), row['sha256'])
    parts = load_gzip(source, row['predictions'], hashes)
    with Image.open(case.path('image')) as img:
        photo = img.convert('RGB')
    _, ref = recorded(case)
    results = {}
    for method in METHODS:
        expected = set(row['scopes'][method]['matched'])
        results[method], native = assessment(case, ref, parts[method], expected)
    return photo, ref, native, results


def choose_reference(results, previous, opportunity):
    base, v2, pro = (results[m] for m in METHODS)
    eligible = [r for r in base['grading']['references'] if r['eligible']]
    def priority(r):
        rid = r['id']
        changed = rid in base['found'] ^ v2['found'] or rid in base['found'] ^ pro['found']
        return (rid in v2['found'] ^ pro['found'], changed, rid in opportunity,
                rid not in base['found'], rid == previous, r['pixels'], -rid)
    return max(eligible, key=priority)['id']


def color_predictions(photo, result):
    colors = {p['id']: COLORS['extra'] if p['status']=='extra' else
              COLORS['small_found'] if p['status']=='small_found' else COLORS['finder']
              for p in result['grading']['predictions']}
    return paint(photo, result['mapped'], colors)


def locate(img, box):
    marked = img.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle(box, outline=INK, width=5)
    draw.rectangle(box, outline='white', width=2)
    return marked


def main_column(put, row, view, case, source, hashes, opportunities):
    photo, ref, native, results = visual_inputs(case, row, source, hashes)
    rid = choose_reference(results, view['focus_reference'], opportunities)
    scene = tuple(view['scene_xyxy'])
    box = tuple(map(int, detail_box(native, rid, photo.size)))
    in_box = set(np.unique(native[box[1]:box[3], box[0]:box[2]]).tolist()) - {0}
    base = results['current']['found']
    eligible = {i for i,p in ref['patches'].items() if p['eligible']}
    put('source', photo, 'Original screenshot', f'{photo.width} × {photo.height} native pixels')
    put('scene', photo.crop(scene), f'{len(ref["patches"])} visible  /  {len(eligible)} scored',
        f'{len(ref["patches"])-len(eligible)} below the size cutoff')
    truth = paint(photo, native, {i:GOLD if i in eligible else COLORS['small'] for i in ref['patches']})
    put('truth', locate(truth, box).crop(scene), 'Recorded geometry', f'White box locates surface R{rid}')
    put('original-detail', photo.crop(box), 'Source pixels enlarged', f'Same crop in each row below')
    put('reference-detail', paint(photo, native, {rid:GOLD}).crop(box), f'Recorded surface R{rid}', 'Gold marks its visible footprint')
    details = {}
    for method, result in results.items():
        found = result['found']
        counts = result['grading']['counts']
        put(method, color_predictions(photo,result).crop(scene),
            f'{len(found)} found / {len(eligible)} scored  ·  {len(eligible-found)} missed',
            f'{counts["proposed"]} full-frame guesses  ·  {counts["unpaired"]} unpaired')
        grades = {r['id']:COLORS[r['status']] for r in result['grading']['references']}
        put('grade-'+method, paint(photo,native,grades).crop(scene),
            f'{len(found)} found  ·  {len(eligible-found)} missed', f'{counts["small_found"]} smaller matches')
        near = result['lookup'][rid]
        index = near['index']
        detail, visible = _detail(photo, result['parts'][index-1] if index else None, box)
        gained, lost = (found-base)&in_box, (base-found)&in_box
        state = 'Found' if rid in found else 'Missed'
        put('detail-'+method, detail, f'{state} R{rid}  ·  {near["overlap"]:.0%} overlap',
            'Five-model baseline' if method=='current' else delta_label(gained,lost))
        details[method] = dict(found=sorted(found), missed=sorted(eligible-found), counts=counts,
                               target=near, in_view=visible, gained_in_crop=sorted(gained), lost_in_crop=sorted(lost))
    return dict(frame=row['key'], sha256=row['sha256'], layout=row['layout'], scene_xyxy=scene,
                focus_reference=rid, crop_xyxy=box, methods=details)


def select_examples(rows, source, hashes):
    """Pick one shared candidate per building, preferring a different decision between models."""
    best = {}
    for row in rows:
        evidence = load_gzip(source,row['boundary_evidence'],hashes)
        v2, pro = (evidence['events'][m] for m in METHODS[1:])
        for candidate,a,b in zip(evidence['inventory'],v2,pro,strict=True):
            key = (candidate['parent'], candidate['proposal'])
            if any((e['parent'],e['proposal'])!=key for e in (a,b)):
                raise ValueError('Methods tested different MobileSAM candidates')
            accepted = int(a['accepted'])+int(b['accepted'])
            priority = (accepted==1,accepted==2,len(candidate['samples']))
            winner = best.get(row['layout'])
            if winner is None or priority>winner[0]:
                best[row['layout']] = priority, (row,candidate,a,b)
    return {name:record for name,(_,record) in best.items()}


def decision_column(put, example, case, source, hashes, model_files):
    row, candidate, a, b = example
    photo, ref, native, results = visual_inputs(case,row,source,hashes)
    base = results['current']
    part = next(p for p in base['parts'] if p['id']==candidate['parent'])
    alpha,(x,y,_,_) = _native_alpha(part['support'],photo.size)
    parent_img = Image.new('L',photo.size)
    parent_img.paste(alpha,(x,y))
    parent = np.asarray(parent_img)>0
    key = 'proposals/'+row['sha256']+'.npz'
    archive = REVIEWS/'mobile-regions-01'/key
    checked(archive,model_files[key])
    with np.load(archive,allow_pickle=False) as data:
        mask = np.unpackbits(data['packed'][candidate['proposal']],axis=1,count=int(data['shape'][2])).astype(bool)
    sam = np.asarray(Image.fromarray(mask).resize(photo.size, Image.Resampling.NEAREST))
    inside = parent & sam
    box = tuple(map(int,focus_box(inside,1,photo.size)))
    pts = np.asarray(candidate['samples'],dtype=int)
    height,width = mask.shape
    if (photo.height,photo.width)!=(4*height,4*width) or pts.ndim!=2 or pts.shape[1]!=8:
        raise ValueError('Boundary samples do not match the frozen four-pixel lattice')
    short = row['key'].split(row['layout']+'-')[-1].replace('-',' ')
    put('locate',locate(photo,box),short,row['key'].split('/')[0].replace('_',' ')+' floor')
    put('zoom',photo.crop(box),'Original pixels','Same crop throughout this example')
    truth = paint(photo,native,{i:GOLD if p['eligible'] else COLORS['small'] for i,p in ref['patches'].items()})
    put('geometry',truth.crop(box),'Recorded surface boundaries','Gray marks below-cutoff surfaces')
    put('existing',paint(photo,parent.astype(np.uint8),{1:COLORS['finder']}).crop(box),
        'Current wall before voting','Cyan marks the parent surface')
    prompt = paint(photo,inside.astype(np.uint8),{1:INSIDE})
    draw = ImageDraw.Draw(prompt)
    for sample in pts[::max(1,(len(pts)+29)//30)]:
        iy,ix,oy,ox = (sample[2:6]*4+2).tolist()
        draw.line((ix,iy,ox,oy),fill=INK,width=2)
        for x,y,color in ((ix,iy,INSIDE),(ox,oy,OUTSIDE)):
            draw.ellipse((x-2,y-2,x+2,y+2),fill=color,outline=INK)
    put('candidate',prompt.crop(box),f'MobileSAM shape {candidate["proposal"]}',
        f'{len(pts)} depth comparisons along its border')
    for method, event in zip(METHODS[1:],(a,b)):
        result = results[method]
        step = event['median_step']
        relative = 'piece appears nearer' if step>0 else 'piece appears farther' if step<0 else 'level depth'
        decision = 'Used' if event['accepted'] else 'Skipped'
        put('exam-'+method,color_predictions(photo,result).crop(box),
            decision+' split  ·  '+relative,
            f'{event["sign_agreement"]:.0%} of pairs agree  ·  {result["grading"]["counts"]["found"]} surfaces found')
    return dict(frame=row['key'],sha256=row['sha256'],proposal=candidate['proposal'],parent=candidate['parent'],
                crop_xyxy=box,sample_count=len(pts),events={METHODS[1]:a,METHODS[2]:b})


def stages():
    rows=[('source','Source screenshot','Original','The full input to every method.',227),
          ('scene','Source building','Original','One building per column. All overviews use this view.',255)]
    rows.extend((m,NAMES[m],'Prediction','Complete proposed surfaces.' if m=='current' else
                 'Same MobileSAM outlines. Only the depth vote differs between challengers.',255) for m in METHODS)
    rows.append(('truth','Actual surfaces','Reference','Gold outlines give the recorded shapes and gray shows smaller surfaces.',255))
    rows.extend(('grade-'+m,NAMES[m],'Scored reference','Green marks found surfaces and amber marks misses.',255) for m in METHODS)
    rows.extend([('original-detail','Original pixels','Close-up','Follow the same spot across all three methods.',225),
                 ('reference-detail','Recorded target','Reference close-up','One scored visible face.',225)])
    rows.extend(('detail-'+m,NAMES[m],'Prediction close-up','Cyan marks the best corresponding guess. Overlap uses the entire surface.',225) for m in METHODS)
    rows.extend([('locate','Source location','Boundary test','One candidate per building, favoring a decision the two models disagreed on.',225),
                 ('zoom','Original pixels','Boundary test','The same enlargement used for the votes below.',225),
                 ('geometry','Actual surfaces','Reference','Gold marks the recorded boundaries.',225),
                 ('existing','Starting wall','Prediction','The five-model result before the single-outline test.',225),
                 ('candidate','Candidate and samples','Model input','Cyan is the proposed piece. Dots show sample pairs inside and outside its boundary.',225)])
    rows.extend(('exam-'+m,NAMES[m],'Actual decision','The final predicted regions in this crop.',225) for m in METHODS[1:])
    return rows


@image_only()
def render(out, source):
    source,out=Path(source),Path(out)
    manifest=read(source/'manifest.json')
    hashes=manifest['artifacts']
    checked(source/'cases.json',hashes['cases.json'])
    rows=read(source/'cases.json')
    by_key={r['key']:r for r in rows}
    cohorts,_,fixtures=groups()
    cases={f'{c.floor}/{c.id}':c for group in cohorts.values() for c in group}
    checked(ARCHIVE,manifest['reference_sha256'])
    if fixtures!=manifest['fixtures'] or len(rows)!=56 or set(cases)!=set(by_key):
        raise ValueError('The saved suite has changed')
    prior=REVIEWS/'recovery-summary-01'
    prior_hash=read(prior/'manifest.json')['artifacts']['cases.json']
    checked(prior/'cases.json',prior_hash)
    control={r['key']:r for r in read(prior/'cases.json')}
    for row in rows:
        previous=control[row['key']]
        if (row['sha256']!=previous['sha256'] or
            row['scopes']['current']['matched']!=previous['scopes']['combined']['matched']):
            raise ValueError(f"Control differs on {row['key']}")
    old=REVIEWS/'recovery-01'
    roster_hash=read(old/'manifest.json')['artifacts']['sheet-views.json']
    checked(old/'sheet-views.json',roster_hash)
    roster=read(old/'sheet-views.json')['views']
    examples=select_examples(rows,source,hashes)
    model_files=read(REVIEWS/'mobile-regions-01/manifest.json')['artifacts']
    totals={m:dict(found=sum(r['scores'][m]['matched'] for r in rows),
                   scored=sum(r['scores'][m]['eligible'] for r in rows)) for m in METHODS}
    specs=stages()
    width=LEFT+len(roster)*(COL+GAP)+GAP
    height=TOP+sum(h+82 for _,_,_,_,h in specs)+10
    page=Image.new('RGB',(width,height),PAPER)
    draw=ImageDraw.Draw(page)
    label(draw,(16,17),'Can one outline separate a merged wall?',41,bold=True)
    label(draw,(18,76),'Same 56 pictures, same starting walls, same candidate outlines.  /  Java 26.3 at clear noon.',21)
    legend=((COLORS['finder'],'Predicted surface'),(COLORS['extra'],'Extra guess'),(GOLD,'Recorded surface'),
            (COLORS['found'],'Found'),(COLORS['missed'],'Missed'),(COLORS['small_found'],'Small match'))
    for i,(color,title) in enumerate(legend):
        x=20+i*257
        draw.rectangle((x,122,x+18,140),fill=color,outline=MUTED)
        label(draw,(x+27,121),title,18)
    tops={}
    y=TOP
    for key,title,kind,note,h in specs:
        tops[key]=y,h
        draw.line((14,y,width-14,y),fill='#a9bdb5',width=2)
        label(draw,(15,y+17),kind.upper(),15,fill=MUTED,bold=True)
        for j,line in enumerate(wrap(title,16)):
            label(draw,(15,y+48+j*28),line,22,bold=True)
        for j,line in enumerate(wrap(note,23)):
            label(draw,(15,y+117+j*22),line,16,fill=MUTED)
        if key in METHODS:
            label(draw,(15,y+252),f'{totals[key]["found"]} / {totals[key]["scored"]} in 56 pictures',17,bold=True)
        y+=h+82
    summaries=[]
    for col,view in enumerate(roster):
        x=LEFT+col*(COL+GAP)
        layout,key=view['layout'],view['frame']
        label(draw,(x,TOP-82),layout.title(),27,bold=True)
        variant=key.split(layout+'-')[-1].replace('-',' ')+' / '+key.split('/')[0].replace('_',' ')
        label(draw,(x,TOP-45),variant,16,fill=MUTED)
        def put(key,img,first,second):
            y,h=tops[key]
            tile(page,x,y,h,img,first,second)
        summary=main_column(put,by_key[key],view,cases[key],source,hashes,control[key]['remaining']['in_filtered_sam'])
        if layout in examples:
            example=examples[layout]
            summary['boundary_example']=decision_column(put,example,cases[example[0]['key']],source,hashes,model_files)
        else:
            label(draw,(x,tops['locate'][0]+60),'No eligible single outline',19,bold=True)
            label(draw,(x,tops['locate'][0]+95),'Across this building’s saved views.',16,fill=MUTED)
            summary['boundary_example']=None
        summaries.append(summary)
    out.mkdir(parents=True,exist_ok=True)
    path=out/'comparison.jpg'
    page.save(path,quality=93,subsampling=0,optimize=True)
    save(out/'sheet-views.json',dict(source=source.name,cases_sha256=hashes['cases.json'],
         control_cases_sha256=prior_hash,view_roster_sha256=roster_hash,reference_sha256=digest(ARCHIVE),
         suite_counts=totals,views=summaries,image=dict(file=path.name,sha256=digest(path),width=width,height=height)))
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=REVIEWS/'single-wall-depth-01')
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'single-wall-sheet-{uuid4().hex[:12]}')
    args=parser.parse_args()
    print(render(args.out,args.source))
