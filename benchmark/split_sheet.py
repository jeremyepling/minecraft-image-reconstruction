"""Compare two independently refined surface maps on the eight fixed review views."""
import gzip
import json
import textwrap
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.camera_sheet import font, label
from benchmark.mask_metrics import footprint, measure
from benchmark.overhang_review import detail_box
from benchmark.surface_detection_maps import detection_maps
from benchmark.surface_detection_sheet import BG, COLORS, INK, MUTED, PAPER, paint
from benchmark.surface_reference import recorded
from engine.surface_perimeter import _native_alpha


NAMES = {'current': 'Previous finder', 'wall-depth-anything': 'Wall Separator · Depth Anything V2',
         'wall-moge': 'Wall Separator · MoGe', 'strip-deeplsd': 'Underside · DeepLSD',
         'strip-scalelsd': 'Underside · ScaleLSD', 'strip-both': 'Underside · DeepLSD + ScaleLSD',
         'strip-mlsd': 'Underside · M-LSD Tiny'}
LEFT, COL, GAP, HEADER = 238, 384, 14, 263


def _tile(page, x, top, height, photo, *captions):
    y = top + 14
    draw = ImageDraw.Draw(page)
    draw.rectangle((x, y, x+COL-1, y+height-1), fill=BG)
    fitted = ImageOps.contain(photo, (COL, height), Image.Resampling.LANCZOS)
    page.paste(fitted, (x+(COL-fitted.width)//2, y+(height-fitted.height)//2))
    for i, line in enumerate(captions):
        label(draw, (x, y+height+9+24*i), line, 17, fill=INK if i==0 else MUTED, bold=i==0)


def _lookup_scores(parts, reference, refs):
    """Use the assigned guess when matched, otherwise the actual best whole-mask overlap."""
    ids = reference['id_map']
    bins = max(reference['patches'])+1
    areas = np.bincount(ids.ravel(), minlength=bins)
    table = []
    for part in parts:
        mask = footprint(part, ids.shape, reference['image_size'], use_mask=True)
        hits = np.bincount(ids[mask], minlength=bins)
        table.append(hits/np.maximum(mask.sum()+areas-hits, 1))
    ious = np.asarray(table).reshape(-1, bins)
    result = {}
    for r in refs:
        if not r['eligible']:
            continue
        rid = r['id']
        pi = r['prediction']
        if pi is None and len(ious) and ious[:,rid].max()>0:
            pi = int(ious[:,rid].argmax())+1
        result[rid] = dict(found=r['status']=='found', overlap=float(ious[pi-1,rid]) if pi else 0.,
                           proposal=parts[pi-1]['id'] if pi else None, index=pi)
    return result


def _choose_focus(scored, reference, fallback):
    methods = list(scored)
    current = scored['current']
    def priority(rid):
        original = current[rid]
        others = [scored[m][rid] for m in methods[1:]]
        changed = any(r['found']!=original['found'] for r in others)
        movement = max(abs(r['overlap']-original['overlap']) for r in others)
        if not changed and movement<.10:
            return (0, rid==fallback, 0, 0, -rid)
        underside = reference['patches'][rid]['normal'][1] == -1
        return (2 if changed else 1, underside, movement, reference['patches'][rid]['support_px'], -rid)
    rid=max(current, key=priority)
    reason={0:'Previous review focus', 1:'Outline changed', 2:'Match changed'}[priority(rid)[0]]
    return rid, reason


def _mark_grades(rgb, truth, report, changed):
    colors = {r['id']: COLORS[r['status']] for r in report['references']}
    graded = paint(rgb, truth, colors)
    if changed:
        # Keep the status fill and put a white rim just outside each changed surface, including holes.
        bits = np.isin(truth, list(changed))
        outer = np.asarray(Image.fromarray(bits).filter(ImageFilter.MaxFilter(5))) & ~bits
        graded.paste('white', mask=Image.fromarray(outer))
    return graded


def _detail(rgb, part, box):
    crop = rgb.crop(box)
    if part is None:
        return crop, False
    result = _native_alpha(part['support'], rgb.size)
    if result is None:
        return crop, False
    alpha, (x,y,_,_) = result
    mask = Image.new('L', crop.size)
    mask.paste(alpha, (x-box[0], y-box[1]))
    return paint(crop, (np.asarray(mask)>0).astype(np.uint8), {1:COLORS['finder']}), bool(mask.getbbox())


def _underside_selection(source, cases, lookup, strip):
    """One illustration per building. True underside identities enter only after prediction."""
    choices, totals = {}, {}
    for row in cases:
        key, layout = row['key'], row['layout']
        scopes = row['scopes']
        base, new = scopes['current']['undersides'], scopes[strip]['undersides']
        n = scopes['current']['underside_total']
        totals.setdefault(layout, Counter()).update(eligible=n,current=len(base),challenger=len(new),
                                                    gained=len(set(new)-set(base)),lost=len(set(base)-set(new)))
        if not n:
            continue
        _, ref = recorded(lookup[key])
        ids = [i for i,p in ref['patches'].items() if p['eligible'] and p['normal'][1]==-1]
        if len(ids)!=n:
            raise ValueError(f'Underside count changed for {key}')
        with gzip.open(source/row['predictions'], 'rt', encoding='utf8') as f:
            parts = json.load(f)
        checks = {}
        for m in ('current',strip):
            _, pairs = measure(parts[m],ref,use_mask=True)
            assigned = {rid:i+1 for i,rid in pairs}
            if set(ids)&set(assigned)!=set(scopes[m]['undersides']):
                raise ValueError(f'Underside matches changed for {key}/{m}')
            refs = [dict(id=i,eligible=True,prediction=assigned.get(i),status='found' if i in assigned else 'missed') for i in ids]
            checks[m] = _lookup_scores(parts[m],ref,refs)
        for rid in ids:
            before, after = checks['current'][rid], checks[strip][rid]
            if after['found'] and not before['found']:
                priority, role = (2, after['overlap']-before['overlap']), 'Recovered underside'
            elif not after['found']:
                priority, role = (1, -after['overlap']), 'Lost underside' if before['found'] else 'Still missed'
            else:
                priority, role = (0, after['overlap']), 'Already found'
            priority += (ref['patches'][rid]['support_px'],key,-rid)
            if layout not in choices or priority>choices[layout][0]:
                choices[layout] = (priority, dict(frame=key,reference=rid,role=role,sha256=row['sha256'],
                                                 methods={m:c[rid] for m,c in checks.items()}))
    return {layout:value for layout,(_,value) in choices.items()},totals


def _low_angle_section(page, top, source, cases, lookup, sample, strip):
    choices, totals = _underside_selection(source,cases,lookup,strip)
    suite = Counter()
    for counts in totals.values():
        suite.update(counts)
    rows = {r['key']:r for r in cases}
    draw = ImageDraw.Draw(page)
    draw.rectangle((0,top,page.width,top+105),fill='#e2ebe4')
    label(draw,(18,top+16),'Where did the underside test help?',34,bold=True)
    label(draw,(19,top+65),
          f"Across {len(cases)} suite pictures the finder went from {suite['current']} to {suite['challenger']} of {suite['eligible']} scored undersides. "
          "The low-angle examples below are chosen after scoring. The totals above cover only the original eight views.",20)
    start = top+210
    yvals = [start+285*i for i in range(5)]
    for y,title,note in zip(yvals,('Source location','Original pixels','Recorded underside','Previous finder','Underside Finder'),
                            ('White box locates the crop.', 'Same crop for all four detail rows.',
                             'Recorded geometry for scoring.', 'Original predicted outline.', NAMES[strip])):
        draw.line((15,y,page.width-15,y),fill='#bdcdc7',width=2)
        label(draw,(15,y+23),title,21,bold=True)
        for j,line in enumerate(textwrap.wrap(note,width=23)):
            label(draw,(15,y+60+j*23),line,16,fill=MUTED)
    views = []
    for col, view in enumerate(sample):
        layout=view['layout']
        x=LEFT+col*(COL+GAP)
        count=totals[layout]
        label(draw,(x,top+127),layout.title(),25,bold=True)
        label(draw,(x,top+167),f"{count['current']} → {count['challenger']} of {count['eligible']} scored undersides",18)
        if layout not in choices:
            draw.rectangle((x,start+14,x+COL-1,start+211),fill=BG)
            label(draw,(x+14,start+75),'No scored underside',21,bold=True)
            label(draw,(x+14,start+112),'No miss is counted for this building.',17)
            views.append(dict(layout=layout,frame=None,reference=None,role='No scored underside',suite_counts=count))
            continue
        item=choices[layout]
        key,rid=item['frame'],item['reference']
        case,row=lookup[key],rows[key]
        if digest(case.path('image'))!=item['sha256']:
            raise ValueError(f'Changed low-angle screenshot for {key}')
        with Image.open(case.path('image')) as opened:
            rgb=opened.convert('RGB')
        with gzip.open(source/row['predictions'],'rt', encoding='utf8') as stream:
            parts=json.load(stream)
        native,_,report=detection_maps(case,parts[strip],row['scores'][strip])
        ref=next(r for r in report['references'] if r['id']==rid)
        box=detail_box(native,rid,rgb.size)
        yy,xx=np.where(native>0)
        scene=[max(0,int(xx.min())-30),max(0,int(yy.min())-30),
               min(rgb.width,int(xx.max())+31),min(rgb.height,int(yy.max())+31)]
        overview=rgb.copy()
        locator=ImageDraw.Draw(overview)
        locator.rectangle(box,outline=INK,width=5)
        locator.rectangle(box,outline='white',width=2)
        short_key=key.split('/')[-1].split(layout+'-')[-1].replace('-',' ')
        height=200
        _tile(page,x,yvals[0],height,overview.crop(scene),short_key,key.split('/')[0].replace('_',' ')+' floor')
        _tile(page,x,yvals[1],height,rgb.crop(box),item['role'],f'One underside in {layout.title()}')
        _tile(page,x,yvals[2],height,paint(rgb,native,{rid:COLORS[ref['status']]}).crop(box),
              f"R{rid}   ·   {ref['pixels']:,} visible pixels",'Full-surface overlap is scored below.')
        for j,m in enumerate(('current',strip)):
            chosen=item['methods'][m]
            pi=chosen['index']
            crop,in_box=_detail(rgb,parts[m][pi-1] if pi else None,box)
            status='Found' if chosen['found'] else 'Missed'
            if m==strip and item['role']=='Recovered underside':
                status='New match'
            info=(f'P{pi} matches R{rid}.' if chosen['found'] else f'P{pi} is the closest guess.' if pi and in_box
                  else 'Closest guess falls outside this crop.' if pi else 'No guess touches this surface.')
            _tile(page,x,yvals[3+j],height,crop,f"{status}   ·   {chosen['overlap']:.1%} whole-surface overlap",info)
        views.append(dict(**item, layout=layout, suite_counts=count, focus_native_xyxy=list(box), scene_xyxy=scene))
    return dict(methods=['current',strip], suite_counts=suite, views=views, suite_frames=len(cases),
                selection='Per building, prefer a newly found underside with the largest overlap improvement, then the least-overlapping miss, then the strongest retained match. No scored underside is a separate status.')


def render(out, cases, lookup, selected, *, source=None):
    source = out if source is None else source
    methods = ('current', selected['wall'], selected['strip'])
    recommended = selected['recommended']
    short = dict(zip(methods, ('Previous finder', 'Wall Separator', 'Underside Finder')))
    runs = ROOT/'benchmark/reviews'
    sample = read(runs/'perimeters-01/surface-detection.json')['views']
    previous = {v['frame']:v for v in read(runs/'overhang-01/focus.json')['views']}
    rows = {r['key']:r for r in cases}
    # Heights reserve two short captions below every image.
    stages = [('Source view', 'Original', 'Same screenshot used for all three methods.', 240)]
    stages.extend((short[m], 'Predictions', 'Every predicted region around the building.', 240) for m in methods)
    stages.extend((short[m], 'Recorded surfaces', 'White rims mark matches gained or lost against the current finder.' if m!='current'
                   else 'Recorded surfaces colored by their match to the current finder.', 240) for m in methods)
    stages.append(('Missed surfaces', 'Recorded surfaces', f'{short[recommended]}. The box locates the detail below.', 240))
    stages.extend((short[m], 'Extra predictions', 'All unpaired guesses across the entire screenshot.', 188) for m in methods)
    stages.append(('Original pixels', 'Close-up', 'Inspect the shadow, step or gap in the clean source crop.', 282))
    stages.append(('Recorded detail', 'Close-up', 'One recorded surface. All five close-ups use the same crop.', 282))
    stages.extend((short[m], 'Close-up', 'Matched outline or the guess with the highest whole-surface overlap.', 282) for m in methods)
    tops, cursor = [], HEADER
    for _, _, _, height in stages:
        tops.append(cursor)
        cursor += height+84
    supplemental_top=cursor+35
    page = Image.new('RGB', (LEFT+len(sample)*(COL+GAP)+GAP,supplemental_top+210+5*285+10), PAPER)
    draw = ImageDraw.Draw(page)
    label(draw, (19,16), 'Which surface splits helped?', 39, bold=True)
    label(draw, (20,72), 'Same eight building views. Each experiment starts from the previous finder. A match needs at least 50% overlap with one recorded surface.', 21)
    legend = [('finder','Predicted outline'),('found','Found'),('missed','Missed'),('small_found','Small found'),('small','Small missed'),('extra','Extra guess')]
    x = 20
    for code, name in legend:
        draw.rectangle((x,112,x+18,130), fill=COLORS[code], outline=MUTED)
        label(draw,(x+26,113),name,18)
        x += int(draw.textlength(name,font=font(18)))+56
    label(draw,(x+18,113),'White rim on a recorded face = gained or lost match.',18)
    for n,(title,kind,note,_) in enumerate(stages):
        y=tops[n]
        draw.line((15,y,page.width-15,y),fill='#bdcdc7',width=2)
        label(draw,(15,y+21),kind.upper(),15,fill=MUTED,bold=True)
        label(draw,(15,y+48),title,23,bold=True)
        for i,line in enumerate(textwrap.wrap(note,width=24)):
            label(draw,(15,y+87+23*i),line,16,fill=MUTED)
    reports, totals = [], {m:Counter() for m in methods}
    for col, view in enumerate(sample):
        key = view['frame']
        case, row = lookup[key], rows[key]
        sha = digest(case.path('image'))
        if sha!=row['sha256'] or sha!=view['image_sha256'] or sha!=previous[key]['sha256']:
            raise ValueError(f'Surface review screenshot changed for {key}')
        scene = view['scene_xyxy']
        with Image.open(case.path('image')) as opened:
            rgb = opened.convert('RGB')
        with gzip.open(source/row['predictions'], 'rt', encoding='utf8') as f:
            parts = json.load(f)
        _, reference = recorded(case)
        predictions, reports_by_method, scored = {}, {}, {}
        for method in methods:
            truth, predictions[method], report = detection_maps(case, parts[method], row['scores'][method])
            reports_by_method[method] = report
            totals[method].update(report['counts'])
            scored[method] = _lookup_scores(parts[method],reference,report['references'])
        focus, reason = _choose_focus(scored,reference,previous[key]['reference'])
        box = detail_box(truth, focus, rgb.size)
        x = LEFT + col*(COL+GAP)
        def tile(n, image, *captions):
            _tile(page, x, tops[n], stages[n][3], image, *captions)
        base = reports_by_method['current']['counts']
        label(draw, (x,169), row['layout'].title(), 28, bold=True)
        label(draw, (x,208), f"{base['visible']} visible   ·   {base['scored']} scored   ·   {base['small']} small", 18)
        tile(0, rgb.crop(scene), key.split('/')[-1].replace(row['layout']+'-','').replace('holdout-','').replace('dev-',''),
             key.split('/')[0].replace('_',' ')+' floor')
        direction = {-1:'underside',0:'vertical face',1:'top face'}[reference['patches'][focus]['normal'][1]]
        tile(11, rgb.crop(box), f"R{focus}   ·   {direction}", reason)
        meta = dict(frame=key, layout=row['layout'], sha256=sha, scene_xyxy=scene, focus_reference=focus,
                    surface=direction, focus_reason=reason, focus_native_xyxy=list(box), methods={})
        for i,method in enumerate(methods):
            report = reports_by_method[method]
            counts = report['counts']
            pred = predictions[method]
            gained = [rid for rid,r in scored[method].items() if r['found'] and not scored['current'][rid]['found']]
            lost = [rid for rid,r in scored[method].items() if not r['found'] and scored['current'][rid]['found']]
            painted = paint(rgb,pred,{p['id']:COLORS['finder'] for p in report['predictions']})
            tile(1+i,painted.crop(scene),f"{counts['found']} found / {counts['scored']} scored   ·   {counts['proposed']} guesses",
                 f"{len(gained)} gained   ·   {len(lost)} lost" if i else f"{counts['missed']} scored surfaces missed")
            graded = _mark_grades(rgb,truth,report,gained+lost)
            tile(4+i,graded.crop(scene),f"{counts['found']} found   ·   {counts['missed']} missed",
                 f"{counts['small_found']} of {counts['small']} small surfaces paired")
            if method==recommended:
                misses = paint(rgb,truth,{r['id']:COLORS['missed'] for r in report['references'] if r['status']=='missed'}).copy()
                locator = ImageDraw.Draw(misses)
                locator.rectangle(box,outline=INK,width=5)
                locator.rectangle(box,outline='white',width=2)
                tile(7,misses.crop(scene),f"{short[method]}   ·   {counts['missed']} missed",f"White box follows R{focus} below.")
                target = next(r for r in report['references'] if r['id']==focus)
                tile(12,paint(rgb,truth,{focus:COLORS[target['status']]}).crop(box),f"R{focus}   ·   {direction}",
                     f"{target['pixels']:,} visible pixels in the full picture")
            extras = {p['id']:COLORS['extra'] for p in report['predictions'] if p['status']=='extra'}
            tile(8+i,paint(rgb,pred,extras),f"{counts['unpaired']} extra guesses", 'Full screenshot')
            selected_focus = scored[method][focus]
            pi = selected_focus['index']
            detail, in_box = _detail(rgb,parts[method][pi-1] if pi else None,box)
            status = 'Found' if selected_focus['found'] else 'Missed'
            if focus in gained:
                status = 'New match'
            elif focus in lost:
                status = 'Lost match'
            detail_note = (f"P{pi} paired with R{focus}." if selected_focus['found'] else
                           f"P{pi} is the closest guess." if pi and in_box else
                           'Closest guess falls outside this crop.' if pi else 'No guess touches this surface.')
            overlap = selected_focus['overlap']
            pretty = f'{overlap:.1%}' if abs(overlap-.5)<.005 else f'{overlap:.0%}'
            tile(13+i,detail,f"{status}   ·   {pretty} whole-surface overlap",detail_note)
            meta['methods'][method] = dict(**report, gained=gained, lost=lost, focus=selected_focus)
        reports.append(meta)
    summary='   ·   '.join(f"{short[m]} {totals[m]['found']}/{totals[m]['scored']}" for m in methods)
    label(draw, (15,180), 'SURFACES IN EACH VIEW', 16, bold=True)
    label(draw, (page.width-1240,25), summary+'   ·   eight pictures shown', 23, bold=True)
    label(draw, (page.width-1240,65), 'Compared here: '+NAMES[methods[1]]+' and '+NAMES[methods[2]]+'.', 19)
    supplemental = _low_angle_section(page,supplemental_top,source,cases,lookup,sample,selected['strip'])
    page.save(out/'comparison.jpg', quality=94, subsampling=0)
    save(out/'sheet-views.json',dict(selected=selected, methods=list(methods), totals=totals, views=reports,
         supplemental_undersides=supplemental,
         focus_rule='Prefer changed scored matches, especially undersides, then overlap changes of at least ten points. Otherwise retain the previous overhang review focus.'))
