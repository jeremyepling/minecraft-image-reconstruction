"""Picture-by-picture diagnostic of visible reference surfaces and the finder's proposals."""
from collections import Counter

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageOps

from benchmark.camera_round import save
from benchmark.camera_sheet import font
from benchmark.surface_detection_maps import detection_maps
from benchmark.surface_sheet import text


PAPER,INK,MUTED='#f8f7f1','#183135','#4b6465'
BG='#e2e7e4'
COLORS=dict(found='#28da9c',missed='#ffb12b',extra='#ff589e',small_found='#72beff',small='#c1cad1',finder='#3ce9ea')
LABEL,COL,GAP=238,386,14
ROWS=[
    ('Source view', 'Same camera as the pipeline sheet',355),
    ('Finder output', 'Every proposed region',565),
    ('Recorded surfaces', 'Every visible region, graded',445),
    ('Missed surfaces', 'Scored misses in amber',425),
    ('Unmatched guesses', 'Full frame and largest extra',575),
    ('Reference detail', 'One recorded surface',460),
    ('Prediction detail', 'Same area, same scale',480)]


def paint(image,ids,colors):
    """One hue per status, with a dark rim and a bright edge between adjacent IDs."""
    if not colors:
        return image
    lut=np.zeros((int(ids.max())+1,3),np.uint8)
    active=np.zeros(len(lut),bool)
    for index,color in colors.items():
        if index<len(lut):
            lut[index]=ImageColor.getrgb(color)
            active[index]=True
    inside=active[ids]
    borders=np.zeros_like(inside)
    for axis in (0,1):
        a,b=(np.s_[:-1,:],np.s_[1:,:]) if axis==0 else (np.s_[:,:-1],np.s_[:,1:])
        different=ids[a]!=ids[b]
        borders[a]|=different
        borders[b]|=different
    borders&=inside
    rim=np.asarray(Image.fromarray(borders).filter(ImageFilter.MaxFilter(5))) & inside
    rgb=np.array(image)
    rgb[inside]=np.rint(.44*rgb[inside]+.56*lut[ids[inside]]).astype(np.uint8)
    rgb[rim]=[31,45,49]
    rgb[borders]=lut[ids[borders]]
    return Image.fromarray(rgb)


def focus_box(ids,target,size):
    yy,xx=np.where(ids==target)
    if not len(xx):
        return (0,0,*size)
    x0,x1,y0,y1=xx.min(),xx.max()+1,yy.min(),yy.max()+1
    diag=max(x1-x0,y1-y0)
    # Keep part of each neighboring plane in view, including around a one-pixel sliver.
    w=min(size[0],max(118,int(diag*1.36)))
    h=min(size[1],max(108,int((y1-y0)*1.38),int(w*.70)))
    x=int(np.clip((x0+x1-w)//2,0,size[0]-w))
    y=int(np.clip((y0+y1-h)//2,0,size[1]-h))
    return (x,y,x+w,y+h)


def panel(page,image,box,x,y,h):
    crop=image.crop(box)
    view=ImageOps.contain(crop,(COL,h),Image.Resampling.LANCZOS)
    pos=(x+(COL-view.width)//2,y+(h-view.height)//2)
    ImageDraw.Draw(page).rectangle((x,y,x+COL-1,y+h-1),fill=BG)
    page.paste(view,pos)


def caption(draw,x,y,*lines):
    for i,line in enumerate(lines):
        text(draw,x,y+27*i,line,17,fill=MUTED)


def diagnostics(out,examples,parts_by_key,lookup,by_key,method):
    """Render the same selected frame for each layout. All masks are scored offline."""
    rows=[]
    totals=Counter()
    data=[]
    for ex in examples:
        key=ex['key']
        actual,pred,report=detection_maps(lookup[key],parts_by_key[key],by_key[key]['scores'][method])
        refs=report['references']
        missed=[r for r in refs if r['status']=='missed']
        matched=[r for r in refs if r['status']=='found']
        issue=(max(missed,key=lambda r:(r['pixels'],-r['id'])) if missed else
               min(matched,key=lambda r:r['match_iou']) if matched else None)
        box=focus_box(actual,issue['id'],ex['image'].size) if issue else ex['box']
        report.update(frame=key,layout=ex['layout'],image_sha256=by_key[key]['sha256'],scene_xyxy=ex['box'],
                      focus_reference=issue['id'] if issue else None,focus_native_xyxy=list(box))
        totals.update(report['counts'])
        rows.append(report)
        data.append((ex,actual,pred,report,issue,box))

    heading=288
    width=LABEL+len(data)*(COL+GAP)+GAP
    page=Image.new('RGB',(width,max(width+1,heading+sum(r[2] for r in ROWS)+24)),PAPER)
    draw=ImageDraw.Draw(page)
    text(draw,20,16,'Which visible surfaces did the finder recover?',39,True)
    text(draw,20,71,f"{totals['found']} of {totals['scored']} scored surfaces found across these {len(rows)} pictures. {totals['visible']} surfaces are actually visible. The small-surface check paired {totals['small_found']} more. Of {totals['proposed']} predictions, {totals['unpaired']} remained unpaired.",20)
    legend=[('found','Found'),('missed','Missed'),('small_found','Small surface paired'),('small','Small, unpaired'),('extra','Extra prediction')]
    x=22
    for status,name in legend:
        draw.rectangle((x,119,x+18,137),fill=COLORS[status],outline=MUTED)
        text(draw,x+26,116,name,18,True)
        x+=int(draw.textlength(name,font=font(18,True)))+64
    text(draw,x+30,116,'Each match needs 50% overlap: shared area divided by the area covered by either outline. Each surface can match once.',18,fill=MUTED)
    y=heading
    starts=[]
    for title,note,h in ROWS:
        starts.append(y)
        draw.line((16,y,width-16,y),fill='#cad4d0',width=2)
        text(draw,16,y+22,title,21,True)
        # Keep short notes clear of the first picture.
        words=note.split()
        for j in range(0,len(words),3):
            text(draw,16,y+60+24*(j//3),' '.join(words[j:j+3]),16,fill=MUTED)
        y+=h
    for col,(ex,actual,pred,report,issue,zbox) in enumerate(data):
        image,scene=ex['image'],ex['box']
        full=(0,0,*image.size)
        x=LABEL+col*(COL+GAP)
        c=report['counts']
        preds=report['predictions']
        refs=report['references']
        text(draw,x,160,ex['layout'].title(),27,True)
        text(draw,x,200,f"{c['found']} found / {c['scored']} scored  ·  {c['missed']} missed",20,True)
        caption(draw,x,235,f"{c['visible']} actually visible  ·  {c['small']} below size cutoff")
        panel(page,image,scene,x,starts[0]+18,276)
        caption(draw,x,starts[0]+307,'Saved screenshot, enlarged around building')

        y=starts[1]+18
        raw=paint(image,pred,{p['id']:COLORS['finder'] for p in preds})
        panel(page,raw,scene,x,y,270)
        caption(draw,x,y+282,f"{c['proposed']} proposals in the complete frame")
        panel(page,raw,full,x,y+318,180)
        caption(draw,x,y+510,'Full frame')

        y=starts[2]+18
        truth=paint(image,actual,{r['id']:COLORS[r['status']] for r in refs})
        panel(page,truth,scene,x,y,285)
        caption(draw,x,y+300,f"{c['found']} found  ·  {c['missed']} missed",
                f"{c['small_found']} of {c['small']} small surfaces paired separately")

        y=starts[3]+18
        missmap=paint(image,actual,{r['id']:COLORS['missed'] for r in refs if r['status']=='missed'})
        if issue:
            missmap=missmap.copy()
            frame=ImageDraw.Draw(missmap)
            frame.rectangle(zbox,outline=INK,width=4)
            frame.rectangle(zbox,outline='white',width=2)
        panel(page,missmap,scene,x,y,285)
        caption(draw,x,y+301,f"{c['missed']} missed scored surfaces" if c['missed'] else 'All scored surfaces were found.',
                'White box locates the detail below.' if issue else '')

        y=starts[4]+18
        unmatched=[p for p in preds if p['status']!='found']
        extras=[p for p in unmatched if p['status']=='extra']
        unpaired=paint(image,pred,{p['id']:COLORS[p['status']] for p in unmatched})
        panel(page,unpaired,full,x,y,215)
        caption(draw,x,y+224,f"{c['unpaired']} extra  ·  {c['small_found']} paired to small surfaces")
        if extras:
            extra=max(extras,key=lambda p:p['pixels'])
            panel(page,unpaired,focus_box(pred,extra['id'],image.size),x,y+266,213)
            caption(draw,x,y+491,f"Largest extra: P{extra['id']}",extra['reason'].capitalize())
        else:
            caption(draw,x,y+284,'Every proposal paired to a visible surface.')

        if issue is None:
            continue
        rid=issue['id']
        found=issue['status']=='found'
        reason='Lowest-overlap match' if found else 'Largest missed surface'
        y=starts[5]+20
        panel(page,paint(image,actual,{rid:COLORS[issue['status']]}),zbox,x,y,330)
        caption(draw,x,y+343,f"R{rid}   {reason}",'Recorded location, enlarged')

        y=starts[6]+20
        pi=issue['prediction'] if found else issue['best_prediction']
        record=next((p for p in preds if p['id']==pi),None)
        proposed=paint(image,pred,{pi:COLORS['finder']}) if record else image
        panel(page,proposed,zbox,x,y,330)
        if record:
            iou=issue['match_iou'] if found else issue['iou']
            caption(draw,x,y+343,f"P{pi} vs R{rid}   ·   {iou:.0%} outline overlap",
                    f"P{pi} covers {issue['coverage']:.0%} of R{rid}." if not found else f"P{pi} matched R{rid}.",
                    (f"It matched R{record['reference']} instead." if record['reference'] else 'It had no one-to-one match.') if not found else '')
        else:
            caption(draw,x,y+343,f'No proposal covers R{rid}.')
    page.save(out/'surface-detection.jpg',quality=95,subsampling=0)
    save(out/'surface-detection.json',dict(method=method,stage='Surface Finder',totals=dict(totals),views=rows,
         scoring='One-to-one IoU >= 0.5. Main score: four or more visible block faces and at least 24 samples on the four-pixel grid. Remaining predictions can pair separately to smaller references at the same IoU on native pixels.',
         details='Largest missed scored reference by visible pixel area. If all scored surfaces match, show the lowest-IoU match. R and P numbers are local to each picture.'))
    return {r['frame']:r for r in rows}
