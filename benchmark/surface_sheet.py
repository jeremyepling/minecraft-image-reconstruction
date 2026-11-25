"""Create the review contact sheet from recorded outputs, with native-frame feedback regions."""
import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_round import read, save
from benchmark.camera_sheet import font
from benchmark.surface_metrics import DOWNSAMPLE
from benchmark.surface_reference import recorded
from engine.geometry import project, transform


PAPER, INK, DIM = '#f7f5ee', '#172f32', '#476164'
CYAN, GOLD, RED = '#37fff4', '#ffd055', '#ff6684'
W, M, GAP = 1580, 34, 17
COL = (W-2*M-2*GAP)//3


def text(d,x,y,s,size=20,bold=False,fill=INK):
    d.text((x,y),s,font=font(size,bold),fill=fill,anchor='lt')


def poly(d,points,color,width=2,close=False):
    p = [tuple(x) for x in np.asarray(points)]
    if len(p)>1:
        d.line(p+([p[0]] if close else []),fill=color,width=width,joint='curve')


def box_of(q,size,pad):
    xy = np.asarray(q).reshape(-1,2)
    lo = np.maximum([0,0],np.floor(xy.min(0)-pad)).astype(int)
    hi = np.minimum(size,np.ceil(xy.max(0)+pad)).astype(int)
    return tuple(map(int,[*lo,*hi]))


def panel(canvas,im,box,x,y,width,height):
    crop = im.crop(box)
    scale = min(width/crop.width,height/crop.height)
    view = crop.resize(tuple(max(1,round(n*scale)) for n in crop.size),
                       Image.Resampling.NEAREST if scale>1.15 else Image.Resampling.LANCZOS)
    ImageDraw.Draw(canvas).rectangle((x,y,x+width-1,y+height-1),fill='#dfdfd7')
    canvas.paste(view,(x+(width-view.width)//2,y+(height-view.height)//2))


def captured(case):
    with Image.open(case.path('image')) as im:
        src = im.convert('RGB')
    return src,recorded(case)[1]


def scene_box(t,src):
    ys,xs = np.nonzero(t['id_map'])
    return box_of([[xs.min()*DOWNSAMPLE,ys.min()*DOWNSAMPLE],
                   [(1+xs.max())*DOWNSAMPLE,(1+ys.max())*DOWNSAMPLE]],src.size,24)


def overlay_scene(src,t,parts,scores):
    ids = t['id_map']
    boundaries = np.zeros_like(ids,dtype=bool)
    boundaries[:,1:] |= ids[:,1:] != ids[:,:-1]
    boundaries[1:,:] |= ids[1:,:] != ids[:-1,:]
    overlay = src.copy()
    overlay.paste(GOLD,(0,0,*src.size),Image.fromarray(boundaries).resize(src.size,Image.Resampling.NEAREST))
    d = ImageDraw.Draw(overlay)
    matches = {r['proposal_index'] for r in scores['matches']}
    for i,p in enumerate(parts):
        poly(d,p['quad'],CYAN if i in matches else RED,3,True)
    return overlay


def pitch_panel(src,patch_truth,row):
    im = src.copy()
    d = ImageDraw.Draw(im)
    q = np.asarray(row['quad'])
    poly(d,q,'#ffffff',1,True)
    H = transform(q)
    inside = project(np.linalg.inv(H),patch_truth)
    for pt in patch_truth[np.all((inside>=0)&(inside<=1),axis=1)]:
        x,y=pt
        d.ellipse((x-2,y-2,x+2,y+2),fill=GOLD)
    pred = row['prediction']
    if pred.get('counts') is not None:
        counts = pred['counts']
        phases = (pred.get('grid') or {}).get('phase_blocks') or pred.get('evidence',{}).get('phase_by_axis') or [None,None]
        for axis, phase in enumerate(phases):
            if phase is None:
                continue
            for loc in np.arange(phase,counts[axis]+1e-7,1)/counts[axis]:
                uv = [[loc,0],[loc,1]] if axis==0 else [[0,loc],[1,loc]]
                poly(d,project(H,uv),CYAN,1)
    return im


def count_string(c):
    return 'unresolved' if c is None else f'{c[0]:.2f} × {c[1]:.2f} blocks'


def gallery(out,groups):
    cases = sorted([c for items in groups.values() for c,_ in items if c.floor=='light_gray_concrete' and c.frame['angle']=='low-left'],
                   key=lambda c:(c.frame['layout'],c.frame['palette']))
    canvas = Image.new('RGB',(1440,100+228*((len(cases)+2)//3)+36),PAPER)
    d = ImageDraw.Draw(canvas)
    text(d,30,18,'The six test structures',36,True)
    text(d,30,67,'Native captures on light gray concrete  /  two palettes per layout  /  clear noon',22)
    for idx,c in enumerate(cases):
        x,y = 30+(idx%3)*466,107+(idx//3)*228
        im,t = captured(c)
        panel(canvas,im,scene_box(t,im),x,y,447,179)
        text(d,x,y+185,c.frame['layout'].title()+'  /  '+c.frame['palette'],19,True)
    canvas.save(out/'fixtures.jpg',quality=93,subsampling=0)


def render(out,groups,stages):
    gallery(out,groups)
    canvas = Image.new('RGB',(W,1985),PAPER)
    d = ImageDraw.Draw(canvas)
    text(d,M,20,'Finding planes and counting blocks',42,True)
    text(d,M,80,'Java 26.3  /  fresh confirmation  /  clear noon  /  light gray concrete floor',23,fill=DIM)
    items = groups['confirmation']
    c,folder = next((c,p) for c,p in items if c.frame['layout']=='crossing' and c.frame['palette']=='plaster' and c.frame['angle']=='low-left')
    im,t = captured(c)
    report = read(out/folder/'outlines.json')
    box = scene_box(t,im)
    midx,midy=(box[0]+box[2])/2,(box[1]+box[3])/2
    zoom = box_of([[midx-.19*(box[2]-box[0]),midy-.08*(box[3]-box[1])],
                   [midx+.19*(box[2]-box[0]),midy+.44*(box[3]-box[1])]],im.size,0)
    text(d,M,135,'Visible outlines on the elevated crossing',27,True)
    text(d,M,177,'Gold marks true boundaries. Cyan outlines match a visible plane at IoU ≥ 0.50. Pink outlines miss that threshold.',21)
    for i,(method,title) in enumerate((('contact','Ground contact'),('edges','Connected edges'),('edges+shade','Edges with shading'))):
        x = M+i*(COL+GAP)
        row = report[method]
        s = row['scores']
        overlay = overlay_scene(im,t,row['proposals'],s)
        text(d,x,220,title,23,True)
        panel(canvas,overlay,box,x,255,COL,276)
        text(d,x,545,f"{s['matched']} matched  /  {s['proposed']} proposed  /  {s['eligible']} visible",18)
        panel(canvas,overlay,zoom,x,584,COL,225)
    text(d,M,823,'The enlarged views show the same opening, bridge underside and stair edges.',19,fill=DIM)
    regions = [dict(image=c.id,floor=c.floor,region_xyxy=zoom,stage='outlines')]
    eligible = []
    for case,p in items:
        data=read(out/p/'surfaces.json')
        for old,new in zip(data['texel']['oracle'],data['repeat']['oracle'],strict=True):
            q=np.asarray(new['quad'])
            area=.5*abs(q[:,0]@np.roll(q[:,1],1)-q[:,1]@np.roll(q[:,0],1))
            eligible.append((new['score']['half_block_both'],new['prediction']['counts'] is not None,area,case,p,old,new))
    text(d,M,885,'Block spacing on the building faces',27,True)
    text(d,M,925,'Diagnostic footprints start inside blocks. Gold dots mark true grid intersections. Cyan lines show returned boundary phases.',20)
    for n,(heading,pool) in enumerate((('Accurate repeat count',[e for e in eligible if e[0]]),
                                        ('Missed or unresolved count',[e for e in eligible if not e[0]]))):
        if not pool:
            continue
        _,_,_,case,_,old,new = max(pool,key=lambda e:(e[6]['prediction']['grid'] is not None,e[1],e[2]))
        src,gt = captured(case)
        q = new['quad']
        crop = box_of(q,src.size,5)
        y=973+n*302
        text(d,M,y,heading+'  /  '+case.frame['layout']+'  /  '+case.frame['palette']+'  /  '+case.frame['angle'],21,True)
        for i,(title,view) in enumerate((('Measured span   '+count_string(new['score']['expected_counts']),src),
                              ('Texel baseline   '+count_string(old['prediction']['counts']),pitch_panel(src,gt['vertices'][new['patch_id']],old)),
                              ('Texture repeat   '+count_string(new['prediction']['counts']),pitch_panel(src,gt['vertices'][new['patch_id']],new)))):
            x = M+i*(COL+GAP)
            panel(canvas,view,crop,x,y+36,COL,208)
            text(d,x,y+250,title,17,bold=True)
        regions.append(dict(image=case.id,floor=case.floor,region_xyxy=crop,stage='pitch'))
    text(d,M,1582,'Depth from the same outlines',27,True)
    text(d,M,1620,'Distances are in blocks along the viewing direction. Each column uses the same two predicted footprints.',20)
    candidates = []
    for case,p in items:
        data = read(out/p/'surfaces.json')[stages['pitch']]
        measured = [r for r in data['estimated_lift'] if 'depth' in r]
        if measured:
            candidates.append((len(measured),max(r['depth_error_blocks'] for r in measured),case,data))
    if candidates:
        _,_,case,data = max(candidates,key=lambda c:c[:2])
        src,_ = captured(case)
        shown = sorted((r for r in data['estimated_lift'] if 'depth' in r),key=lambda r:-r['depth_error_blocks'])[:2]
        patches = {p['id']:p for p in data['model']['planes']}
        known = {r['id']:r for r in data['known_lift']}
        crop = box_of([patches[r['id']]['quad'] for r in shown],src.size,24)
        for i,title in enumerate(('Recorded world depth','Estimated camera','Recorded camera diagnostic')):
            x = M+i*(COL+GAP)
            img = src.copy()
            pen = ImageDraw.Draw(img)
            labels = []
            for j,r in enumerate(shown):
                color = (CYAN,GOLD)[j]
                q = patches[r['id']]['quad']
                poly(pen,q,color,4,True)
                val = (r['truth_depth'],r['depth'],known.get(r['id'],{}).get('depth'))[i]
                letter = chr(ord('A')+j)
                pt = np.mean(q,axis=0)
                pen.text(tuple(pt),letter,font=font(18,True),fill=color,stroke_width=2,stroke_fill='#102323',anchor='mm')
                labels.append(letter+'  '+(f'{val:.2f}' if val is not None else 'unknown'))
            text(d,x,1662,title,21,True)
            panel(canvas,img,crop,x,1701,COL,196)
            text(d,x,1911,'    /    '.join(labels)+' blocks',19,True)
        regions.append(dict(image=case.id,floor=case.floor,region_xyxy=crop,stage='lift'))
    else:
        text(d,M,1692,'No matched patch yielded metric depth.',24)
    canvas.save(out/'comparison.jpg',quality=94,subsampling=0)
    save(out/'review-regions.json',regions)
