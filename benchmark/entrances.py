"""Inspect oblique openings with a frozen policy and explain each depth decision."""
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageOps,ImageDraw

from benchmark.oblique_recess import dark_boxes,internal_repeat,rectify_wall
from engine.corner_texture import predict
from engine.geometry import UNIT,project
from engine.rim_spacing import local_spacing,texture_detail

POLICIES=('dark','regions','edges','combined','corroborated_additions')
PROFILE=Path(__file__).parent/'entrance-policy.json'


def iou(a,b):
    shared=np.maximum(0,np.minimum(a[1],b[1])-np.maximum(a[0],b[0])).prod()
    union=np.prod(a[1]-a[0])+np.prod(b[1]-b[0])-shared
    return shared/max(1e-9,union)


def propose(wall,policy):
    rect,pitch=wall['rect'],wall['pitch']
    if policy=='dark':
        return dark_boxes(wall['seeds'].copy())
    if policy=='edges':
        from engine.entrance_edges import boxes
        return boxes(rect,pitch)
    if policy not in ('regions','combined'):
        raise ValueError(f'Unknown proposal method: {policy}')
    from engine.entrance_regions import boxes
    regions=boxes(rect,pitch)
    if policy=='regions':
        return regions
    combined=list(regions)
    for box in propose(wall,'edges'):
        if all(iou(box,r)<.4 for r in combined):
            combined.append(box)
    return combined


def agreed_extras(wall,base):
    regions=propose(wall,'regions')
    edges=propose(wall,'edges')
    return [r for r in regions if all(iou(r,d)<.35 for d in base)
            and any(iou(r,e)>=.55 for e in edges)]


def rim_pixels(wall,a,b):
    return project(wall['H'],a+UNIT*(b-a)).round(3).tolist()


def measure(wall,policy,review_extras=False):
    if 'rect' not in wall:
        return dict(policy=policy,status='unresolved',reason=wall['reason'],candidates=[])
    if policy=='corroborated_additions' or (policy=='dark' and review_extras):
        # Freeze the incumbent crops and their depth evidence. Adding an aperture
        # changes the exterior wall mask, so never recompute the incumbent rows.
        base=propose(wall,'dark')
        extras=agreed_extras(wall,base)
        kept=measure_boxes(wall,base,base)
        if policy=='dark':
            additions=[dict(rim=rim_pixels(wall,a,b),depth=None,status='review_only',
                            reason='two rim finders agree; depth unverified') for a,b in extras]
        else:
            additions=measure_boxes(wall,extras,[*base,*extras]) if extras else []
        for row in additions:
            row['proposal_source']='local region and hard edges'
        rows=kept+additions
    else:
        boxes=propose(wall,policy)
        rows=measure_boxes(wall,boxes,boxes)
    return dict(policy=policy,status='candidates_evaluated' if rows else 'no_candidates',
                grid=wall['grid'],calibration=wall['calibration'],candidates=rows)


def measure_boxes(wall,boxes,excluded):
    rect,pitch=wall['rect'],wall['pitch']
    field=texture_detail(rect)
    front=wall['front'] & ~wall['seeds'].repeat(pitch,axis=0).repeat(pitch,axis=1)
    for a,b in excluded:
        lo=np.maximum(0,np.floor(a*pitch-4)).astype(int)
        hi=np.ceil(b*pitch+4).astype(int)
        front[lo[1]:hi[1],lo[0]:hi[0]]=False
    decisions=[]
    for a,b in boxes:
        q=(a+UNIT*(b-a))*pitch
        local=local_spacing(field,front,a*pitch,b*pitch,pitch)
        row=dict(rim=rim_pixels(wall,a,b),local_spacing=local,depth=None)
        if not all(local):
            row.update(status='unresolved',reason='weak exterior block repetition')
        else:
            ratios=[v['ratio'] for v in local]
            pred=predict(rect,q,wall['vp'],wall['distance'],pitch,ratios)
            depth=pred['depth']
            row.update(texture_score=pred.get('score'),score_margin=pred.get('margin'))
            if depth is None:
                row.update(status='unresolved',reason=pred.get('reason','depth hypotheses are too close or weak'))
            elif depth==0:
                row.update(status='flush_supported',depth=0,reason='texture repeat supports the facade plane')
            else:
                check=internal_repeat(field,a*pitch,b*pitch,wall['vp'],wall['distance'],depth,pitch,ratios)
                if check['veto']:
                    row.update(status='unresolved',reason='interior repeat favors the front plane over proposed depth')
                else:
                    row.update(status='depth_supported',depth=int(depth),reason='distinct texture scale at the proposed back plane')
        decisions.append(row)
    return decisions


def inspect(path,profile=PROFILE):
    config=json.loads(profile.read_text())
    policy=config['selected']
    if policy not in POLICIES:
        raise ValueError('Unknown entrance policy')
    with Image.open(path) as im:
        return measure(rectify_wall(ImageOps.exif_transpose(im)),policy,config.get('review_extras',False))


def inspect_to_folder(path,out):
    from benchmark.live_color_gallery import label_font
    result=inspect(path)
    out.mkdir(parents=True,exist_ok=True)
    (out/'openings.json').write_text(json.dumps(result,indent=2)+'\n')
    with Image.open(path) as im:
        preview=ImageOps.exif_transpose(im).convert('RGB')
    pen=ImageDraw.Draw(preview)
    font=label_font(17)
    for row in result['candidates']:
        points=[tuple(p) for p in row['rim']]
        d=row['depth']
        color,label=('#f7c45e','unresolved') if d is None else (('#a3c6ff','flush') if d==0 else ('#51ddad',f'{d} blocks'))
        if row['status']=='review_only':
            color,label='#ffa1d1','review rim'
        elif d==1:
            label='1 block'
        pen.line(points+points[:1],fill=color,width=3)
        x,y=min(p[0] for p in points),min(p[1] for p in points)
        pen.text((x,max(0,y-22)),label,fill=color,font=font,stroke_width=2,stroke_fill='#151a1f')
    if not result['candidates']:
        pen.text((12,12),result.get('reason','No entrance candidates'),fill='white',font=font,stroke_width=2,stroke_fill='black')
    preview.save(out/'review.png')
    print(f"{result['policy']}: {result['status']}, {len(result['candidates'])} proposals -> {out}")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=inspect(a.image)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
