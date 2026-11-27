"""Predict a visible wall's vertical spacing using neighboring textured uprights."""
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.relative_face_round import SHARED
from engine.masked_count import _options
from benchmark.native_surface_counter import basis
from benchmark.full_block_lattice import verify_sprite
from engine.geometry import project, transform, UNIT
from engine.surface_model import lens

OUT = ROOT / 'runs/garden-pillar-ruler-01'


def crop_info(folder, plane, label):
    crop = plane['source_crop']
    x,y,right,bottom = crop['xyxy']
    rgba = np.array(Image.open(folder/crop['path']).convert('RGBA'))
    mask=rgba[:,:,3]>0
    widths=mask.sum(1); heights=mask.sum(0)
    tall = np.quantile(heights[heights>0],.8)
    sample=dict(rgb=rgba[:,:,:3],mask=mask)
    candidates = []
    if bottom-y >= 2.4*(right-x) and tall >= 32 and abs((plane.get('normal_camera') or [0,1,0])[1])<.65:
        for peak in _options(sample,1)[0]:
            if peak['source']=='repeated texture' and peak['rows']>=8 and peak['agreement']>=.35:
                lag=float(peak['lag'])
                if 2.5<=tall/lag<=12:
                    candidates.append((lag,float(peak['agreement'])))
    return dict(label=label,id=plane['id'],box=crop['xyxy'],height=float(tall),
                center=[(x+right)/2,(y+bottom)/2],color=np.median(rgba[:,:,:3][mask],axis=0),
                mask=mask,rgba=rgba,candidates=candidates)


def estimate(receiver, items):
    x,y,X,Y=receiver['box']
    if X-x<.5*(Y-y) or Y-y<18 or X-x>4*(Y-y):
        return None
    cx=receiver['center'][0]
    near=[]
    for p in items:
        if not p['candidates'] or p is receiver:continue
        px,py,pX,pY=p['box']
        overlap=max(0,min(Y,pY)-max(y,py))/(Y-y)
        if overlap>=.6 and abs(p['center'][0]-cx)<=1.8*max(X-x,Y-y):
            near.append(p)
    pairs=[]
    for a,b in combinations(sorted(near,key=lambda p:p['center'][0]),2):
        left,right=a['center'][0],b['center'][0]
        if not left<cx<right or np.linalg.norm(a['color']-b['color'])>48:continue
        for pa,qa in a['candidates']:
            for pb,qb in b['candidates']:
                ratio=(pa/a['height'])/(pb/b['height'])
                if abs(np.log(ratio))>.16 or abs(np.log(pa/pb))>.38:continue
                score=(qa+qb)/2-abs(np.log(ratio))-.12*(right-left)/max(X-x,Y-y)
                pitch=pa+(pb-pa)*(cx-left)/(right-left)
                pairs.append((score,pitch,a['label'],b['label'],pa,pb))
    if not pairs:return None
    score,pitch,a,b,pa,pb=max(pairs)
    return dict(pillars=[a,b],pillar_pitch=[pa,pb],pitch=pitch,visible_span=receiver['height']/pitch)


def grade(receiver,ids,faces):
    x,y,X,Y=receiver['box']
    face_ids, weights=np.unique(ids[y:Y,x:X][receiver['mask']],return_counts=True)
    planes=Counter()
    for i,n in zip(face_ids,weights):
        if i:planes[faces[i]['surface_id']]+=int(n)
    if not planes:return None
    plane,n=planes.most_common(1)[0]
    if n<.6*weights.sum():return None
    delta=[];rows=set()
    for i,n in zip(face_ids,weights):
        if not i or faces[i]['surface_id']!=plane or n<8:continue
        f=faces[i]; xyz=np.array(f['corners_world']);xy=np.array(f['corners_px'])
        if f['direction'] in ['up','down'] or np.ptp(xyz[:,1])!=1:continue
        for j in range(4):
            k=(j+1)%4
            if abs(xyz[j,1]-xyz[k,1])==1:
                delta.append(abs(xy[j,1]-xy[k,1]));rows.add(f['xyz_world'][1])
    return dict(pitch=float(np.median(delta)),visible_rows=len(rows)) if delta else None


def supported_rims(mask):
    """Measure top and bottom on tall visible columns, excluding clipped pillar and door silhouettes."""
    heights=mask.sum(0)
    full=np.flatnonzero(heights>=.85*np.quantile(heights[heights>0],.8))
    if len(full)<8 or np.ptp(full)<.22*mask.shape[1]:return None
    ends=np.array([np.flatnonzero(mask[:,u])[[0,-1]] for u in full])
    x=np.flatnonzero(heights>=.4*np.median(heights[full]))[[0,-1]]
    y=[]
    for boundary in ends.T:
        a,b=np.triu_indices(len(full),1)
        wide=full[b]-full[a]>=.25*np.ptp(full)
        slope=np.median(((boundary[b]-boundary[a])/(full[b]-full[a]))[wide])
        if abs(slope)>.55:return None
        y.append(slope*x+np.median(boundary-slope*full))
    top,bottom=y
    return np.array([[x[0],bottom[0]],[x[1],bottom[1]],[x[1],top[1]],[x[0],top[0]]])


def wall_chart(p, prediction, normal, up, K, rgb, robust=False):
    mask=p['mask']
    x,y,X,Y=p['box']
    if mask.mean()<.50 or not .65<(X-x)/(Y-y)<2. or normal is None or abs(normal[1])>.65:
        return None
    height=round(prediction['visible_span'])
    if height<2 or abs(height-prediction['visible_span'])>.35:
        return None
    center=np.array(p['center'])
    H=basis(np.array(normal),up,center,K)
    if H is None:return None
    steps=project(H,np.eye(2))-center
    pitch_x=prediction['pitch']*abs(steps[0,0])/max(.25,abs(steps[1,1]))
    width=round(np.quantile(mask.sum(1),.8)/pitch_x)
    if width<1:return None
    # Fit top/bottom to actual mask support in each of four corner regions.
    # This proposes an origin at the exposed rim; a clipped rim can be off-phase.
    ys,xs=np.nonzero(mask)
    def tip(u,v):
        horizontal=np.quantile(xs,.02 if u==0 else .98)
        strip=abs(xs-horizontal)<=max(2,.07*(X-x))
        return [x+horizontal,y+np.quantile(ys[strip],.98 if v==0 else .02)]
    quad=np.array([tip(u,v) for u,v in UNIT])
    if robust:
        fitted=supported_rims(mask)
        if fitted is None:return None
        quad=fitted+[x,y]
        width=round(np.mean(np.abs(quad[[1,2],0]-quad[[0,3],0]))/pitch_x)
        if width<1:return None
    grid=transform(quad)@np.diag([1/width,1/height,1])
    cells=[]
    offsets=np.stack(np.meshgrid(np.linspace(.12,.88,5),np.linspace(.12,.88,5)),-1).reshape(-1,2)
    for v in range(height):
        for u in range(width):
            coords=np.floor(project(grid,offsets+[u,v])).astype(int)-[x,y]
            if np.any(coords<0) or np.any(coords>=np.array([X-x,Y-y])):continue
            inside=mask[coords[:,1],coords[:,0]]
            if not inside[12] or inside.mean()<.8:continue
            corners=project(grid,UNIT+[u,v])
            hit=verify_sprite(rgb,corners[[3,2,1,0]])
            cells.append(dict(u=u,v=v,block_id=hit[0] if hit else 'smooth_stone',
                              confidence=hit[2] if hit else 0.,fallback=hit is None))
    return dict(id=p['id'],H=grid.tolist(),cells=cells,normal_camera=normal,confidence=0.,
                method='bracketing-pillar-pitch') if cells else None


def link_bays(charts, items, rgb):
    """Test the hypothesis that similar windows follow the same visible wall grid."""
    info={p['id']:p for p in items}
    ordered=sorted(charts,key=lambda c:info[c['id']]['center'][0])
    if len(ordered)<2:return charts
    bases=[];pixels=[];records=[];offset=0;last_left=None;last_pitch=None
    for c in ordered:
        p=info[c['id']];base=info[ordered[0]['id']]
        if (np.linalg.norm(p['color']-base['color'])>32 or
                np.dot(c['normal_camera'],ordered[0]['normal_camera'])<.94 or
                abs(p['center'][1]-base['center'][1])>.4*base['height']):
            return charts
        H=np.array(c['H'])
        uv=np.array([[cell['u'],cell['v']] for cell in c['cells']])
        local=uv[:,None,:]+UNIT
        xy=project(H,local.reshape(-1,2))
        left=project(H,[[0,1.5]])[0,0]
        pitch=project(H,[[1,1.5]])[0,0]-left
        if pitch<=0:return charts
        if last_pitch is not None:
            offset+=round((left-last_left)/(.5*(pitch+last_pitch)))
        last_left,last_pitch=left,pitch
        bases.extend((local+[offset,0]).reshape(-1,2));pixels.extend(xy)
        records.extend((cell['u']+offset,cell['v'],p) for cell in c['cells'])
    shared,_=cv2.findHomography(np.asarray(bases),np.asarray(pixels),0)
    if shared is None:return charts
    cells={}
    for u,v,p in records:
        corners=project(shared,UNIT+[u,v])
        fractions=np.stack(np.meshgrid(np.linspace(.15,.85,5),np.linspace(.15,.85,5)),-1)
        q=np.floor(project(shared,fractions+[u,v])).astype(int)-p['box'][:2]
        h,w=p['mask'].shape
        if (q<0).any() or (q[:,:,[0]]>=w).any() or (q[:,:,[1]]>=h).any():continue
        support=p['mask'][q[:,:,1],q[:,:,0]]
        if not support[2,2] or support.mean()<.8:continue
        hit=verify_sprite(rgb,corners[[3,2,1,0]])
        cells[u,v]=dict(u=u,v=v,block_id=hit[0] if hit else 'smooth_stone',
                        confidence=hit[2] if hit else 0.,fallback=hit is None)
    if len(cells)<=max(len(c['cells']) for c in charts):return charts
    return [dict(ordered[0],method='linked-pillar-bays',H=shared.tolist(),cells=list(cells.values()))]


def review_figure(image, added, items, predictions, dest):
    if not added:return
    canvas=image.copy(); draw=ImageDraw.Draw(canvas)
    by_id={p['id']:p for p in items}
    by_label={p['label']:p for p in items}
    donors=set();boxes=[]
    for chart in added:
        p=by_id[chart['id']];donors.update(predictions[p['label']]['pillars'])
        for cell in chart['cells']:
            poly=project(np.array(chart['H']),UNIT+[cell['u'],cell['v']])
            pts=[tuple(pt) for pt in poly];draw.line(pts+[pts[0]],fill='#66ffb2',width=2)
        boxes.append(p['box'])
    for label in sorted(donors):
        p=by_label[label];x,y,X,Y=p['box'];boxes.append(p['box'])
        draw.rectangle((x,y,X,Y),outline='#ffc864',width=2)
        draw.text((x,y-14),label,fill='#ffe199',stroke_width=2,stroke_fill='#17212c')
    for c in added:
        p=by_id[c['id']];x,y,X,Y=p['box']
        draw.text((x+3,Y-15),p['label'],fill='#8affc5',stroke_width=2,stroke_fill='#17212c')
    boxes=np.array(boxes)
    canvas=canvas.crop((max(0,boxes[:,0].min()-12),max(0,boxes[:,1].min()-24),
                        min(canvas.width,boxes[:,2].max()+12),min(canvas.height,boxes[:,3].max()+12)))
    canvas.save(dest)


def main():
    results={}
    for case in read(SHARED/'fixtures.json')['cases']:
        name=case['id']
        folder=WORKTREES/'floating-build-local-grid/runs/floating-build-grid-01/sky'/name/'source-handoff'
        planes=sorted(read(folder/'surfaces.json')['planes'],key=lambda r:-r['support_area'])
        items=[crop_info(folder,p,f'S{i:02}') for i,p in enumerate(planes,1) if p.get('source_crop')]
        predictions={p['label']:estimate(p,items) for p in items}
        packet=read(ROOT/'runs/surface-crop-first-01/registered-seam/sky'/name/'charts.json')
        image=Image.open(case['variants']['sky']['image']).convert('RGB')
        assert digest(Path(case['variants']['sky']['image']))==packet['source_sha256']
        rgb=np.asarray(image)
        tops=[p['normal_camera'] for p in planes if p.get('normal_camera') and abs(p['normal_camera'][1])>=.68]
        K,_=lens([],image.size)
        by_id={p['id']:p for p in planes}
        added=[]; robust=[]
        existing={c['id'] for c in packet['charts']}
        for p in items:
            if p['id'] in existing or not predictions[p['label']] or not tops:continue
            chart=wall_chart(p,predictions[p['label']],by_id[p['id']].get('normal_camera'),np.array(tops[0]),K,rgb)
            if chart:added.append(chart)
            chart=wall_chart(p,predictions[p['label']],by_id[p['id']].get('normal_camera'),np.array(tops[0]),K,rgb,True)
            if chart:robust.append(chart)
        save(OUT/'sky'/name/'charts.json',dict(packet,charts=packet['charts']+added))
        save(OUT/'linked'/'sky'/name/'charts.json',dict(packet,charts=packet['charts']+link_bays(added,items,rgb)))
        for label,charts in [('robust',robust),('robust-linked',link_bays(robust,items,rgb))]:
            save(OUT/label/'sky'/name/'charts.json',dict(packet,charts=packet['charts']+charts))
        review_figure(image,robust,items,predictions,OUT/'sky'/name/'pillar-grid.png')
        frame=Path(case['frame'])
        ids=np.load(frame/'labels.npz')['face_id'];faces={f['id']:f for f in read(frame/'labels.json')['faces']}
        results[name]={p['label']:dict(prediction=predictions[p['label']],reference=grade(p,ids,faces)) for p in items if predictions[p['label']]}
    save(OUT/'results.json',results)
    for k,v in results.items():print(k,len(v),str(v)[:1200] if 'garden' in k else '')


if __name__=='__main__':main()
