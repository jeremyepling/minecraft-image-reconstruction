"""Level supported near-horizontal lines in the native crop and check visible wall rows."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from benchmark.expanded_baseline import ROOT, WORKTREES, read, save, digest
from benchmark.relative_face_round import SHARED
from benchmark.raw_crop_pitch import measure
from engine.geometry import transform, project

OUT = ROOT / 'runs/castle-feedback-01/rows'


def row_slope(rgba):
    rgb, mask = rgba[:, :, :3], rgba[:, :, 3] > 0
    if min(mask.shape) < 8:
        return None
    segments = cv2.createLineSegmentDetector().detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))[0]
    if segments is None:
        return None
    lines = segments.reshape(-1,2,2)
    delta = lines[:,1]-lines[:,0]
    length = np.linalg.norm(delta,axis=1)
    xy = np.rint(lines[:,:1] + delta[:,None]*np.linspace(.1,.9,7)[None,:,None]).astype(int)
    interior = cv2.erode(mask.astype('uint8'),np.ones((3,3),'uint8'))
    support = interior[xy[:,:,1].clip(0,mask.shape[0]-1),xy[:,:,0].clip(0,mask.shape[1]-1)].mean(1)
    keep = (np.abs(delta[:,1]) < .45*np.abs(delta[:,0])) & (length>=6) & (support>=.85)
    if keep.sum()<3:
        return None
    slopes = delta[keep,1]/delta[keep,0]
    weights = np.minimum(32,length[keep])
    inliers = abs(slopes[:,None]-slopes)<.07
    best = inliers[np.argmax(inliers@weights)]
    if best.sum()<3 or weights[best].sum()<.55*weights.sum():
        return None
    inlier_lines=lines[keep][best]
    return float(np.average(slopes[best],weights=weights[best])), inlier_lines.round(1).tolist()


def correction(image, slope):
    h,w=image.shape[:2]
    offset=max(slope*w,0)
    mat=np.array([[1.,0.,0.],[-slope,1.,offset]])
    out=cv2.warpAffine(image,mat,(w,h+int(np.ceil(abs(slope)*w))),flags=cv2.INTER_LINEAR)
    out[:,:,3]=255*(out[:,:,3]==255)
    return out


def truth_edges(native, faces, box, rgba):
    x,y,r,b=box
    idx,n=np.unique(native[y:b,x:r][rgba[:,:,3]>0],return_counts=True)
    planes={}
    for i,weight in zip(idx,n):
        if i and i in faces:
            sid=faces[i]['surface_id']
            planes[sid]=planes.get(sid,0)+weight
    if not planes:
        return []
    sid=max(planes,key=planes.get)
    if planes[sid]<.6*n.sum():
        return []
    edges=[]
    columns=[]
    for i,weight in zip(idx,n):
        f=faces.get(i)
        if not f or f['surface_id']!=sid or weight<12 or f['direction'] not in ('north','south','west','east'):
            continue
        world=np.array(f['corners_world']);px=np.array(f['corners_px'])
        if np.ptp(world[:,1])!=1:
            continue
        for level in [world[:,1].min(),world[:,1].max()]:
            edge=px[world[:,1]==level]
            if len(edge)==2:
                edges.append(edge)
        for i in range(4):
            if world[i,1]!=world[(i+1)%4,1]:
                columns.append(px[[i,(i+1)%4]])
    return [np.asarray(edges),np.asarray(columns)] if edges and columns else []


def audit_angle(edges, H):
    if len(edges)<4:
        return None
    d=np.diff(project(H,edges.reshape(-1,2)).reshape(-1,2,2),axis=1)[:,0]
    return round(float(np.median(np.degrees(np.arctan2(np.abs(d[:,1]),np.abs(d[:,0]))))),2)


def pixel_period(grid,H):
    if not grid:
        return [None,None]
    across,down=[np.diff(project(H,edges.reshape(-1,2)).reshape(-1,2,2), axis=1)[:,0,:] for edges in grid]
    # Scanlines run horizontally/vertically. Their spacing is the distance
    # between successive slanted parallel lines, measured along the scan axis.
    u,v=np.median(across*np.sign(across[:,0:1]),axis=0),np.median(down*np.sign(down[:,1:2]),axis=0)
    area=abs(u[0]*v[1]-u[1]*v[0])
    return [float(area/max(abs(v[1]),1e-6)),float(area/max(abs(u[0]),1e-6))]


def main():
    report={}
    annotations=read(ROOT/'benchmark/reviews/castle-feedback-01/annotations.json')['surfaces']
    for case in read(SHARED/'fixtures.json')['cases']:
        ident=case['id']; dest=OUT/ident; dest.mkdir(exist_ok=True,parents=True)
        source=WORKTREES/'floating-build-local-grid/runs/floating-build-grid-01/sky'/ident/'source-handoff'
        handoffs='/'.join(digest(source/name) for name in ['surfaces.json','counts.json'])
        surfaces=sorted(read(source/'surfaces.json')['planes'],key=lambda r:-r['support_area'])
        counts={r['id']:r for r in read(source/'counts.json')['surfaces']}
        ref=Path(case['frame']);native=np.load(ref/'labels.npz')['face_id']
        faces={f['id']:f for f in read(ref/'labels.json')['faces']}
        rows={}
        for ix,p in enumerate(surfaces,1):
            label=f'S{ix:02}'; crop=p.get('source_crop'); count=counts.get(p['id'],{})
            if not crop:continue
            for ann in annotations:
                if ann['frame']==ident and ann['surface']==label:
                    assert ann['source_sha256']==digest(Path(case['variants']['sky']['image'])) and ann['mask']==p['id'] and ann['handoffs']==handoffs
            rgba=np.asarray(Image.open(source/crop['path']).convert('RGBA'))
            fit=row_slope(rgba)
            slope,lines=fit if fit else (None,[])
            grid=truth_edges(native,faces,crop['xyxy'],rgba)
            edges=grid[0] if grid else []
            raw=audit_angle(edges,np.eye(3));old=None; old_period=[]; old_reference=[]
            if count.get('quad'):
                w,h=Image.open(source/count['sample']).size
                old_H=np.diag([w,h,1])@np.linalg.inv(transform(count['quad']))
                old=audit_angle(edges,old_H)
                old_period=[a['lag'] if a else None for a in measure(source/count['sample'])[0]]
                old_reference=pixel_period(grid,old_H)
            changed=raw;period=None; corrected_H=np.eye(3)
            if slope is not None:
                corrected=correction(rgba,slope)
                image=dest/(label+'.png');Image.fromarray(corrected).save(image)
                corrected_H=np.array([[1,0,0],[-slope,1,0],[0,0,1]])
                changed=audit_angle(edges,corrected_H)
                period=[a['lag'] if a else None for a in measure(image)[0]]
            rows[label]=dict(mask=p['id'],slope=slope,lines=lines,raw_angle=raw,old_angle=old,
                             candidate_angle=changed,period=period,reference_period=pixel_period(grid,corrected_H),
                             old_period=old_period,old_reference=old_reference)
        report[ident]=rows
    save(OUT/'measurements.json',report)


if __name__=='__main__':
    main()
