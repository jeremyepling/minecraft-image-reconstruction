"""Grade frozen wall proposals using native face IDs and captured corner reprojections."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.local_wall_rulers import CONTROLS,WORK,OUT,ROOT,ARMS,read,inputs
from engine.geometry import project, transform

REVIEW=ROOT/'benchmark/reviews/local-wall-rulers-01'


def seam_score(a, footprint, labels, ids):
    """Native point-to-line matching within the third, untouched ordinary tile."""
    H=transform(np.array(a['quad']))
    q=np.array(footprint)
    low=np.maximum(0,np.floor(q.min(0)-4).astype(int))
    high=np.minimum(ids.shape[::-1],np.ceil(q.max(0)+5).astype(int))
    crop=ids[low[1]:high[1],low[0]:high[0]]
    target=np.zeros(crop.shape,np.uint8)
    for axis in (0,1):
        left,right=(crop[:-1],crop[1:]) if axis==0 else (crop[:,:-1],crop[:,1:])
        yy,xx=np.where((left!=right)&(left>0)&(right>0))
        for y,x in zip(yy,xx):
            f,g=[labels['faces'][n-1] for n in (left[y,x],right[y,x])]
            if f['block']==g['block'] or f['normal_world']!=g['normal_world']:
                continue
            if any(labels['blocks'][v['block']-1].get('geometry_kind','full_cube')!='full_cube' for v in (f,g)):
                continue
            n=np.array(f['normal_world'])
            if n[1] or abs(n@(np.array(f['corners_world'][0])-g['corners_world'][0]))>1e-5:
                continue
            target[y,x]=1
    region=np.zeros_like(target)
    cv2.fillConvexPoly(region,np.rint(q-low-.5).astype('i'),1)
    region=cv2.dilate(region,np.ones((3,3),np.uint8)).astype(bool)
    # Exclude openings, nonvertical faces and silhouettes from eligible guessed seams.
    eligible=np.zeros_like(crop,dtype=bool)
    for k in np.unique(crop):
        if k:
            f=labels['faces'][k-1]
            b=labels['blocks'][f['block']-1]
            if f['normal_world'][1]==0 and b.get('geometry_kind','full_cube')=='full_cube':
                eligible|=crop==k
    pred=np.zeros_like(target)
    uv=project(np.linalg.inv(H),q)
    for axis in (0,1):
        lo=np.floor(uv.min(0)-1).astype(int)
        hi=np.ceil(uv.max(0)+1).astype(int)
        for n in range(lo[axis],hi[axis]+1):
            a0,b0=lo.copy(),hi.copy()
            a0[axis]=b0[axis]=n
            pt=np.rint(project(H,[a0,b0])-low-.5).astype('i')
            cv2.line(pred,tuple(pt[0]),tuple(pt[1]),1)
    # Two-pixel Euclidean tolerance; count each side independently.
    gt=(target>0)&region
    pr=(pred>0)&eligible&region
    near_gt=cv2.distanceTransform((~gt).astype('B'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)<=2
    near_pr=cv2.distanceTransform((~pr).astype('B'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)<=2
    return dict(guess_matched=int((pr&near_gt).sum()),guess_unmatched=int((pr&~near_gt).sum()),
                ref_matched=int((gt&near_pr).sum()),ref_unmatched=int((gt&~near_pr).sum()))


def evaluate(a, labels, face_ids):
    quad=np.array(a['quad'])
    t=(np.arange(13)+.5)/13
    xy=project(transform(quad),np.stack(np.meshgrid(t,t),axis=-1)).astype(int)
    y=xy[:,:,1]; x=xy[:,:,0]
    if min(y.min(),x.min())<0 or y.max()>=face_ids.shape[0] or x.max()>=face_ids.shape[1]:
        return dict(unit=False,reason='clipped candidate')
    ident,n=Counter(face_ids[y,x].ravel()).most_common(1)[0]
    out=dict(unit=False,owner=float(n/169))
    if not ident:
        return dict(out,reason='background')
    face=labels['faces'][ident-1]
    block=labels['blocks'][face['block']-1]
    full=block.get('geometry_kind','full_cube')=='full_cube'
    q=project(np.array(labels['camera']['world_to_pixel_homogeneous']),face['corners_world'])
    assert np.max(abs(q-np.array(face['corners_px'])))<1e-3
    err=min(max(np.linalg.norm(quad-np.roll(order,k,axis=0),axis=1)) for order in [q,q[::-1]] for k in range(4))
    return dict(out,unit=bool(full and n>=.9*169 and err<=2.),error_px=float(err),full=full,
                block=block['block_id'],geometry_kind=block.get('geometry_kind','full_cube'))


def main():
    REVIEW.mkdir(exist_ok=True,parents=True)
    frozen=read(CONTROLS)['pilot']['frames']
    cases=[]
    from benchmark.full_block_lattice import atlas
    for f in frozen:
        pred=read(OUT/(f['id']+'.json'))
        source=f['reference']
        if 'labels.npz' in source:
            labels=read(WORK/source['labels.json']['path'])
            ids=np.load(WORK/source['labels.npz']['path'])['face_id']
        else:
            e=next(e for e in fixed.fixed_frames() if e['id']==f['id'])
            labels,ids,_=fixed.fixed_reference(e)
        for r in pred['records']:
            exact=[evaluate(o,labels,ids)['unit'] for o in r['options']]
            r['finite_exact_options']=sum(exact)
            r['baseline_corner_exact']=exact[0]
            for a in r['arms'].values():
                a['audit']=evaluate(a,labels,ids)
                a['audit']['sprite_eligible_for_owner']=a['audit'].get('block') in atlas().ids[atlas().labels.index(a['texture'])]
                a['audit']['seams']=seam_score(a,r['options'][0]['quad'],labels,ids)
        cases.append(pred)
        if f['id'].startswith(('test-castle-02','test-walled-garden','validation-harbor-03')):
            image=Image.open(WORK/f['rgb']['path']).convert('RGB')
            pen=ImageDraw.Draw(image)
            for r in pred['records']:
                a=r['arms']['native_mixed']; q=[tuple(pt) for pt in a['quad']]
                pen.line(q+[q[0]],fill=(79,238,149) if a['accepted'] else (239,97,82),width=3)
            image.thumbnail((900,650)); image.save(REVIEW/(f['id']+'.jpg'),quality=91)
    summary={}
    for group in ['development','transfer','all','fixed-104','visible-builds-v3','decorative-builds-v1']:
        rows=[c for c in cases if group=='all' or c['dataset']==group or
              (group in ('development','transfer') and (c['split']=='train')==(group=='development'))]
        counts={}
        for arm in [*ARMS,'native_mixed']:
            allrows=[r['arms'][arm] for c in rows for r in c['records']]
            accept=[a for a in allrows if a['accepted']]
            counts[arm]=dict(frames=len(rows),tested=len(allrows),accepted=len(accept),unit=sum(a['audit']['unit'] for a in accept),
                unit_proposed=sum(a['audit']['unit'] for a in allrows),wrong=len(accept)-sum(a['audit']['unit'] for a in accept),
                alias_selected=sum(a['pitch']!=1 for a in allrows),alias_accepted=sum(a['pitch']!=1 for a in accept),
                partial_owner=sum(a['audit'].get('full') is False for a in accept),
                mixed_sprites=sum(len(set(a['fitting_textures']))>1 for a in accept),
                sprite_matches_owner=sum(a['audit']['sprite_eligible_for_owner'] for a in accept))
        summary[group]=counts
    (REVIEW/'cases.json').write_text(json.dumps(cases,separators=(',',':')))
    (REVIEW/'stage.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['all']))


if __name__=='__main__':
    main()
