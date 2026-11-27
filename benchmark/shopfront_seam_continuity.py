"""Test image-only seam continuity on jagged neighboring surface masks."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import ImageFilter

from benchmark import facade_inset_round as suite
from benchmark.texture_camera_eval import read, save
from engine import diagonal_texture_join as texture, guarded_wall_outlines as walls, inset_returns
from engine.pillar_seams import native_mask, orientation
from engine.plane_consensus import with_mask
from engine.surface_perimeter import _native_alpha


def contact(left, right, size, smooth):
    """Measure contrast across the actual boundary against interior control strips."""
    a, b = left[1], right[1]
    box = (max(0, max(a[0], b[0])-12), max(0, max(a[1], b[1])-12),
           min(size[0], min(a[2], b[2])+12), min(size[1], min(a[3], b[3])+12))
    if box[2]-box[0] < 16 or box[3]-box[1] < 16:
        return
    first, second = (walls._on_box(patch, box) for patch in (left, right))
    if np.count_nonzero(first & second) > .10 * min(left[0].sum(), right[0].sum()):
        return
    a, b = first & ~second, second & ~first
    h, w = a.shape
    rgb = smooth[box[1]:box[3], box[0]:box[2]]
    xy, across, internal = [], [], []
    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
        for gap in (1,2,3,4):
            y, x = np.nonzero(a & np.roll(b, (-gap*dy,-gap*dx), axis=(0,1)))
            keep = (x>=9)&(x<w-9)&(y>=9)&(y<h-9)
            y, x = y[keep], x[keep]
            # Use the nearest gap only; exclude the one-pixel perimeter and
            # compare equal distances inside the two participating regions.
            inside = ~a[y+dy,x+dx] & a[y-5*dy,x-5*dx] & b[y+(gap+5)*dy,x+(gap+5)*dx]
            if gap>1:
                inside &= ~b[y+(gap-1)*dy,x+(gap-1)*dx]
            y, x = y[inside], x[inside]
            if not len(y):
                continue
            lc, rc = rgb[y-dy,x-dx], rgb[y+(gap+1)*dy,x+(gap+1)*dx]
            xy.append(np.stack((x+box[0],y+box[1]),1))
            across.extend(np.abs(lc-rc).mean(1))
            internal.extend((np.abs(lc-rgb[y-5*dy,x-5*dx]).mean(1) +
                             np.abs(rc-rgb[y+(gap+5)*dy,x+(gap+5)*dx]).mean(1))/2)
    if not xy:
        return
    points = np.unique(np.concatenate(xy), axis=0)
    if len(points)<18 or max(np.ptp(points,axis=0))<16:
        return
    uv = points - points.mean(0)
    _, _, basis = np.linalg.svd(uv,full_matrices=False)
    bend = np.quantile(abs(uv @ basis[1]),.9)
    if bend < 3.5:
        return
    return dict(contact_pixels=len(points),bend_px=round(float(bend),2),
                cross_q75=round(float(np.quantile(across,.75)),2),
                within_q75=round(float(np.quantile(internal,.75)),2))


def predict(image, parts, camera, *, protect_junctions=False, phase_only=False):
    rgb = np.asarray(image)
    smooth = np.asarray(image.filter(ImageFilter.GaussianBlur(1.2)), np.int16)
    patches = {}
    for i,p in enumerate(parts):
        if not p.get('support') or p.get('normal_camera') is None:
            continue
        patch = _native_alpha(p['support'],image.size)
        if patch:
            a,box = patch
            mask = np.asarray(a,bool)
            if mask.sum()>=64:
                patches[i] = mask,box
    geom = None
    if camera and camera.get('P') is not None:
        matrix, rot = orientation(camera,image.size)
        inv = np.linalg.inv(matrix[:,:3])
        eye = -inv @ matrix[:,3]
        ray = inv @ [image.width/2,.72*image.height,1.]
        if abs(ray[1])>1e-6 and ray[1]*eye[1]<0:
            geom = inv, eye, eye - ray*(eye[1]/ray[1]), rot
    fits = {}
    proposals = []
    for i,j in combinations(patches,2):
        if protect_junctions and (parts[i].get('junction_split') or parts[j].get('junction_split')):
            continue
        if np.dot(parts[i]['normal_camera'],parts[j]['normal_camera'])<.965:
            continue
        obs=contact(patches[i],patches[j],image.size,smooth)
        if not obs:
            continue
        smooth_seam = obs['cross_q75']<=5 and obs['cross_q75']<=obs['within_q75']+2
        reason = 'smooth' if smooth_seam and not phase_only else 'discontinuous'
        if reason=='discontinuous' and geom is not None:
            inv, eye, center, rot = geom
            axis = int(np.argmax(abs(rot.T @ parts[i]['normal_camera'])))
            if axis==int(np.argmax(abs(rot.T @ parts[j]['normal_camera']))):
                for k in (i,j):
                    if (k,axis) not in fits:
                        level=round(center[axis])
                        fits[k,axis]=texture.texture_fit(rgb,*patches[k],axis,eye,inv,range(level-24,level+25))
                a,b = fits[i,axis],fits[j,axis]
                if a and b and a['level']==b['level'] and min(a['margin'],b['margin'])>=.08 and min(a['corr'],b['corr'])>=.65:
                    shared=a['valid'] & b['valid']
                    if shared.sum()>=28:
                        av,bv=a['template'][shared],b['template'][shared]
                        if texture.correlation(av,bv)>=.72 and abs(av.mean()-bv.mean())<=8:
                            reason='repeating'
        proposals.append(dict(a=i,b=j,reason=reason,**obs))
    # Refuse multi-way contacts without direct evidence for the outer two pieces.
    used=Counter(i for p in proposals if p['reason']!='discontinuous' for i in (p['a'],p['b']))
    changed=[]; retired=set()
    gray=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    for p in proposals:
        i,j=p['a'],p['b']
        accepted=p['reason']!='discontinuous' and used[i]==used[j]==1
        p.update(inputs=[parts[i]['id'],parts[j]['id']],accepted=accepted)
        if not accepted:
            continue
        base=max((i,j),key=lambda k:patches[k][0].sum())
        fields={k:v for k,v in parts[base].items() if k not in ('context_crop','source_crop','corner_junctions','junction_split')}
        union=native_mask(parts[i],image.size)|native_mask(parts[j],image.size)
        item=with_mask(fields,union,image.size,gray,np.ones(2))
        changed.append(dict(item,id=fields['id']+'-seam-union'))
        retired.update((i,j))
    return [p for i,p in enumerate(parts) if i not in retired]+changed,proposals


def grade(task):
    job,protect,phase_only=task
    s,f,_,_=job
    image,base,_,camera=suite.ordinary_inputs(*job)
    parts,strict,phase,outlines,_=inset_returns.detect(image,base,'26.3',camera=camera)
    expected=next(r for r in read(suite.ROOT/'benchmark/reviews/facade-inset-round-01/combined.json')['rows'] if r['id']==f['id'])
    assert [len(e) for e in (strict,phase,outlines)] == [expected[k] for k in ('returns','texture','outlines')]
    changed,pairs=predict(image,parts,camera,protect_junctions=protect,phase_only=phase_only)
    row=dict(id=f['id'],split=s,rgb_sha256=f['files']['rgb.png'],pairs=pairs)
    truth=suite.visible_map(s,f)
    row['surface_map_sha256']=sha256(truth.astype('<i4').tobytes()).hexdigest()
    # Fixed reader-selected issue regions. These are examined only after prediction.
    roi=None
    if f['id']=='legacy-light_gray_concrete-dev-shopfront-plaster-low-left':
        roi=np.zeros(truth.shape,bool)
        roi[490:545,790:850]=True
    elif 'holdout-registry' in f['id']:
        roi=truth==(10 if f['id'].endswith('high-right') else 6)
    if roi is not None:
        active=np.zeros(truth.shape,bool)
        hits={}
        for part in parts:
            current=native_mask(part,image.size)
            pixels=int((roi & current).sum())
            if pixels:
                hits[part['id']]=pixels
            active|=current
        row['inspected_region']=dict(visible_pixels=int(roi.sum()),covered=int((roi & active).sum()),
                                     contributors=hits,surface_ids=np.unique(truth[roi]).tolist())
    before=suite.surface_stats(parts,truth,image.size)
    after=suite.surface_stats(changed,truth,image.size) if any(p['accepted'] for p in pairs) else before
    row['baseline_masks']=len(parts)
    for p in pairs:
        sid=[]
        for i in (p.pop('a'),p.pop('b')):
            mask=native_mask(parts[i],image.size)
            hits=np.bincount(truth[mask]); k=int(hits.argmax())
            sid.append([k,round(float(hits[k]/hits.sum()),4)])
        p.update(dominant=sid,same_plane=sid[0][0]!=0 and sid[0][0]==sid[1][0] and min(v[1] for v in sid)>=.8)
    row.update(gained=sorted(set(after['matched'])-set(before['matched'])),
               lost=sorted(set(before['matched'])-set(after['matched'])),
               paired_pixel_delta=after['paired_pixels']-before['paired_pixels'],
               unpaired_delta=after['unmatched_predictions']-before['unmatched_predictions'],
               background_delta=after['background_pixels']-before['background_pixels'])
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--protect-junctions',action='store_true')
    parser.add_argument('--phase-only',action='store_true')
    parser.add_argument('--all-cohorts',action='store_true')
    parser.add_argument('--out',type=Path,default=Path('benchmark/reviews/shopfront-seam-continuity-01'))
    args=parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        splits=[*suite.framed.SIZES,'shelter'] if args.all_cohorts else suite.framed.EVALUATION
        rows=list(pool.map(grade,((j,args.protect_junctions,args.phase_only) for j in suite.jobs(splits))))
    assert len(rows)==(204 if args.all_cohorts else 104)
    args.out.mkdir(parents=True,exist_ok=True)
    name='phase-only.json' if args.phase_only else 'guarded.json' if args.protect_junctions else 'cases.json'
    save(args.out/name,dict(source='a95816d',protect_junctions=args.protect_junctions,phase_only=args.phase_only,rows=rows))
    for name,subset in [('shopfront',[r for r in rows if 'shopfront' in r['id']]),('others',[r for r in rows if 'shopfront' not in r['id']])]:
        pairs=[p for r in subset for p in r['pairs']]
        print(name,len(subset),len(pairs),Counter((p['reason'],p['accepted'],p['same_plane']) for p in pairs))
        print({k:sum(len(r[k]) if isinstance(r[k],list) else r[k] for r in subset) for k in ('gained','lost','paired_pixel_delta','unpaired_delta','background_delta')})


if __name__=='__main__':
    main()
