"""Does nearby screenshot texture improve block counts on fixed Surface Finder masks?"""
import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageFilter

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_metrics import count_totals, reference_planes
from benchmark.count_round import FINDER, cached_regions
from benchmark.mask_metrics import measure
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE, recorded
from engine.geometry import UNIT, bilinear, project, transform
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_pitch import _aspect
from engine.surface_sample import sample_surface
from runtime.mode import image_only


MARGINS = (0,8,16)
METHODS = ('tight','context8','context16','padding8','padding16')
SETTINGS = dict(margins_native_px=list(MARGINS), maximum_blurred_rgb_step=12, blur_native_px=1,
                maximum_chart_padding_per_side=64, minimum_interpolation_context_taps=4)


def ownership(image, parts):
    sw,sh = parts[0]['support']['size']
    labels = np.zeros((sh,sw),np.int32)
    for i,part in enumerate(parts,1):
        if part['support']['size'] != [sw,sh]:
            raise ValueError('All predicted masks must use a common grid')
        for y,a,b in part['support']['runs']:
            labels[y,a:b] = i
    x = ((2*np.arange(image.width)+1)*sw)//(2*image.width)
    y = ((2*np.arange(image.height)+1)*sh)//(2*image.height)
    return labels[np.ix_(y,x)],x,y


def context_distance(part, index, labels, native_x, native_y, smooth, radius):
    """Native pixels reachable through unclaimed, color-continuous pixels exterior to the tight box."""
    y,a,b = np.asarray(part['support']['runs']).T
    left,right = np.searchsorted(native_x,[a.min(),b.max()])
    top,bottom = np.searchsorted(native_y,[y.min(),y.max()+1])
    x0,y0 = max(0,left-radius),max(0,top-radius)
    x1,y1 = min(labels.shape[1],right+radius),min(labels.shape[0],bottom+radius)
    box = np.s_[y0:y1,x0:x1]
    owner = labels[box]
    allowed = owner==0
    allowed[top-y0:bottom-y0,left-x0:right-x0] = False
    rgb = smooth[box]
    horizontal = np.max(abs(rgb[:,1:]-rgb[:,:-1]),axis=2)<=SETTINGS['maximum_blurred_rgb_step']
    vertical = np.max(abs(rgb[1:]-rgb[:-1]),axis=2)<=SETTINGS['maximum_blurred_rgb_step']
    active = owner==index
    distance = np.full(active.shape,radius+1,np.uint8)
    for n in range(1,radius+1):
        reached = np.zeros_like(active)
        reached[:,1:] |= active[:,:-1] & horizontal
        reached[:,:-1] |= active[:,1:] & horizontal
        reached[1:] |= active[:-1] & vertical
        reached[:-1] |= active[1:] & vertical
        active = reached & allowed & (distance>radius)
        if not active.any():
            break
        distance[active] = n
    return distance, (x0,y0)


def _padding(sample, radius):
    quad = np.asarray(sample['quad'])
    inverse = np.linalg.inv(transform(quad))
    wh = np.asarray(sample['mask'].shape[::-1])
    # Estimate the maximum chart-pixel displacement of a native-pixel step at each corner.
    at = project(inverse,quad)
    dx = abs(project(inverse,quad+[1.,0.])-at)
    dy = abs(project(inverse,quad+[0.,1.])-at)
    maximum = np.max(np.maximum(dx,dy),axis=0)*wh
    return np.ceil(radius*maximum).clip(1,SETTINGS['maximum_chart_padding_per_side']).astype(int)


def extend(sample, distance, offset, radius):
    """Retain original chart samples exactly. Only the optional evidence channel sees new pixels."""
    w,h = sample['mask'].shape[::-1]
    px,py = _padding(sample,radius)
    width,height = w+2*px,h+2*py
    lo = -np.array([px/w,py/h])
    span = np.array([width/w,height/h])
    chart = transform(sample['quad'])
    corners = lo+UNIT*span
    divisor = corners@chart[2,:2] + chart[2,2]
    if not (np.all(divisor>1e-7) or np.all(divisor<-1e-7)):
        return None
    # The pixel centers and scale of the original warp stay exactly fixed.
    u,v = np.meshgrid((np.arange(width)-px+.5)/w,(np.arange(height)-py+.5)/h)
    xy = project(chart,np.stack((u,v),axis=-1))
    if not np.isfinite(xy).all():
        return None
    rgb = np.zeros((height,width,3),np.uint8)
    support = np.zeros((height,width),bool)
    central = np.s_[py:py+h,px:px+w]
    support[central] = sample['mask']
    rgb[central] = sample['rgb']
    x,y = np.floor(xy-.5).astype(np.int64).transpose(2,0,1)
    x -= offset[0]
    y -= offset[1]
    dh,dw = distance.shape
    if dh<2 or dw<2:
        return None
    valid = (x>=0)&(x<dw-1)&(y>=0)&(y<dh-1)&~support
    yy,xx = np.nonzero(valid)
    by,bx = y[yy,xx],x[yy,xx]
    ok = np.maximum.reduce((distance[by,bx],distance[by+1,bx],distance[by,bx+1],distance[by+1,bx+1]))<=radius
    yy,xx = yy[ok],xx[ok]
    if not len(xx):
        return None
    extra = np.zeros_like(support)
    extra[yy,xx] = True
    rgb[extra] = np.rint(bilinear(np.asarray(sample['source_image']),xy[extra])).astype(np.uint8)
    original_aspect = sample.get('normal_aspect') or _aspect(chart,sample['native_size'])
    padded = dict(sample,quad=project(chart,corners).tolist(),rgb=rgb,mask=support,
                  normal_aspect=original_aspect*span[0]/span[1] if original_aspect else None)
    return padded,extra,lo,span


def project_reading(reading, lo, span):
    """Express both repeat counts and joint origin in the original chart's UV coordinates."""
    counts = [None if n is None else float(n/s) for n,s in zip(reading['counts'],span)]
    phase = [None if n is None or p is None else float((p+v*n)%1.)
             for n,p,v in zip(counts,reading['phase'],lo)]
    return dict(counts=counts,phase=phase)


def evaluate(image, parts):
    """Image-only predictions for every arm and surface, before opening recorded geometry."""
    if not parts:
        return []
    labels,native_x,native_y = ownership(image,parts)
    smooth = np.asarray(image.filter(ImageFilter.GaussianBlur(SETTINGS['blur_native_px'])),np.int16)
    output=[]
    for i,part in enumerate(parts,1):
        sample = sample_surface(image,part)
        unpadded = count(sample)
        readings = {name:dict(counts=unpadded['counts'],phase=unpadded['phase']) for name in METHODS}
        evidence = dict(context8=0,context16=0)
        quad = None if sample is None else sample['quad']
        if sample is not None:
            distance,offset = context_distance(part,i,labels,native_x,native_y,smooth,max(MARGINS))
            for radius in MARGINS[1:]:
                if not np.any(distance<=radius):
                    continue
                expanded = extend(sample,distance,offset,radius)
                if expanded is None:
                    continue
                padded,extra,lo,span = expanded
                evidence[f'context{radius}'] = int(extra.sum())
                blank = padded['rgb'].copy()
                blank[extra] = 0
                for name,chart in [(f'padding{radius}',dict(padded,rgb=blank)),
                                   (f'context{radius}',dict(padded,mask=padded['mask']|extra))]:
                    readings[name] = project_reading(count(chart),lo,span)
        for reading in readings.values():
            reading['dimensions'] = dimensions(part,quad,image.size,reading['counts'])
        output.append(dict(id=part['id'],quad=quad,measures=readings,expected_chart=None,evidence=evidence))
    return output


def totals(rows):
    result={}
    for method in METHODS:
        stats=count_totals(rows,method)
        records=[r for picture in rows for r in picture['surfaces'] if r['expected'] is not None]
        stats['both_phases_returned_on_matches']=sum(all(v is not None for v in r['measures'][method]['phase']) for r in records)
        if method in ('context8','context16'):
            stats['extra_chart_samples']=sum(r['evidence'][method] for p in rows for r in p['surfaces'])
            stats['proposals_with_extra_samples']=sum(r['evidence'][method]>0 for p in rows for r in p['surfaces'])
            paired=[]
            for r in records:
                target=r['expected']
                matched=r['measures'][method]['dimensions']['blocks']==target
                blank=r['measures'][method.replace('context','padding')]['dimensions']['blocks']==target
                paired.append((matched,blank))
            stats['exact_gained_from_extra_rgb']=paired.count((True,False))
            stats['exact_lost_from_extra_rgb']=paired.count((False,True))
        result[method]=stats
    return result


def run(out, perimeter_run=None, dev_only=False):
    from engine.surface_boundary_refine import refine, SETTINGS as edge_settings
    frozen=read(FINDER/'manifest.json')
    groups_by_split,suite,fixtures=groups()
    if fixtures!=frozen['fixtures'] or digest(ARCHIVE)!=frozen['reference_sha256'] or digest(FINDER/'cases.json')!=frozen['artifact_sha256']['cases.json']:
        raise ValueError('Surface Finder fixture changed')
    source=read(FINDER/'cases.json')
    lookup={f'{c.floor}/{c.id}':c for cohort in groups_by_split.values() for c in cohort}
    cached={}
    if perimeter_run is not None:
        parent=read(perimeter_run/'manifest.json')
        if parent['fixtures']!=fixtures or parent['selection']['selected']!='color-edges':
            raise ValueError('Expected the color-edges perimeter round on this fixture')
        if digest(perimeter_run/'cases.json')!=parent['artifacts']['cases.json']:
            raise ValueError('Perimeter run cases changed')
        cached={r['key']:r for r in read(perimeter_run/'cases.json')}
    out.mkdir(parents=True,exist_ok=False)
    stamp=revision()
    results=[]
    for row in source:
        if dev_only and row['cohort']!='development':
            continue
        key=row['key']
        case=lookup[key]
        if digest(case.path('image'))!=row['sha256']:
            raise ValueError('Screenshot changed')
        with Image.open(case.path('image')) as src:
            image=src.convert('RGB')
        if perimeter_run is None:
            parts=refine(image,cached_regions(row,frozen))
        else:
            record=cached[key]
            path=perimeter_run/record['predictions']
            if record['sha256']!=row['sha256'] or digest(path)!=parent['artifacts'][record['predictions']]:
                raise ValueError('Per-frame prediction changed')
            with gzip.open(path,'rt',encoding='utf8') as stream:
                parts=json.load(stream)['color-edges']
        surfaces=evaluate(image,parts)
        _,ref=recorded(case)
        metrics,matches=measure(parts,ref,use_mask=True)
        sizes=reference_planes(ref,read(case.path('labels'))['faces'])
        targets=dict(matches)
        for i,reading in enumerate(surfaces):
            reading['expected']=sizes[targets[i]]['size'] if i in targets else None
        results.append(dict(key=key,layout=row['layout'],cohort=row['cohort'],sha256=row['sha256'],
                            eligible=metrics['eligible'],surfaces=surfaces))
        if len(results)%8==0:
            print(len(results),'pictures',flush=True)
    score=dict(all=totals(results))
    for label in ('layout','cohort'):
        score[label]={name:totals([r for r in results if r[label]==name]) for name in sorted({r[label] for r in results})}
    save(out/'cases.json',results)
    save(out/'scores.json',score)
    save(out/'manifest.json',dict(run_id=out.name,code=stamp,version=suite['version'],fixtures=fixtures,
        reference_sha256=digest(ARCHIVE),baseline_run_sha256=digest(FINDER/'manifest.json'),
        frozen_perimeter_manifest_sha256=digest(perimeter_run/'manifest.json') if perimeter_run is not None else None,
        methods=list(METHODS),finder='color-edges',edge_settings=edge_settings,counter='masked-repeat',settings=SETTINGS,
        source_sha256={name:digest(ROOT/name) for name in ('benchmark/perimeter_context_count.py','engine/surface_boundary_refine.py','engine/masked_count.py','engine/surface_sample.py')},
        reproduce='python -m benchmark.perimeter_context_count',
        scope='Outer margin of the native tight RGB crop. Interior gaps and other predicted surfaces excluded. Padding controls have the same chart and no extra RGB. All final counts refer to original support.',
        frames=[{k:r[k] for k in ('key','layout','cohort','sha256')} for r in results],
        artifact_sha256={name:digest(out/name) for name in ('scores.json','cases.json')}))
    print({m:(v['correct'],v['incorrect'],v['missing']) for m,v in score['all'].items()},flush=True)


@image_only()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perimeter-run',type=Path,help='Optional completed perimeter round with verified predictions')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--dev-only',action='store_true')
    args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    run((args.out or ROOT/'runs'/f'count-context-{stamp}-{uuid4().hex[:6]}').resolve(),args.perimeter_run,args.dev_only)


if __name__=='__main__':
    main()
