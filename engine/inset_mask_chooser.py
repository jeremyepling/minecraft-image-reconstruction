"""Optional surface partition with no plane depth or placement event."""
from collections import defaultdict

import numpy as np
from PIL import ImageFilter

from engine.inset_mask_proposals import proposals
from engine.pillar_seams import native_mask
from engine.plane_consensus import with_mask


# The accepted leaf of the frozen development tree. All other leaves fall
# below its 0.90 acceptance threshold; the held-out shapes did not tune it.
MIN_JAMB_IOU = 0.6846031124048366


def _select(image, original_parts, camera, surface_parts, confirmed, native):
    used={(e['host'],e['side'],e['sill']) for e in confirmed}
    groups=defaultdict(list)
    for row in proposals(image,original_parts,camera,masks=native):
        key=tuple(row[n] for n in ('host','side','sill'))
        f=row['features']
        if (key in used or row['strict'] or f['rim_min']<.55
                or f['host_fraction']<.90):
            continue
        if f['side_iou']>MIN_JAMB_IOU:
            groups[key].append((.95,f['depth_margin'],row))
    if not groups:
        return surface_parts,[]
    originals={id(p):mask for p,mask in zip(original_parts,native)}
    masks={p['id']:originals[id(p)] if id(p) in originals else native_mask(p,image.size) for p in surface_parts}
    areas={k:int(v.sum()) for k,v in masks.items()}
    additions=defaultdict(list)
    proposals_by_group=[max(group,key=lambda t:t[:2]) for group in groups.values()]
    for probability,margin,row in sorted(proposals_by_group,key=lambda t:t[:2],reverse=True):
        host=row['host']
        if host not in masks:
            continue
        x,y,r,b=row['roi_xyxy']
        selected=row['mask'] & masks[host][y:b,x:r]
        area=int(selected.sum())
        if area<64 or area/max(1,row['mask'].sum())<.90:
            continue
        threshold=.10
        if any(np.count_nonzero(m[y:b,x:r]&selected)>threshold*min(area,areas[name])
               for name,m in masks.items() if name!=host):
            continue
        full=np.zeros_like(masks[host])
        full[y:b,x:r]=selected
        ident=f'{host}-learned-outline-{len(additions[host])}'
        masks[ident]=full
        areas[ident]=area
        areas[host]-=area
        masks[host]=masks[host]&~full
        record=dict(output=ident,host=host,side=row['side'],sill=row['sill'],evidence=row['features'],
                    probability=probability,pixels=area, photo_grid_level=row['level']+row['depth'])
        additions[host].append(record)
    if not additions:
        return surface_parts,[]
    gray=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    result=[]
    logs=[]
    for p in surface_parts:
        key=p['id']
        if key not in additions:
            result.append(p)
            continue
        parent={k:v for k,v in p.items() if k not in ('source_crop','context_crop','corner_junctions','outline','sampling_chart')}
        for e in additions[key]:
            kid=dict(parent,id=e['output'],parent_id=key)
            for k in ('inset_back_plane','visible_inset','diagonal_join'):
                kid.pop(k,None)
            result.append(with_mask(kid,masks[e['output']],image.size,gray,np.ones(2)))
            logs.append(e)
        result.append(with_mask(parent,masks[key],image.size,gray,np.ones(2)))
    return result,logs


def possible(size, parts):
    """Cheap bounds test for two nearby return masks inside a large wall."""
    boxes=[]
    for part in parts:
        support=part.get('support')
        if not support or not support.get('runs') or part.get('normal_camera') is None:
            continue
        runs=np.asarray(support['runs'])
        sy=size[1]/support['size'][1]
        sx=size[0]/support['size'][0]
        boxes.append((runs[:,1].min()*sx,runs[:,0].min()*sy,runs[:,2].max()*sx,(runs[:,0].max()+1)*sy))
    for i,(hx,hy,hr,hb) in enumerate(boxes):
        if (hr-hx+1)*(hb-hy+1)<1200:
            continue
        # Allow one extra pixel for native resampling of the support grid.
        nearby=[box for j,box in enumerate(boxes) if i!=j and box[0]>=hx-14 and box[1]>=hy-14
                and box[2]<=hr+14 and box[3]<=hb+14]
        for n,(x,y,r,b) in enumerate(nearby):
            if any(max(x,X)-min(r,R)<=16 and max(y,Y)-min(b,B)<=16 for X,Y,R,B in nearby[:n]):
                return True
    return False


def supplemental(image, original_parts, camera, baseline_parts, strict_events, *, atlas=None):
    """Partition only visible support. The returned events belong to the outline stage."""
    if not camera or camera.get('P') is None or not possible(image.size,original_parts):
        return baseline_parts,[]
    from engine.inset_outline_count_guard import apply
    native=[native_mask(p,image.size) for p in original_parts]
    provisional,events=_select(image,original_parts,camera,baseline_parts,strict_events,native)
    final,accepted=apply(image,baseline_parts,provisional,events,camera,atlas)
    diagnostics=[dict(host=e['host'],side=e['side'],sill=e['sill'],output=e['output'],pixels=e['pixels'],
                      measured_dimensions=e['photo_grid_span'], depth='unresolved') for e in accepted]
    return final,diagnostics
