"""Optional surface partition with no plane depth or placement event."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import ImageFilter

from benchmark.inset_return_proposals import proposals
from benchmark.texture_camera_eval import read
from engine.pillar_seams import native_mask
from engine.plane_consensus import with_mask


@lru_cache(maxsize=1)
def settings():
    return read(Path(__file__).parent/'reviews/inset-mask-chooser-01/model.json')


def _probability(tree,features):
    while 'feature' in tree:
        tree=tree['low'] if features[tree['feature']]<=tree['threshold'] else tree['high']
    return tree['probability']


def refine(image, original_parts, camera, surface_parts, confirmed, *, learned=True):
    s=settings()
    used={(e['host'],e['side'],e['sill']) for e in confirmed}
    groups=defaultdict(list)
    for row in proposals(image,original_parts,camera):
        key=tuple(row[n] for n in ('host','side','sill'))
        f=row['features']
        if (key in used or row['strict'] or f['rim_min']<s['min_rim_side']
                or f['host_fraction']<s['min_host_share']):
            continue
        prob=_probability(s['tree'],f) if learned else 1.
        if prob>=s['probability']:
            groups[key].append((prob,f['depth_margin'],row))
    if not groups:
        return surface_parts,[]
    masks={p['id']:native_mask(p,image.size) for p in surface_parts}
    additions=defaultdict(list)
    proposals_by_group=[max(group,key=lambda t:t[:2]) for group in groups.values()]
    for probability,margin,row in sorted(proposals_by_group,key=lambda t:t[:2],reverse=True):
        host=row['host']
        if host not in masks:
            continue
        x,y,r,b=row['roi_xyxy']
        selected=row['mask'] & masks[host][y:b,x:r]
        area=int(selected.sum())
        if area<64 or area/max(1,row['mask'].sum())<s['min_remaining_share']:
            continue
        threshold=s['max_other_foreground_overlap']
        if any(np.count_nonzero(m[y:b,x:r]&selected)>threshold*min(area,m.sum())
               for name,m in masks.items() if name!=host):
            continue
        full=np.zeros_like(masks[host])
        full[y:b,x:r]=selected
        ident=f'{host}-learned-outline-{len(additions[host])}'
        masks[ident]=full
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
