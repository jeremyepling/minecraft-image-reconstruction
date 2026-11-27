"""Replay additional return proposals admitted by the trained layout-only tree."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import ImageFilter

from benchmark.inset_decision_model import probability
from benchmark.inset_return_grid import refine as strict_refine
from benchmark.inset_return_proposals import proposals
from benchmark.texture_camera_eval import read
from engine.pillar_seams import native_mask
from engine.plane_consensus import with_mask


@lru_cache(maxsize=1)
def model():
    return read(Path(__file__).parent/'reviews/facade-inset-round-01/model.json')


def refine(image,parts,camera):
    strict,events=strict_refine(image,parts,camera)
    groups=defaultdict(list)
    for p in proposals(image,parts,camera):
        groups[(p['host'],p['side'],p['sill'])].append(p)
    selected=defaultdict(list)
    trained=model()
    for group in groups.values():
        if any(row['strict'] for row in group):
            continue
        row=max(group,key=lambda r:(probability(trained['tree'],r['features']),r['features']['depth_margin']))
        if probability(trained['tree'],row['features'])>=trained['layout_only_threshold']:
            selected[row['host']].append(row)
    if not selected:
        return strict,events
    smooth=np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.9)),np.float32)[...,None]
    result=[]
    events=list(events)
    for part in strict:
        if part['id'] not in selected:
            result.append(part)
            continue
        remaining=native_mask(part,image.size)
        parent={k:v for k,v in part.items() if k not in ('source_crop','context_crop','corner_junctions','outline','sampling_chart')}
        children=[]
        for row in selected[part['id']]:
            x,y,r,b=row['roi_xyxy']
            mask=np.zeros_like(remaining)
            mask[y:b,x:r]=row['mask']&remaining[y:b,x:r]
            if mask.sum()<64:
                continue
            key=part['id']+'-tree-back-'+str(len(children))
            children.append(with_mask(dict(parent,id=key,parent_id=part['id']),mask,image.size,smooth,np.ones(2)))
            remaining&=~mask
            events.append(dict(host=part['id'],side=row['side'],sill=row['sill'],output=key,
                               depth=row['depth'],level=row['level'],rim=row['rim'],tree=True))
        result.extend(children)
        result.append(with_mask(parent,remaining,image.size,smooth,np.ones(2)) if children else part)
    return result,events
