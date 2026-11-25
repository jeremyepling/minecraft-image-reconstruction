"""Evaluation-only checks at the exact points where a wall split sampled depth."""
from collections import Counter

import numpy as np

from benchmark.camera_round import read
from benchmark.count_metrics import reference_planes
from benchmark.surface_metrics import DOWNSAMPLE
from benchmark.surface_reference import recorded


def boundary_truth(case, inventory):
    """Check whether the near sample pairs straddle a real step on the same camera ray."""
    _, ref = recorded(case)
    labels = ref['id_map']
    cells = reference_planes(ref, read(case.path('labels'))['faces'])
    charts = {pid:np.linalg.inv(chart) for pid,chart in ref['charts'].items()}

    def at(y, x, pid):
        return charts[pid] @ np.array([DOWNSAMPLE*(x+.5),DOWNSAMPLE*(y+.5),1.])

    def block_id(y, x, pid):
        point = at(y,x,pid)
        uv = tuple(np.floor(point[:2]/point[2]).astype(int))
        return cells[pid]['faces'].get(uv,{}).get('block_id')

    truth = []
    for candidate in inventory:
        counts = Counter()
        samples = candidate['samples']
        for _,_,yi,xi,yo,xo,_,_ in samples:
            a,b = int(labels[yi,xi]),int(labels[yo,xo])
            if not a or not b:
                counts['background']+=1
                continue
            if a==b:
                counts['same_surface']+=1
                m,n = block_id(yi,xi,a),block_id(yo,xo,b)
                if m and n and m!=n:
                    counts['same_surface_material_change']+=1
                continue
            inside,outside = ref['patches'][a],ref['patches'][b]
            if inside['normal']!=outside['normal'] or inside['offset']==outside['offset']:
                counts['other_boundary']+=1
                continue
            y,x = (yi+yo)/2,(xi+xo)/2
            zi,zo = at(y,x,a)[2],at(y,x,b)[2]
            if min(zi,zo)>0:
                counts['inside_nearer' if zi>zo else 'outside_nearer']+=1
        # A strict majority of all sample pairs must support the same interpretation.
        kind,sign='mixed',0
        for label,value,orientation in (('inside_nearer','parallel step',1),('outside_nearer','parallel step',-1),
                                        ('same_surface','flush wall',0)):
            if counts[label]*2>len(samples):
                kind,sign=value,orientation
                break
        truth.append(dict(parent=candidate['parent'],proposal=candidate['proposal'],kind=kind,sign=sign,
                          material_seam=kind=='flush wall' and counts['same_surface_material_change']*2>len(samples),
                          samples=len(samples),votes=dict(counts)))
    return truth


def grade_votes(truth, events):
    total=Counter(candidates=len(truth))
    for ref,event in zip(truth,events,strict=True):
        if (ref['parent'],ref['proposal'])!=(event['parent'],event['proposal']):
            raise ValueError('Candidate inventory and depth evidence have different order')
        accepted=event['accepted']
        total['accepted']+=int(accepted)
        if ref['kind']=='flush wall':
            total['flush_candidates']+=1
            total['flush_cuts']+=int(accepted)
            if ref['material_seam']:
                total['flush_material_candidates']+=1
                total['flush_material_cuts']+=int(accepted)
        if ref['sign']:
            agrees=np.sign(event['median_step'])==ref['sign']
            total['order_candidates']+=1
            total['order_correct']+=int(agrees)
            total['order_accepted']+=int(accepted)
            total['order_accepted_correct']+=int(accepted and agrees)
    return dict(total)
