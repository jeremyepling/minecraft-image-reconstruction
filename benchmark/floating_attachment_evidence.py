"""Frozen source attachment geometry and texture rules from f088877."""
from collections import Counter, defaultdict
import numpy as np
from engine.geometry import bilinear, project

TEXELS = np.stack(np.meshgrid((np.arange(12)+.5)/12,(np.arange(12)+.5)/12),-1)


def uv_cells(chart):
    return np.array([[c["u"],c["v"]] for c in chart["cells"]],int)


def candidate(parent, child):
    """Fit a shared integer phase over this observed fragment, checking both chart carriers."""
    cuv,puv = uv_cells(child),uv_cells(parent)
    if (parent["axes"]!=child["axes"] or parent["normal_axis"]!=child["normal_axis"]
            or min(len(cuv),len(puv))<4 or np.any(np.ptp(cuv,axis=0)<1)):
        return
    h,g = np.asarray(parent["H"]),np.asarray(child["H"])
    pos = project(np.linalg.solve(h,g),cuv+.5)
    shift = np.rint(np.median(pos-cuv-.5,axis=0)).astype(int)
    mapped = cuv+shift
    err = np.abs(pos-mapped-.5).max(1)
    radius = float(np.quantile(err,.9))
    if radius>.30 or err.max()>.45:
        return
    child_keys,parent_keys = set(map(tuple,mapped)),set(map(tuple,puv))
    if len(child_keys & parent_keys)>.10*len(child_keys):
        return
    touching = sum((x+dx,y+dy) in parent_keys for x,y in child_keys for dx,dy in ((0,1),(0,-1),(1,0),(-1,0)))
    if touching<2:
        return
    return dict(shift=shift.tolist(),p90=radius,max_error=float(err.max()),contacts=touching)


def signatures(chart, rgb):
    points = project(np.array(chart["H"]),uv_cells(chart)[:,None,None,:]+TEXELS)
    tiles = bilinear(rgb,points).astype("f")
    gray = tiles.mean(3)
    vertical = np.abs(np.diff(gray,axis=2)).mean((1,2))
    horizontal = np.abs(np.diff(gray,axis=1)).mean((1,2))
    ratio = vertical / np.maximum(horizontal,1.)
    templates = gray.reshape((len(gray),-1))
    templates -= templates.mean(1,keepdims=True)
    templates /= np.maximum(np.linalg.norm(templates,axis=1,keepdims=True),1e-5)
    return tiles.mean((1,2)), templates, ratio


def texture_depth(parent,child,shift,rgb,allow_grain=True):
    """Only a repeated appearance on a corresponding source row can transfer a known relief step."""
    puv,cuv = uv_cells(parent),uv_cells(child)+shift
    heights = next(i for i,a in enumerate(parent["axes"]) if abs(a)==2)
    pm,pt,pv = signatures(parent,rgb)
    cm,ct,cv = signatures(child,rgb)
    applied = {(c["u"],c["v"]):c.get("normal_offset",0) for c in parent.get("export_cells",parent["cells"])}
    rows, result = defaultdict(list),[]
    for i,uv in enumerate(cuv):
        row = int(uv[heights])
        candidates = np.flatnonzero(puv[:,heights]==row)
        hits = []
        for j in candidates:
            key = tuple(puv[j])
            if key not in applied:
                continue
            color = np.linalg.norm(pm[j]-cm[i])
            correlation = float(pt[j]@ct[i])
            # Strong vertical grain is allowed when the thin line falls in another texel.
            grain = cv[i]>=1.55 and pv[j]>=1.55 and color<15
            if color<40 and (correlation>=.50 or (allow_grain and grain)):
                hits.append((correlation-color/160+.15*(allow_grain and grain),int(applied[key]),j,
                             correlation,float(color),bool(grain)))
        if hits:
            _,offset,j,correlation,color,grain = max(hits)
            rows[row].append(offset)
            result.append(dict(child=i,parent=int(j),offset=offset,correlation=correlation,
                               color_distance=color,strong_grain=grain))
    depth = []
    for uv in cuv:
        votes = Counter(rows[int(uv[heights])]).most_common()
        depth.append(votes[0][0] if votes and votes[0][1]>.5*sum(n for _,n in votes) else 0)
    return len(result)>=4 and len(rows)>=2,depth,result
