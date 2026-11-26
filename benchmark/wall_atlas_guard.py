"""Require two independent textured ground contacts with compatible camera height and phase."""
import numpy as np

from benchmark.wall_atlas_consensus import _atlas, _floor_anchors, _grid, _joint_optics
from benchmark.wall_texture_camera import _abstain, _line_data, _result


def _shade(rgb, anchor):
    h,w = rgb.shape[:2]
    x,y = np.rint(anchor[2]).astype(int)
    if not 8 <= x < w-8 or not 4 <= y < h-40:
        return False
    near = rgb[y+3:y+9,x-3:x+4].mean(2)
    far = rgb[y+25:y+40,x-3:x+4].mean(2)
    return np.median(far)-np.median(near) > 11


def predict(image, version="26.3"):
    atlas = _atlas(version)
    if atlas is None:
        return _abstain("atlas_unavailable")
    data = _line_data(image)
    if data is None:
        return _abstain("missing_lines")
    rgb,mask,seg,horizon = data
    M = _joint_optics(seg,horizon,image.size)
    if M is None:
        return _abstain("no_joint_manhattan_fit")
    anchors = [a for a in _floor_anchors(rgb,mask,seg,M) if _shade(rgb,a)]
    if len(anchors)<2:
        return _abstain("two_ground_contacts_required", contacts=len(anchors))
    options = []
    for a in anchors:
        candidates = _grid(rgb,mask,M,a,np.geomspace(.8,21,90),atlas)
        if not candidates:
            continue
        _,height,_,_ = max(candidates,key=lambda r:r[0])
        fine = _grid(rgb,mask,M,a,height*np.linspace(.972,1.028,23),atlas)
        if fine:
            quality,height,_,ev = max(fine,key=lambda r:r[0])
            if ev["weakest_score"]>=.69 and quality>=.86:
                options.append((quality,height,a,ev))
    pairs = []
    for n,(qa,ha,a,ea) in enumerate(options):
        for qb,hb,b,eb in options[n+1:]:
            delta = (a[3]-b[3])[[0,2]]*(ha+hb)/2
            if np.linalg.norm(delta)<1.8 or abs(ha/hb-1)>.009 or max(abs(delta-np.rint(delta)))>.07:
                continue
            first = (qa,ha,a,ea) if qa>qb else (qb,hb,b,eb)
            second = (qb,hb,b,eb) if qa>qb else (qa,ha,a,ea)
            pairs.append((min(qa,qb),first,second))
    if not pairs:
        return _abstain("independent_ground_contacts_disagree", eligible_contacts=len(options))
    _,(_,height,anchor,ev),(_,hb,b,other) = max(pairs,key=lambda row:row[0])
    pred = _result(M,anchor,height,dict(height=height,anchor=anchor[2].tolist(),axis=anchor[1],**ev,
                   confirming_contact=dict(height=hb,anchor=b[2].tolist(),axis=b[1],**other)))
    pred["reason"] = "two_registered_wall_contacts"
    return pred

