"""Reference-only opportunities from native visible corners. Does not supply decisions."""
from itertools import product
import numpy as np

from benchmark.camera_metrics import truth_camera
from benchmark.camera_silhouette_ruler import runs


def measured(traces, faces):
    vertices={tuple(x):uv for f in faces for x,uv,v in zip(f['corners_world'],f['corners_px'],f['corner_visible']) if v}
    world=np.array(list(vertices))
    uv=np.array(list(vertices.values()))

    rows=[]
    for t in traces:
        xy=np.array(t['xy'])
        options=[np.flatnonzero(np.sum((uv-p)**2,axis=1)<=9.) for p in xy]
        if not all(len(o) for o in options):
            rows.append('unmatched')
            continue
        options=[sorted(cs,key=lambda k:np.linalg.norm(uv[k]-p))[:5] for p,cs in zip(xy,options)]
        result='mixed-depth-or-axis'
        for ids in product(*options):
            delta=np.diff(world[list(ids)],axis=0)
            if np.any(np.count_nonzero(delta,axis=1)!=1):
                continue
            axes=abs(delta).argmax(1)
            if not (axes[0]==axes[2] and axes[1]==axes[3] and axes[0]!=axes[1] and 1 in axes):
                continue
            heights=abs(delta[axes==1,1])
            result=str(heights[0]) if heights[0]==heights[1] else 'unequal'
            break
        rows.append(result)
    return rows


def diagnose(polys, source_runs, pose, faces):
    oracle=runs(polys,truth_camera(pose)[:,:3])
    return dict(source=measured(source_runs,faces),pose_orientation=measured(oracle,faces),
                pose_proposals=oracle)

