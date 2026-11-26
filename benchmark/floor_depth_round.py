"""Offline adapter for calibrated relative-depth placement. Only RGB indexes the model cache."""
import numpy as np

from engine import ground_depth, visible_placement
from engine.visible_raster import observed_support, SAMPLE_STRIDE
from benchmark.floor_depth_fields import field


def measurements(image,parts,camera,kind,anchored=False):
    if camera['P'] is None or not parts:
        return {}
    P=np.asarray(camera['P'])
    P=P/np.linalg.norm(P[2,:3])
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    shape=tuple((n+SAMPLE_STRIDE-1)//SAMPLE_STRIDE for n in image.size[::-1])
    masks=[observed_support(p['support'],shape,image.size) if p.get('support') else np.zeros(shape,bool) for p in parts]
    return ground_depth.plane_measurements(image,parts,masks,P,eye,field(image,kind),inverse=kind=='inverse',anchored=anchored)


def run(image,version,parts,counts,camera,kind,anchored=False):
    levels=measurements(image,parts,camera,kind,anchored)
    def select(part,mask,P,eye,rays,axis,original,scene_distance):
        return round(levels[part['id']][0]) if part['id'] in levels else original
    return visible_placement.place_with_camera(image,version,parts,counts,camera,select)


def predict(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'inverse')


def points(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'points')


def anchored(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'inverse',True)
