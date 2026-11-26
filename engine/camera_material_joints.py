"""Experimental ground camera from material changes and neighboring atlas cells."""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter

from engine import camera_seam
from engine.camera_grid import plane_coordinates
from engine.geometry import UNIT, project
from engine.surface_model import _material_assets


def joint_map(image, radius, adaptive):
    smooth = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)),dtype=np.float32)
    dy, dx = np.gradient(smooth, axis=(0,1))
    mag = np.sqrt(np.sum(dx*dx + dy*dy,axis=-1))
    # Isoluminant changes between materials can align across several cells.
    # The existing seam fitter expects a dark stripe on a lighter field.
    cutoff = np.percentile(mag[int(.75*image.height):],60) if adaptive else 0
    response = 185 - 135*np.clip((mag-cutoff)*radius/(5 if adaptive else 10),0,1)
    rgb = np.array(image)
    lo = int(.66*image.height)
    rgb[lo:] = response[lo:, :, None]
    return Image.fromarray(rgb)


@lru_cache(maxsize=1)
def matcher(version):
    return _material_assets(version,'texture-guided-repeat','shape',None)[1]


def confirm(image,P,version):
    rgb = np.asarray(image)
    w,h=image.size
    x,y = np.meshgrid(np.linspace(.1*w,.9*w,8),np.linspace(.77*h,.94*h,4))
    centers = np.floor(plane_coordinates(P,np.stack((x.ravel(),y.ravel()),1),1,0)).astype(int)
    cells = np.unique(np.concatenate((centers,centers+[1,0],centers+[0,1])),axis=0)
    art = matcher(version)
    if art is None:
        return []
    matches = {}
    for a,b in cells:
        corners = UNIT + [a,b]
        quad = project(P, np.insert(corners,1,0,axis=1))
        if min(quad[:,1]) < .68*h or max(quad[:,1])>=h or min(quad[:,0])<0 or max(quad[:,0])>=w:
            continue
        m = art.rank(rgb,quad)
        if m['scores'][0] >= .55 and m['score_margin'] >= .04:
            matches[(a,b)] = m['texture']
    return [dict(first=[int(a),int(b)],second=[int(v) for v in n],a=art1,b=matches[n]) for (a,b),art1 in matches.items()
            for n in ((a+1,b),(a,b+1)) if n in matches and matches[n]!=art1]


def predict(image,version='26.3'):
    image = image.convert('RGB')
    for radius,adaptive in ((3,False),(3,True),(5,True)):
        result = camera_seam.predict(joint_map(image,radius,adaptive),version)
        if result['P'] is None:
            continue
        edges = confirm(image,np.array(result['P']),version)
        if len(edges)>=2:
            result['evidence'].update(confirmed_edges=edges,radius=radius,adaptive=adaptive)
            result['reason']='Ground material joints and adjacent atlas cells agree'
            return result
    return dict(P=None,scale_status='uncertain',reason='Insufficient verified ground joints',evidence={})
