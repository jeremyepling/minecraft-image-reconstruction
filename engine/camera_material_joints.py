"""Experimental ground camera from material changes and neighboring atlas cells."""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter

from engine import camera_seam
from engine.camera_grid import plane_coordinates
from engine.geometry import UNIT, project
from engine.surface_model import _material_assets


def joint_map(image):
    smooth = np.asarray(image.filter(ImageFilter.GaussianBlur(2.)),dtype=np.float32)
    dy, dx = np.gradient(smooth, axis=(0,1))
    mag = np.sqrt(np.sum(dx*dx + dy*dy,axis=-1))
    # Isoluminant changes between materials can align across several cells.
    # The existing seam fitter expects a dark stripe on a lighter field.
    response = 178 - 110 * np.clip((mag-1.4)/8,0,1)
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
    cells = np.unique(np.floor(plane_coordinates(P,np.stack((x.ravel(),y.ravel()),1),1,0)).astype(int),axis=0)
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
    return [dict(a=art1,b=matches[n]) for (a,b),art1 in matches.items()
            for n in ((a+1,b),(a,b+1)) if n in matches and matches[n]!=art1]


def predict(image,version='26.3'):
    result = camera_seam.predict(joint_map(image.convert('RGB')),version)
    if result['P'] is None:
        return result
    edges = confirm(image,np.array(result['P']),version)
    if len(edges)<2:
        return dict(P=None,scale_status='uncertain',reason='Insufficient distinct neighboring atlas matches',
                    evidence={'confirmed_edges':edges})
    result['evidence']['confirmed_edges']=edges
    result['reason']='Ground material joints and adjacent atlas cells agree'
    return result
