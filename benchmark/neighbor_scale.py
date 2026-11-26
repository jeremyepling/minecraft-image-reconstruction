"""Transfer metric depth across partially shared, image-supported corners."""
from collections import Counter

import numpy as np
from PIL import Image, ImageFilter

from benchmark.mask_blocks import _carrier
from benchmark.surface_metrics import DOWNSAMPLE
from engine.surface_model import intrinsic, lift


SETTINGS = dict(contact_radius_samples=2, minimum_contact_samples=4,
                minimum_edge_length_blocks=.8, grid_residual_blocks=.3,
                maximum_edge_distance_px=6., minimum_normal_alignment=.86)


def _contact_plane(mask, other, donor, axis, P, center):
    donor_axis, level, _ = donor
    if donor_axis == axis:
        return None
    # Only use recipient pixels just outside the donor along a shared outline.
    near = np.asarray(Image.fromarray(other).convert('L').filter(
        ImageFilter.MaxFilter(1 + 2*SETTINGS['contact_radius_samples']))) > 0
    y, x = np.where(mask & near & ~other)
    if len(x) < SETTINGS['minimum_contact_samples']:
        return None
    pixels = np.column_stack(((x+.5)*DOWNSAMPLE, (y+.5)*DOWNSAMPLE, np.ones(len(x))))
    free = [j for j in range(3) if j != donor_axis]
    chart = np.column_stack((P[:,free], P[:,3] + level*P[:,donor_axis]))
    q = np.linalg.solve(chart, pixels.T)
    if np.min(q[2]) <= 0:
        return None
    q = (q[:2] / q[2]).T
    column = free.index(axis)
    inferred = round(float(np.median(q[:,column])))
    if np.quantile(abs(q[:,column]-inferred), .8) > SETTINGS['grid_residual_blocks']:
        return None
    low,high = np.quantile(q[:,1-column], [.05,.95])
    if high-low < SETTINGS['minimum_edge_length_blocks']:
        return None
    # The projected integer intersection must track the observed edge in the image.
    eq = np.zeros(3)
    eq[column],eq[2] = 1.,-inferred
    image_line = np.linalg.solve(chart.T,eq)
    norm = np.linalg.norm(image_line[:2])
    if norm < 1e-9 or np.quantile(abs(pixels@image_line)/norm, .8) > SETTINGS['maximum_edge_distance_px']:
        return None
    return (axis,inferred,1 if center[axis] > inferred else -1)


def transfer(im, inputs, masks, carriers, P):
    if P is None:
        return carriers, []
    scale = np.linalg.norm(P[2,:3])
    focal = np.linalg.norm(P[0,:3]-im.width/2*P[2,:3])/scale
    K = intrinsic(im.size,focal)
    rotation = np.linalg.solve(K,P[:,:3]/scale)
    center = np.linalg.solve(P[:,:3],-P[:,3])
    seeds = [(_carrier(mask,carriers),mask) for mask in masks]
    additions, log = {}, []
    for index,(part,mask) in enumerate(zip(inputs,masks,strict=True)):
        (_,status),_ = seeds[index]
        if status != 'unresolved_scale' or part.get('quad') is None:
            continue
        geom = lift(part['quad'],K,{})
        if geom is None:
            continue
        normal = rotation.T@geom[1]
        axis = int(np.argmax(abs(normal)))
        if abs(normal[axis])/np.linalg.norm(normal) < SETTINGS['minimum_normal_alignment']:
            continue
        votes = Counter()
        for (plane,_),donor_mask in seeds:
            if plane is None:
                continue
            vote = _contact_plane(mask,donor_mask,plane,axis,P,center)
            if vote is not None:
                votes[vote] += 1
        if len(votes) == 1:
            plane = next(iter(votes))
            additions[id(mask)] = plane
            log.append(dict(index=index,plane=plane,donors=votes[plane]))
    return [(m,additions.get(id(m),p),'transferred' if id(m) in additions else status)
            for m,p,status in carriers], log
