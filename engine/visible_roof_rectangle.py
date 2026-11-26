"""Propose a roof from two contiguous perpendicular photographed rims at one height."""
from collections import defaultdict

import numpy as np

from engine.atlas_phase import ATLAS_SHA256
from engine.camera_atlas_floor import _atlas
from engine.roof_span_texture import measure
from engine.thin_atlas_faces import StripEvidence
from engine.visible_horizontal_faces import _keep_prior_visible
from engine.visible_placement import _sample_face


def rects(edges, eye, prior):
    """Both legs start at the camera-facing corner and extend to their last measured cell."""
    result = set()
    rectangles = 0
    for x, y, z in edges[0] & edges[2]:
        dx = 1 if eye[0] < x else -1
        dz = 1 if eye[2] < z else -1
        width = depth = 1
        while (x + width*dx, y, z) in edges[2]:
            width += 1
        while (x, y, z + depth*dz) in edges[0]:
            depth += 1
        if min(width, depth) < 2:
            continue
        rectangles += 1
        result.update((x + i*dx, y, z + j*dz)
                      for i in range(1, width) for j in range(1, depth))
    return result - prior, rectangles


def observed_edges(ctx, upper):
    edges = defaultdict(set)
    for cell in upper:
        for axis in (0, 2):
            if cell[axis] <= ctx.eye[axis] <= cell[axis]+1:
                continue
            sign = 1 if ctx.eye[axis] > cell[axis]+1 else -1
            neighbor = list(cell)
            neighbor[axis] += sign
            if tuple(neighbor) in ctx.prior:
                continue
            hit = _sample_face(ctx.native, cell, axis, sign, ctx.shape)
            if hit is None:
                continue
            box, face, depth, *_ = hit
            pixels = face & (ctx.axes[box] == axis) & np.isclose(depth, ctx.depth[box], atol=1e-8, rtol=0)
            observed = np.count_nonzero(pixels & ctx.masks[axis][box])
            if observed >= 4 and observed >= .70*pixels.sum():
                edges[axis].add(cell)
    return edges


def top_witness(ctx, cell, joint):
    owner, axes, ids = joint
    hit = _sample_face(ctx.native, cell, 1, 1, ctx.shape)
    if hit is None:
        return None
    box, face, depth, *_ = hit
    front = face & (owner[box] == ids[cell]) & (axes[box] == 1) & (depth > ctx.depth[box]+1e-8)
    y, x = np.nonzero(front & ctx.masks[1][box])
    if len(x) < 4 or len(x) < .88*front.sum():
        return None
    # Across-perimeter apparent thickness, in native pixels, for the evidence audit.
    return float(len(x)/max(int(np.ptp(x))+1,int(np.ptp(y))+1))


def recover(image, parts, counts, camera, prior, *, texture=True, atlas_path=None):
    stats = dict(edge_rectangles=0, proposed=0, top_witnesses=0, accepted=0, thin_accepted=0)
    if camera.get('P') is None or not prior or not parts:
        return prior, stats
    P = np.asarray(camera['P'],float)
    eye = np.linalg.solve(P[:,:3],-P[:,3])
    upper = {c for c in prior if 0 < c[1] < eye[1]-1 and (c[0],c[1]+1,c[2]) not in prior}
    if not rects({0:upper,2:upper},eye,prior)[0]:
        return prior, stats
    atlas = _atlas('26.3',atlas_path)
    if atlas is None or atlas.provenance.get('atlas_sha256') != ATLAS_SHA256:
        return prior, stats
    ctx = StripEvidence(image,parts,camera,prior,atlas)
    proposals, count = rects(observed_edges(ctx,upper),eye,prior)
    stats.update(edge_rectangles=count, proposed=len(proposals))
    if not proposals:
        return prior, stats
    # Permit one pixel of silhouette mismatch from the sampled surface mask.
    h,w = ctx.shape
    padded = np.pad(ctx.union,1)
    outside = ~np.logical_or.reduce([padded[y:y+h,x:x+w] for y in range(3) for x in range(3)])
    empty_outside = outside & (ctx.depth == 0)
    first = True
    while proposals:
        joint = ctx.joint(proposals)
        owner, axes, ids = joint
        spill = set(np.unique(owner[empty_outside]))
        kept = {}
        for cell in proposals:
            if ids[cell] in spill:
                continue
            thickness = top_witness(ctx,cell,joint)
            if thickness is None:
                continue
            if first:
                stats['top_witnesses'] += 1
            if texture:
                fit = measure(ctx,cell,1,joint)
                if fit is None or fit['split'] < .65 or fit['phase'] < .07 or fit['depth'] < .10:
                    continue
            kept[cell] = thickness
        first = False
        supported = _keep_prior_visible(ctx.size,ctx.P,ctx.eye,prior,set(kept)) - prior if kept else set()
        if supported == proposals:
            stats.update(accepted=len(supported),thin_accepted=sum(kept[c] <= 3 for c in supported))
            return prior | supported, stats
        proposals = supported
    return prior, stats


def layout(image,parts,counts,camera,prior):
    return recover(image,parts,counts,camera,prior,texture=False)
