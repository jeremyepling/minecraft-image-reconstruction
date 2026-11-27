"""Propose a roof from two contiguous perpendicular photographed rims at one height."""
from collections import defaultdict

from engine import floating_grid_policy as floating
import numpy as np

from engine.atlas_phase import ATLAS_SHA256
from engine.camera_atlas_floor import _atlas
from engine.roof_span_texture import measure
from engine.roof_height_atlas import horizontal_pixels
from engine.surface_model import intrinsic
from engine.thin_atlas_faces import StripEvidence
from engine.visible_horizontal_faces import _keep_prior_visible
from engine.visible_placement import _sample_face


def corners(edges, eye):
    """Both legs start at the camera-facing corner and extend to their last measured cell."""
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
        yield x,y,z,dx,dz,width,depth


def rects(edges, eye, prior):
    result = set()
    rectangles = 0
    for x,y,z,dx,dz,width,depth in corners(edges,eye):
        rectangles += 1
        result.update((x + i*dx, y, z + j*dz)
                      for i in range(1, width) for j in range(1, depth))
    return result - prior, rectangles


def shared_top(ctx,parts,edges,prior):
    """Associate both measured rims with one horizontal surface from the photo."""
    focal=np.linalg.norm(ctx.P[0,:3]-.5*ctx.size[0]*ctx.P[2,:3])
    rotation=np.linalg.solve(intrinsic(ctx.size,focal),ctx.P[:,:3])
    labels=np.zeros(ctx.shape,np.int32)
    for i,part in enumerate(parts,1):
        pixels=horizontal_pixels(part,ctx.size,rotation)
        if pixels is not None:
            x,y=pixels
            labels[y,x]=np.where(labels[y,x]==0,i,-1)
    cache={}
    def label(cell):
        if cell not in cache:
            cache[cell]=0
            hit=_sample_face(ctx.native,cell,1,1,ctx.shape)
            if hit:
                box,face,depth,*_=hit
                seen=face & (ctx.axes[box]==1) & np.isclose(ctx.depth[box],depth,atol=1e-8,rtol=0)
                votes=labels[box][seen]
                positive=votes[votes>0]
                if len(positive)>=4:
                    counts=np.bincount(positive)
                    j=int(counts.argmax())
                    if counts[j]>=4 and counts[j]>=.80*len(votes):
                        cache[cell]=j
        return cache[cell]
    allowed=defaultdict(set)
    for x,y,z,dx,dz,width,depth in corners(edges,ctx.eye):
        left=[label((x+i*dx,y,z)) for i in range(width)]
        right=[label((x,y,z+i*dz)) for i in range(depth)]
        surfaces={s for s in set(left)&set(right) if s>0 and left.count(s)>=2 and right.count(s)>=2}
        if surfaces:
            for i in range(1,width):
                for j in range(1,depth):
                    cell=(x+i*dx,y,z+j*dz)
                    if cell not in prior:
                        allowed[cell].update(surfaces)
    return allowed,labels


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


def top_witness(ctx, cell, joint, unclaimed, shared=None):
    owner, axes, ids = joint
    hit = _sample_face(ctx.native, cell, 1, 1, ctx.shape)
    if hit is None:
        return None
    box, face, depth, *_ = hit
    front = face & (owner[box] == ids[cell]) & (axes[box] == 1) & (depth > ctx.depth[box]+1e-8)
    support = front & ctx.masks[1][box]
    if shared is not None:
        allowed,labels=shared
        support &= np.isin(labels[box],list(allowed[cell]))
    if unclaimed:
        # A rim can bound several neighboring planes. Admit only exposed photo
        # area that the original build did not already explain with another cube.
        fresh = support & (ctx.depth[box] == 0)
        if fresh.sum() < 4 or fresh.sum() < .85*support.sum():
            return None
    y, x = np.nonzero(support)
    if len(x) < 4 or len(x) < .88*front.sum():
        return None
    # Across-perimeter apparent thickness, in native pixels, for the evidence audit.
    return float(len(x)/max(int(np.ptp(x))+1,int(np.ptp(y))+1))


def recover(image, parts, counts, camera, prior, *, texture=True, unclaimed=False, shared_patch=False, atlas_path=None):
    stats = dict(edge_rectangles=0, proposed=0, top_witnesses=0, accepted=0, thin_accepted=0)
    if camera.get('P') is None or not prior or not parts:
        return prior, stats
    P = np.asarray(camera['P'],float)
    eye = np.linalg.solve(P[:,:3],-P[:,3])
    upper = {c for c in prior if (floating.active() or c[1] > 0) and c[1] < eye[1]-1 and (c[0],c[1]+1,c[2]) not in prior}
    if not rects({0:upper,2:upper},eye,prior)[0]:
        return prior, stats
    atlas = _atlas('26.3',atlas_path)
    if atlas is None or atlas.provenance.get('atlas_sha256') != ATLAS_SHA256:
        return prior, stats
    ctx = StripEvidence(image,parts,camera,prior,atlas)
    edges = observed_edges(ctx,upper)
    proposals, count = rects(edges,eye,prior)
    stats.update(edge_rectangles=count, proposed=len(proposals))
    if not proposals:
        return prior, stats
    shared = None
    if shared_patch:
        shared = shared_top(ctx,parts,edges,prior)
        proposals &= shared[0].keys()
        stats['two_rims_same_surface'] = len(proposals)
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
            thickness = top_witness(ctx,cell,joint,unclaimed,shared)
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


def uncovered(image,parts,counts,camera,prior):
    return recover(image,parts,counts,camera,prior,texture=False,unclaimed=True)


def contiguous(image,parts,counts,camera,prior,*,atlas_path=None):
    return recover(image,parts,counts,camera,prior,texture=False,unclaimed=True,shared_patch=True,atlas_path=atlas_path)
