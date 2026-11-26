"""Protect supported existing seams when one accepted two-model outline spans adjacent parents."""
from itertools import combinations

import numpy as np
from PIL import Image, ImageFilter

from benchmark.mask_metrics import footprint
from engine import plane_consensus as consensus
from engine import single_wall_gate as depth_gate
from engine.camera_lines import _runs


# Only the minimum fraction of an existing parent taken by the new outline is varied.
PRESETS = {'permissive': .05, 'moderate': .15, 'substantial': .30}
SETTINGS = dict(min_samples=depth_gate.SETTINGS['min_sample_pairs'], max_gap_cells=2,
                min_normal_angle=consensus.SETTINGS['step_angle_deg'],
                min_step=depth_gate.SETTINGS['min_step'], min_agreement=depth_gate.SETTINGS['min_agreement'],
                min_component_cells=consensus.SETTINGS['min_component_area'])


def boundary_samples(left, right, candidate):
    """Find local, non-overlapping depth samples in the two parents across the candidate's path."""
    distance = depth_gate.SETTINGS['far_distance']
    samples = []
    # Sample only the crossing nearest each parent boundary along each horizontal/vertical ray.
    for axis in (0, 1):
        a, b, within = [np.moveaxis(m, axis, 1) for m in (left, right, candidate)]
        for first, second in ((a, b), (b, a)):
            for gap in range(SETTINGS['max_gap_cells'] + 1):
                step = gap + 1
                adjacent = first[:, :-step] & second[:, step:] & within[:, :-step] & within[:, step:]
                yy, xx = np.nonzero(adjacent)
                valid = (xx >= distance) & (xx + step + distance < first.shape[1])
                yy, xx = yy[valid], xx[valid]
                if not len(xx):
                    continue
                # Exclude overlaps, other surfaces and holes inside the two short sampling strips.
                valid = np.ones(len(xx), dtype=bool)
                for offset in range(distance + 1):
                    valid &= first[yy, xx-offset] & ~second[yy, xx-offset]
                    valid &= second[yy, xx+step+offset] & ~first[yy, xx+step+offset]
                for offset in range(1, step):
                    valid &= ~first[yy, xx+offset] & ~second[yy, xx+offset] & within[yy, xx+offset]
                yy, xx = yy[valid], xx[valid]
                if not len(xx):
                    continue
                positions = np.stack([xx-distance,xx,xx+step,xx+step+distance],axis=1)
                rows = np.broadcast_to(yy[:,None], positions.shape)
                # Orient each vote from the same parent toward the other.
                if first is b:
                    rows, positions = rows[:,::-1], positions[:,::-1]
                samples.append((rows, positions, step) if axis == 1 else (positions, rows, step))
    return samples


def seam_evidence(left, right, candidate, left_normal, right_normal, normalized_depth):
    samples = boundary_samples(left, right, candidate)
    count = sum(len(yy) for yy, _, _ in samples)
    if count < SETTINGS['min_samples']:
        return None
    normal_support = np.linalg.norm(left_normal) > .99 and np.linalg.norm(right_normal) > .99
    angle = float(np.degrees(np.arccos(np.clip(left_normal @ right_normal, -1, 1)))) if normal_support else 0.
    jumps, residuals = [], []
    for yy, xx, width in samples:
        a, b, c, d = normalized_depth[yy,xx].T
        jump = b-c
        # Remove the extrapolated slopes on either side of the seam. A tilted flat wall
        # has a gradient across its texture even when there is no physical step.
        slope = width/(2*depth_gate.SETTINGS['far_distance']) * ((a-b)+(c-d))
        jumps.append(jump)
        residuals.append(jump-slope)
    residual = np.concatenate(residuals)
    median = float(np.median(residual))
    agreement = float(np.mean((residual*np.sign(median)>0) & (np.concatenate(jumps)*np.sign(median)>0)))
    if angle < SETTINGS['min_normal_angle'] and (
            abs(median)<SETTINGS['min_step'] or agreement<SETTINGS['min_agreement']):
        return None
    return dict(samples=count, angle_deg=angle, depth_step=median, sign_agreement=agreement)


def guard(image, parents, accepted, depth, preset):
    """Constrain a new region only where it crosses two geometry-supported parents."""
    parent_ids = {p['id'] for p in parents}
    proposals = [p for p in accepted if p['id'] not in parent_ids]
    if not proposals:
        return list(accepted), []
    shape = (image.height//4, image.width//4)
    if depth.shape != shape or not np.isfinite(depth).all():
        raise ValueError('Depth must be finite and aligned to the screenshot')
    low,high = np.percentile(depth, (2,98))
    depth = (depth-low)/(high-low) if high>low else np.zeros_like(depth)
    masks = [footprint(p,shape,image.size,True) for p in parents]
    areas = np.array([m.sum() for m in masks])
    normals = np.array([p['normal_camera'] for p in parents], dtype=float)
    lengths = np.linalg.norm(normals,axis=1,keepdims=True)
    normals = np.divide(normals,lengths,out=np.zeros_like(normals),where=lengths>0)
    protection, events = {}, []
    for part in proposals:
        region = footprint(part,shape,image.size,True)
        hits = np.array([np.count_nonzero(m & region) for m in masks])
        inside = np.flatnonzero((hits>=SETTINGS['min_samples']) & (hits>=PRESETS[preset]*areas))
        protected = set()
        for a,b in combinations(inside,2):
            evidence = seam_evidence(masks[a],masks[b],region,normals[a],normals[b],depth)
            if evidence:
                protected.update((a,b))
                events.append(dict(proposal=part['id'], parents=[parents[a]['id'],parents[b]['id']], **evidence))
        if protected:
            protection[part['id']] = sorted(protected)
    if not protection:
        return list(accepted), []
    sizes = [p['support']['size'] for p in [*parents,*proposals]]
    size = tuple(max(s[i] for s in sizes) for i in range(2))
    if any(size[0]%w or size[1]%h for w,h in sizes):
        size = image.size
    raw = [consensus.resize(consensus.run_mask(p['support']), size) for p in parents]
    scale = np.array(image.size) / size
    gray = np.asarray(image.convert('L').resize(size,Image.Resampling.BOX).filter(
        ImageFilter.GaussianBlur(.9)),dtype=np.float32)[...,None]
    claimed = np.zeros(size[::-1],dtype=bool)
    kept = []
    for part in proposals:
        mask = consensus.resize(consensus.run_mask(part['support']),size).copy()
        if part['id'] not in protection:
            claimed |= mask
            kept.append(part)
            continue
        for idx in protection[part['id']]:
            mask &= ~raw[idx]
        # Preserve holes and split disconnected pieces after trimming. A surviving piece
        # that now duplicates an existing parent would just relabel that parent.
        for num,runs in enumerate(_runs(mask)):
            area = sum(b-a for _,a,b in runs)
            if area < SETTINGS['min_component_cells']*mask.size/depth.size:
                continue
            piece = np.zeros_like(mask)
            for y,a,b in runs:
                piece[y,a:b] = True
            if any(np.count_nonzero(piece & old)>=consensus.SETTINGS['duplicate_iou'] *
                   np.count_nonzero(piece | old) for old in raw):
                continue
            kept.append(consensus.with_mask(dict(part,id=f"{part['id']}-guard-{num}"),piece,size,gray,scale))
            claimed |= piece
    output = []
    for part, original in zip(parents,raw):
        rest = original & ~claimed
        if np.array_equal(rest,original):
            output.append(part)
        elif rest.any():
            output.append(consensus.with_mask(part,rest,size,gray,scale))
    return output+kept, events
