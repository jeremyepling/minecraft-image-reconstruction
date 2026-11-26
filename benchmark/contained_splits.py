"""Use a supported two-model outline to divide one existing surface."""
import numpy as np
from PIL import Image, ImageFilter

from benchmark.mask_metrics import footprint
from engine import plane_consensus as consensus
from engine import single_wall_gate as gate
from engine.camera_lines import _runs


SETTINGS = dict(min_side_samples=gate.SETTINGS['min_side_area'],
                max_spill_fraction=.05, max_spill_radius_cells=1, min_samples=gate.SETTINGS['min_sample_pairs'],
                min_normal_angle_deg=consensus.SETTINGS['step_angle_deg'],
                min_depth_step=gate.SETTINGS['min_step'], min_agreement=gate.SETTINGS['min_agreement'],
                near_distance=gate.SETTINGS['near_distance'], far_distance=gate.SETTINGS['far_distance'],
                selection='One confirmed seam per starting parent. Prefer more sampled crossings, then larger intersection, then input order.')


def containment(region, masks, union, event):
    area = int(region.sum())
    hits = [int(np.count_nonzero(region & m)) for m in masks]
    if not any(hits):
        event['reason'] = 'no-parent'
        return None
    index = int(np.argmax(hits))
    parent = masks[index]
    event.update(proposal_samples=area, inside_samples=hits[index], unowned_samples=int((region & ~union).sum()),
                 other_parent_samples=max(hits[:index]+hits[index+1:], default=0))
    # A one-cell registration discrepancy can touch another surface's rim. Reject a
    # significant intersection in either direction, including an engulfed small surface.
    for i, other in enumerate(masks):
        if i != index and hits[i] > SETTINGS['max_spill_fraction'] * min(area, int(other.sum())):
            event['reason'] = 'crosses-second-parent'
            return index
    spill = region & ~parent
    if spill.any():
        width = 1 + 2 * SETTINGS['max_spill_radius_cells']
        rim = np.asarray(Image.fromarray(parent.astype(np.uint8)).filter(ImageFilter.MaxFilter(width)), bool)
        if spill.sum() > SETTINGS['max_spill_fraction'] * area or np.any(spill & ~rim):
            event['reason'] = 'outside-parent'
            return index
    if min(hits[index], int(parent.sum())-hits[index]) < SETTINGS['min_side_samples']:
        event['reason'] = 'small-side'
    return index


def seam(region, parent, labels, normals, depth, event):
    pairs = gate.sample_pairs(region, parent, SETTINGS['near_distance'], SETTINGS['far_distance'])
    event['seam_samples'] = len(pairs)
    if len(pairs) < SETTINGS['min_samples']:
        return False
    vote, = gate.votes([dict(samples=pairs, parent=event['parent'], proposal=event['proposal'])], depth)
    pts = np.asarray(pairs)
    # Compare the plane directions just inside each side of the proposed seam.
    # The zero label denotes missing geometry and cannot confirm a cut.
    left, right = labels[pts[:,0],pts[:,1]], labels[pts[:,6],pts[:,7]]
    a, b = normals[left], normals[right]
    lengths = np.linalg.norm(a,axis=1) * np.linalg.norm(b,axis=1)
    valid = (left!=0) & (right!=0) & (lengths>.8)
    angle_votes = np.einsum('ij,ij->i',a,b) < lengths*np.cos(np.radians(SETTINGS['min_normal_angle_deg']))
    normal_support = int(np.count_nonzero(valid & angle_votes))
    normal_ok = normal_support>=SETTINGS['min_samples'] and normal_support>=len(pairs)*SETTINGS['min_agreement']
    depth_ok = abs(vote['median_step'])>=SETTINGS['min_depth_step'] and vote['sign_agreement']>=SETTINGS['min_agreement']
    event.update(normal_valid_samples=int(valid.sum()), normal_support_samples=normal_support, normal_supported=normal_ok,
                 depth_step=vote['median_step'],depth_agreement=vote['sign_agreement'], depth_supported=depth_ok)
    return normal_ok or depth_ok


def partition(image, parent, proposal):
    size = tuple(parent['support']['size'])
    original = consensus.run_mask(parent['support'])
    region = consensus.resize(consensus.run_mask(proposal['support']),size)
    if not np.any(original & region) or not np.any(original & ~region):
        return []
    gray = np.asarray(image.convert('L').resize(size,Image.Resampling.BOX).filter(
        ImageFilter.GaussianBlur(.9)),dtype=np.float32)[...,None]
    scale = np.asarray(image.size) / size
    children = []
    for side, mask, normal in [('inside',original & region,proposal['normal_camera']),
                               ('remainder',original & ~region,parent['normal_camera'])]:
        for runs in _runs(mask):
            piece = np.zeros_like(original)
            for y,start,stop in runs:
                piece[y,start:stop] = True
            child = dict(id=f"{parent['id']}-contained-{len(children)}", parent_id=parent['id'],
                         contained_proposal=proposal['id'], contained_side=side, normal_camera=normal,
                         relative_inverse_depth=parent.get('relative_inverse_depth'))
            children.append(consensus.with_mask(child,piece,size,gray,scale))
    return children


def predict(image, parents, guard_parts, depth, plane_fields):
    """Predict using the saved protected-edge proposals, aligned depth and plane directions."""
    shape = (image.height//gate.STRIDE,image.width//gate.STRIDE)
    labels, normals = plane_fields['relaxed'], plane_fields['relaxed_normals']
    if (depth.shape!=shape or labels.shape!=shape or normals.ndim!=2 or normals.shape[1]!=3 or
            labels.min()<0 or labels.max()>=len(normals) or not np.isfinite(normals).all() or not np.isfinite(depth).all()):
        raise ValueError('Expected aligned, finite model fields')
    masks = [footprint(p,shape,image.size,True) for p in parents]
    union = np.logical_or.reduce(masks) if masks else np.zeros(shape,bool)
    choices, decisions = {}, []
    for proposal in guard_parts:
        if not proposal['id'].startswith('plane-consensus-'):
            continue
        region = footprint(proposal,shape,image.size,True)
        event = dict(proposal=proposal['id'],parent=None,reason=None)
        decisions.append(event)
        index = containment(region,masks,union,event)
        if index is not None:
            event['parent'] = parents[index]['id']
        if event['reason']:
            continue
        if not seam(region,masks[index],labels,normals,depth,event):
            event['reason'] = 'unsupported-seam'
            continue
        event['reason'] = 'competing-split'
        rank = event['seam_samples'],event['inside_samples']
        if index not in choices or rank>choices[index][0]:
            choices[index] = rank,proposal,event
    result = []
    for i,parent in enumerate(parents):
        if i in choices:
            _,proposal,event = choices[i]
            children = partition(image,parent,proposal)
            if children:
                event.update(reason='accepted',action='split',children=[c['id'] for c in children])
                result.extend(children)
                continue
            event['reason'] = 'empty-native-side'
        result.append(parent)
    return result,decisions
