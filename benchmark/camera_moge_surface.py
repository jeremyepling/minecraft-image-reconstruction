"""Test a block-unit camera from source-visible surfaces, MoGe geometry and verified sprites."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from benchmark.camera_moge_fields import BASE, RUN, WORKTREES, failures, read, save
sys.path.append(str(WORKTREES/'mask-scalelsd/runs/env/Lib/site-packages'))
import cv2

from benchmark.camera_structure_scaffold import count_scale
from benchmark.full_block_lattice import consensus, mask_parts, sprite_candidates
from engine.surface_model import intrinsic

cv2.setNumThreads(1)


def unit(vec):
    return vec/max(1e-12,np.linalg.norm(vec))


def mask_at(support, shape):
    w,h = support['size']
    raw = np.zeros((h,w),dtype='B')
    for y,start,end in support['runs']:
        raw[y,start:end] = 1
    return cv2.resize(raw,shape[::-1],interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)


def plane(points):
    center = np.median(points,axis=0)
    keep = points
    for _ in range(2):
        _,_,v = np.linalg.svd(keep-center,full_matrices=False)
        n = v[-1]
        residual = abs((points-center)@n)
        keep = points[residual<=np.quantile(residual,.7)]
        center = np.mean(keep,axis=0)
    return n


def geometry(field,parts):
    valid = field['mask']
    normal = field['normal'].astype(float)
    xyz = field['points'].astype(float)
    xyz[:,:,2] += field['z_shift']
    valid = valid & (xyz[:,:,2]>.01)
    h,w = valid.shape
    masks = [(p,mask_at(p['support'],valid.shape)&valid) for p in parts if p.get('support')]
    building = np.logical_or.reduce([m for p,m in masks]) if masks else np.zeros_like(valid)
    floor = valid & ~building & (normal[:,:,1]<-.60)
    floor[:int(.74*h)] = False
    normal_votes = []
    floor_up = None
    if floor.sum()>=60:
        pts = xyz[floor][::max(1,int(floor.sum())//1500)]
        n = plane(pts)
        if n[1]>0: n=-n
        floor_up = unit(n+unit(np.median(normal[floor],axis=0)))
    for p,mask in masks:
        if mask.sum()<20:
            continue
        n = unit(np.median(normal[mask],axis=0))
        pts = xyz[mask][::max(1,int(mask.sum())//500)]
        n2 = plane(pts)
        if abs(n@n2)>.9:
            n = unit(n+np.sign(n@n2)*n2)
        normal_votes.append((n,min(15.,np.sqrt(mask.sum())/4)))
    vertical = [(n*np.sign(-n[1]),wgt) for n,wgt in normal_votes if abs(n[1])>.78]
    if floor_up is not None: vertical.append((floor_up,40.))
    if not vertical: return None
    up = unit(sum(n*wt for n,wt in vertical))
    right = unit(np.array([1.,0.,0.])-up*up[0])
    forward = np.cross(up,right)
    az = [(np.arctan2(n@forward,n@right),wgt) for n,wgt in normal_votes if abs(n@up)<.35]
    if not az: return None
    circular = sum(wt*np.exp(4j*theta) for theta,wt in az)
    strength = abs(circular)/sum(wt for theta,wt in az)
    angle = np.angle(circular)/4
    xaxis = right*np.cos(angle)+forward*np.sin(angle)
    R = np.column_stack((xaxis,up,np.cross(up,xaxis)))
    K = intrinsic(field['native_size'],float(field['focal_px']))
    height = float(-np.median(xyz[floor]@up)) if floor.sum()>=60 else None
    return K@R, xyz@R, dict(floor_model_height=height, wall_axis_agreement=float(strength),
                              surface_votes=len(normal_votes),horizontal_votes=len(az),floor_samples=int(floor.sum()))


def propose(image,field,source):
    g = geometry(field,source['parts'])
    if g is None:
        return dict(camera=None,linear=None,reason='insufficient observed planes')
    linear,xyz,details = g
    out = dict(camera=None,linear=linear.tolist(),**details)
    if details['wall_axis_agreement']<.55 or not (.3*image.height<float(field['focal_px'])<3*image.height):
        return dict(out,reason='inconsistent orthogonal planes or focal')
    parts = mask_parts(source['parts'],image.size,linear)
    by_id = {p['id']:p for p in source['parts']}
    counts = {c['id']:c for c in source['counts'].get('surfaces',[])}
    rgb = np.asarray(image)
    anchors, scales = [],[]
    for p in parts[:14]:
        axis = p['axis']
        valid = mask_at(by_id[p['id']]['support'],field['mask'].shape)&field['mask']
        if valid.sum()<25: continue
        plane_dist = float(np.median(xyz[:,:,axis][valid]))
        if abs(plane_dist)<.015: continue
        cs = count_scale(counts[p['id']],linear,axis) if p['id'] in counts else None
        sprites = sprite_candidates(p,rgb,linear)
        anchors.extend(sprites)
        for kind,dist in [('repeated-wall',cs),('whole-sprite',np.median([a['xyz'][0][axis] for a in sprites]) if sprites else None)]:
            if dist is not None and dist/plane_dist>0:
                scales.append(dict(surface=p['id'],axis=axis,kind=kind,factor=float(dist/plane_dist)))
    out.update(scales=scales,anchors=anchors)
    # Separate one-block matches set phase. The independently observed floor plane sets block elevation.
    chosen, eye, _ = consensus(anchors)
    if len(chosen)<2 or not details['floor_model_height']:
        return dict(out,reason='missing independent sprite phase or observed ground level')
    if np.ptp(np.array([a['xyz'][0] for a in chosen]),axis=0).max()<1.7:
        return dict(out,reason='insufficient anchor separation')
    matched = {a['surface'] for a in chosen}
    factors = [s['factor'] for s in scales if s['surface'] in matched and s['kind']=='whole-sprite']
    if not factors: return dict(out,reason='no positive source-to-model scale')
    factor = float(np.median(factors))
    verify = [s for s in scales if abs(s['factor']/factor-1)<.12]
    if len({s['surface'] for s in verify})<2:
        return dict(out,reason='independent visible surface scale disagrees')
    height = details['floor_model_height']*factor
    if not .15<height<120 or abs((height-eye[1]+.5)%1-.5)>.22:
        return dict(out,reason='model ground and texture phase disagree')
    eye[1] += round(height-eye[1])
    P = np.column_stack((linear,-linear@eye))
    camera = dict(P=P.tolist(),scale_status='supported',camera_source='moge-surface-first',
                  reason='independent visible surface scales plus ground and separated full-sprite phase agree')
    return dict(out,camera=camera,reason=camera['reason'],verified_scales=verify)


def predict(e):
    base,out = BASE/e['dataset']/e['id'], RUN/e['dataset']/e['id']
    with Image.open(Path(e['frame'])/'rgb.png') as im, np.load(out/'field.npz') as field:
        result=propose(im.convert('RGB'),dict(field),read(base/'handoffs/placement-input.json.gz'))
    save(out/'proposal.json',dict(result,source_sha256=e['rgb_sha256']))
    return result['reason']


if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['train','held'],required=True)
    args=parser.parse_args()
    entries=[e for e in failures() if (e['split']=='train')==(args.split=='train')]
    with ProcessPoolExecutor(max_workers=4) as pool:
        print(Counter(pool.map(predict,entries)),flush=True)
