"""Image-only fallback for the frozen placement replay. All geometry is view-supported."""
import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

from benchmark.camera_labels import rasterize
from benchmark.camera_metrics import registration
from benchmark.camera_round import read, digest
from benchmark.mask_blocks import _footprints, _write_read, decoded_pack, registered
from engine.camera_grid import predict
from engine.surface_model import lens
from engine.masked_count import count
from engine.surface_sample import sample_surface

STRIDE = 4
MIN_SUPPORT = 4
MIN_COVERAGE = .5
UNIT = np.array([[0,0],[1,0],[1,1],[0,1]])


def unit(a):
    return a/max(np.linalg.norm(a),1e-10)


def approximate_camera(im, parts, masks):
    K, _ = lens([p['quad'] for p in parts if p.get('quad')], im.size)
    order = sorted(range(len(parts)),key=lambda i:-masks[i].sum())
    normals = [unit(np.asarray(parts[i]['normal_camera'])) for i in order if parts[i].get('normal_camera')]
    front = next((n for n in normals if abs(n[1])<.65),np.array([0.,0.,-1.]))
    up = next((n*np.sign(-n[1]) for n in normals if abs(n[1])>.78),np.array([0.,-1.,0.]))
    up = unit(up-front*(up@front))
    right = unit(np.cross(up,front))
    rotation = np.column_stack([right,up,front])
    pitches = []
    for index in order[:5]:
        sample = sample_surface(im,parts[index])
        if sample is None:
            continue
        width,height = count(sample)['counts']
        quad = np.array(sample['quad'])
        for idx,axis_count in enumerate((width,height)):
            if axis_count and axis_count>=2:
                a,b = (0,1) if idx==0 else (0,3)
                pitches.append(np.linalg.norm(quad[a]-quad[b])/axis_count)
    # Weak block-size prior only when the image counter finds no repeat.
    native_pitch = float(np.clip(np.median(pitches) if pitches else im.height/24,im.height/90,im.height/5))
    distance = K[0,0]/native_pitch
    mask = np.logical_or.reduce(masks)
    yy,xx = np.where(mask)
    if not len(xx):
        return None,None
    bottom = yy>=np.quantile(yy,.98)
    anchor = np.array([np.median(xx[bottom]+.5)*STRIDE,np.median(yy[bottom]+.5)*STRIDE,1])
    ground_point = np.linalg.solve(K,anchor)*distance
    eye_height = max(.5,float(-ground_point@up))
    translation = -rotation@np.array([0.,eye_height,0.])
    return K@np.column_stack([rotation,translation]),distance


def face_raster(P, coord, axis, sign, shape):
    face = np.tile(coord,(4,1))
    face[:,axis] += int(sign>0)
    free = [a for a in range(3) if a!=axis]
    face[:,free] += UNIT
    verts = face@P[:,:3].T+P[:,3]
    if verts[:,2].min()<=1e-5:
        return None
    verts[:,:2] /= STRIDE
    try:
        return rasterize(verts,shape[1],shape[0])
    except np.linalg.LinAlgError:
        return None


def render(cells,P,shape):
    eye = np.linalg.solve(P[:,:3],-P[:,3])
    depth = np.zeros(shape)
    owner = np.full(shape,-1,int)
    for index,xyz in enumerate(cells):
        for axis in range(3):
            if xyz[axis]<=eye[axis]<=xyz[axis]+1:
                continue
            sign = 1 if eye[axis]>xyz[axis] else -1
            adj = list(xyz); adj[axis]+=sign
            if tuple(adj) in cells:
                continue
            hit = face_raster(P,xyz,axis,sign,shape)
            if hit:
                box,mask,z,_,_ = hit
                win = mask & (z>depth[box])
                depth[box][win] = z[win]
                owner[box][win] = index
    return owner


def rough_planes(im, parts, masks, P, fallback_distance):
    scale = np.linalg.norm(P[2,:3])
    P = P/scale
    eye = np.linalg.solve(P[:,:3],-P[:,3])
    focal = np.linalg.norm(P[0,:3] - im.width/2*P[2,:3])
    K = np.array([[focal,0,im.width/2],[0,focal,im.height/2],[0,0,1.]])
    rotation = np.linalg.solve(K,P[:,:3])
    shape=masks[0].shape
    yy,xx=np.indices(shape)
    px=np.stack([(xx+.5)*STRIDE,(yy+.5)*STRIDE,np.ones(shape)],axis=-1)
    rays=px@np.linalg.inv(P[:,:3]).T
    union=np.logical_or.reduce(masks)
    last=np.where(union,yy,-1).max(axis=0)
    boundary=union & (yy==last) & (yy<shape[0]-1)
    floor_z=np.divide(-eye[1],rays[:,:,1],out=np.full(shape,np.nan),where=abs(rays[:,:,1])>1e-7)
    contacts=boundary & (floor_z>1) & (floor_z<150)
    distance = fallback_distance or (np.median(floor_z[contacts]) if contacts.any() else focal/(im.height/24))
    inv_depths=[p.get('relative_inverse_depth') for p in parts if p.get('relative_inverse_depth',0)>0]
    typical = np.median(inv_depths) if inv_depths else 1
    cells = {}
    for p,mask in zip(parts,masks):
        if not mask.any() or p.get('normal_camera') is None:
            continue
        n=rotation.T@p['normal_camera']
        axis=int(np.argmax(abs(n)))
        touching=mask & contacts
        if axis!=1 and touching.sum()>=4 and touching.sum()>=.12*mask.any(axis=0).sum():
            intercepts = eye[axis]+floor_z[touching]*rays[touching,axis]
            level=int(np.rint(np.median(intercepts)))
        else:
            ratio=typical/max(p.get('relative_inverse_depth') or typical,.001)
            z=distance*np.clip(ratio,.7,1.4)
            level=int(np.rint(eye[axis]+np.median(rays[mask,axis])*z))
        dz=np.divide(level-eye[axis],rays[:,:,axis],out=np.zeros(shape),where=abs(rays[:,:,axis])>1e-7)
        keep=mask & (dz>1.5) & (dz<150)
        if keep.sum()<MIN_SUPPORT:
            continue
        xyz=np.floor(eye+dz[keep,None]*rays[keep]).astype(int)
        sign=1 if eye[axis]>level else -1
        xyz[:,axis]=level-int(sign>0)
        xyz=xyz[xyz[:,1]>=0]
        coords,counts=np.unique(xyz,axis=0,return_counts=True)
        for coord in coords[counts>=MIN_SUPPORT]:
            hit=face_raster(P,coord,axis,sign,shape)
            if hit is None:
                continue
            box,pixels,_,_,_=hit
            samples=int((pixels & mask[box]).sum())
            if samples>=MIN_SUPPORT and samples>=MIN_COVERAGE*pixels.sum():
                cells.setdefault(tuple(coord.tolist()),[]).append(mask)
    return cells


def process(job):
    row,root,baseline,out,expected=job
    key=row['key']
    folder=root/'runs/visible-builds-v1/data/regression/frames'/('legacy-'+key.replace('/','-'))
    source=root/'benchmark/reviews/sky-gap-01'/row['predictions']
    assert digest(folder/'rgb.png')==row['sha256'] and digest(source)==expected
    parts=json.loads(gzip.decompress(source.read_bytes()))['gap']
    with Image.open(folder/'rgb.png') as image:
        im=image.convert('RGB')
    masks=_footprints(parts,im.size,True)
    camera=predict(im,'26.3','seam-fit')
    prior=decoded_pack(baseline/'packs'/(key+'.zip'))
    if camera['P'] is None:
        assert not prior
        P,distance=approximate_camera(im,parts,masks)
    else:
        P=np.array(camera['P'])
        distance=None
    evidence=rough_planes(im,parts,masks,P,distance)
    union=set(prior)|set(evidence)
    # Reject speculative cells with no visible sample from a supporting region
    # in the assembled 3-D view, and keep all originally exported locations.
    owner=render(union,P,masks[0].shape)
    accepted=set(prior)
    for index,coord in enumerate(union):
        if coord in evidence and any(np.count_nonzero((owner==index)&m)>=MIN_SUPPORT for m in evidence[coord]):
            accepted.add(coord)
    archive=out/'packs'/(key+'.zip'); archive.parent.mkdir(parents=True,exist_ok=True)
    cells=_write_read(archive,accepted)
    projected=render(cells,P,masks[0].shape)>=0
    path=out/'masks'/(key+'.png'); path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(projected).save(path)
    # Only the evaluator reads the pose and labels from this point on.
    pose=read(folder/'camera.json')
    target={tuple(b['xyz_world']) for b in read(folder/'labels.json')['blocks']}
    coords=registered(cells,registration(P,pose))
    actual=np.load(folder/'labels.npz')['face_id'][2::4,2::4]>0
    return dict(key=key,cohort=row['cohort'],status=camera['scale_status'],coords=sorted(coords),cells=len(cells),
                correct=len(coords&target),extra=len(coords-target),missed=len(target-coords),
                intersection=int((actual&projected).sum()),union=int((actual|projected).sum()),
                predicted_pixels=int(projected.sum()),visible_pixels=int(actual.sum()),P=P.tolist(),
                image_sha=digest(folder/'rgb.png'),mask_sha=expected,pack_sha=digest(archive))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','baseline','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    suite=args.root/'benchmark/reviews/sky-gap-01'
    manifest=read(suite/'manifest.json')
    jobs=[(r,args.root,args.baseline,args.out,manifest['artifacts'][r['predictions']]) for r in read(suite/'cases.json')]
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(process,jobs))
    coords={r['key']:r.pop('coords') for r in rows}
    (args.out/'coords.json').write_text(json.dumps(coords))
    totals=Counter()
    for r in rows:
        totals.update({k:r[k] for k in ('cells','correct','extra','missed','intersection','union','predicted_pixels','visible_pixels')})
    report=dict(totals=totals,producing=sum(r['cells']>0 for r in rows),cases=rows,
                suite_sha=digest(suite/'manifest.json'),runner_sha=digest(Path(__file__)))
    (args.out/'results.json').write_text(json.dumps(report,indent=2))
    print(totals)


if __name__=='__main__':
    os.environ['MCIR_IMAGE_ONLY']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    main()
