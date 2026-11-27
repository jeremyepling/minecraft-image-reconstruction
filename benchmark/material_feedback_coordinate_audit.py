"""Trace native labeled pixels and face corners through recorded and estimated cameras."""
import argparse
from concurrent.futures import ProcessPoolExecutor

from benchmark import material_feedback_geometry as exp, typed_structures as typed
from benchmark.pavement_opening_eval import visible_faces
from benchmark.thin_face_placement_run import ref_inputs
import numpy as np


def assess(W,face_cells,axes,normals,face_ids,corners,corner_pixels):
    eye=-np.linalg.solve(W[:,:3],W[:,3])
    inv=np.linalg.inv(W[:,:3])
    xyh=corners @ W[:,:3].T+W[:,3]
    projected=xyh[:,:2]/xyh[:,2:]
    offsets=np.linalg.norm(projected-corner_pixels,axis=1)
    y,x=np.nonzero(face_ids)
    lookup=face_ids[y,x]
    cells,axis,sign=face_cells[lookup],axes[lookup],normals[lookup]
    rows=np.arange(len(x))
    level=cells[rows,axis]+(sign>0)
    values={}
    for label,sample,offset in [('native_centers',np.ones(len(x),bool),.5),
                                ('coarse_native_centers',(x%4==2)&(y%4==2),.5),
                                ('coarse_placement_centers',(x%4==2)&(y%4==2),0.)]:
        xs,ys=x[sample],y[sample]
        rays=np.column_stack((xs+offset,ys+offset,np.ones(len(xs)))) @ inv.T
        active=np.arange(len(xs)); ax=axis[sample]; dirs=rays[active,ax]
        d=np.divide(level[sample]-eye[ax],dirs,out=np.full(len(xs),-1.),where=abs(dirs)>1.e-10)
        xyz=np.floor(eye+d[:,None]*rays).astype(int)
        xyz[active,ax]=level[sample]-(sign[sample]>0)
        resolved=(d>0)&np.all(xyz==cells[sample],axis=1)
        values[label]=dict(correct=int(resolved.sum()),samples=len(resolved))
    return dict(rays=values,visible_corners=len(offsets),within_half_pixel=int((offsets<=.5).sum()),
                within_one_pixel=int((offsets<=1.).sum()),median_px=float(np.median(offsets)),
                p95_px=float(np.percentile(offsets,95)),max_px=float(offsets.max()))


def check(job):
    split,frame,_,_=job
    key=frame['id']
    labels,ids=visible_faces(split,frame)
    _,_,transform,_=ref_inputs(job)
    inp,_,_=exp.source_inputs(key,exp.CACHE)
    shift=exp.published()[1][key]['translation']
    T=transform.copy()
    T[:3,3]+=np.asarray(shift,dtype=T.dtype)
    camera=np.asarray(inp['camera']['P']) @ np.linalg.inv(T)
    ref=np.asarray(labels['camera']['world_to_pixel_homogeneous'])
    world=[]; pixels=[]
    face_cells=np.zeros((len(labels['faces'])+1,3),int)
    axes=np.zeros(len(face_cells),int); signs=np.zeros(len(axes),int)
    visible_coords=set()
    for face in labels['faces']:
        i=face['id'];normal=np.array(face['normal_world']);ax=int(np.argmax(abs(normal)))
        face_cells[i]=face['xyz_world']; axes[i]=ax; signs[i]=normal[ax]
        for q,uv,vis in zip(face['corners_world'],face['corners_px'],face['corner_visible'],strict=True):
            if vis and tuple(q) not in visible_coords and np.all(np.array(uv)>=0) and np.all(np.array(uv)<ids.shape[::-1]):
                world.append(q);pixels.append(uv);visible_coords.add(tuple(q))
    args=(face_cells,axes,signs,ids,np.asarray(world),np.asarray(pixels))
    return dict(id=key,split=split,source_sha256=frame['files']['rgb.png'],
                reference=assess(ref,*args),estimated=assess(camera,*args))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    splits=[*typed.SUITE.framed.SIZES,'shelter']
    jobs=list(typed.SUITE.jobs(splits))
    selected=[min((j for j in jobs if j[0]==s),key=lambda j:j[1]['files']['rgb.png']) for s in splits]
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(check,selected))
    typed.compact(exp.OUT/'reference-coordinate-audit.json',rows)
