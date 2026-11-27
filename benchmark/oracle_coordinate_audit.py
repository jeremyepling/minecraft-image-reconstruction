"""Independent pixel-center ray/plane and camera-coordinate audit of the visible-only oracle."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark import typed_structures as typed
from benchmark.pavement_opening_eval import visible_faces
from benchmark.reference_surface_feedback import oracle_parts
from calibration.world import GROUND
from engine.pillar_seams import native_mask

OUT = Path(__file__).resolve().parent/'reviews/material-feedback-surfaces-01'


def xyz_cube_face(point, axis, sign):
    base = np.asarray(point).copy()
    base[axis] += int(sign > 0)
    first, second = np.eye(3,dtype=int)[[i for i in range(3) if i!=axis]]
    return np.array([base, base+first, base+first+second, base+second])


def read_pose(pose):
    full = np.array(pose['view_rotation_projection']).reshape((4,4),order='F')
    eye = np.array(pose['eye_world'])
    width,height = pose['image_size']
    def project(xyz):
        p = (xyz-eye) @ full[:,:3].T + full[:,3]
        return (p[:,:2] / p[:,3,None] * [.5,-.5] + .5) * [width,height]
    proj = full[[0,1,3]]
    inv = np.linalg.inv(proj[:,:3])
    origin = eye - inv @ proj[:,3]
    return project, origin, inv


def direct_rays(pose, states, ground):
    width,height = pose['image_size']
    project,eye,inv = read_pose(pose)
    points = sorted(states)
    point_ids = {p:i for i,p in enumerate(points)}
    result = np.zeros((height,width),np.int32)
    depth = np.full((height,width),np.inf)
    for index,point in enumerate(points):
        for axis in range(3):
            sign = 1 if eye[axis] > point[axis]+1 else -1
            if point[axis] <= eye[axis] <= point[axis]+1:
                continue
            neighbor = list(point)
            neighbor[axis] += sign
            if neighbor[1] <= ground or tuple(neighbor) in states:
                continue
            corners = xyz_cube_face(point, axis, sign)
            xy = project(corners)
            left,top=np.maximum(0,np.floor(xy.min(0)).astype(int))
            right,bottom=np.minimum([width,height],np.ceil(xy.max(0)).astype(int))
            if left>=right or top>=bottom:
                continue
            x=(np.arange(left,right)+.5)*2/width-1
            y=1-(np.arange(top,bottom)+.5)*2/height
            rays=inv[:,0]*x[None,:,None]+inv[:,1]*y[:,None,None]+inv[:,2]
            t=(corners[0,axis]-eye[axis])/rays[...,axis]
            other=[i for i in range(3) if i!=axis]
            pos=(eye-np.array(point))[other]+t[...,None]*rays[...,other]
            inside=(pos[...,0]>=0)&(pos[...,1]>=0)&(pos[...,0]<=1)&(pos[...,1]<=1)
            z=depth[top:bottom,left:right]
            nearest=inside & (t>0) & (t<z)
            z[nearest]=t[nearest]
            result[top:bottom,left:right][nearest]=1+index*6+axis*2+int(sign>0)
    return result,point_ids


def frame(job):
    split, f, _, _=job
    path=typed.source_path(split,f)
    dataset=path.name=='rgb.png'
    folder=path.parent
    def load(name, file):
        return typed.read(typed.checked(folder/file,f['files'][name]))
    pose=load('camera.json','camera.json' if dataset else path.stem+'.json')
    rows=load('world-readback.json','world-readback.json' if dataset else path.stem+'-blocks.json')
    ground=f.get('ground_y',GROUND)
    states={tuple(s[k] for k in 'xyz'):s['block_id'].removeprefix('minecraft:') for s in rows
            if s['block_id'].removeprefix('minecraft:')!='air' and s['y']>ground}
    labels, ids = visible_faces(split,f)
    direct, points = direct_rays(pose,states,ground)
    codes = np.zeros(len(labels['faces'])+1,np.int32)
    sids = np.zeros_like(codes)
    project,_,_ = read_pose(pose)
    groups={}
    face_error=0.
    for face in labels['faces']:
        point=tuple(face['xyz_world'])
        normal=np.array(face['normal_world'])
        axis=np.argmax(abs(normal))
        codes[face['id']]=1+points[point]*6+int(axis)*2+int(normal[axis]>0)
        sids[face['id']]=face['surface_id']
        vertices=np.asarray(face['corners_world'])
        face_error=max(face_error,float(np.abs(project(vertices)-face['corners_px']).max()))
        groups.setdefault(face['surface_id'],[]).append(point)
    direct_blocks={(n-1)//6 for n in np.unique(direct) if n}
    ref_blocks={points[tuple(b['xyz_world'])] for b in labels['blocks']}
    id_errors=sum(states[tuple(b['xyz_world'])]!=b['block_id'].removeprefix('minecraft:') for b in labels['blocks'])
    mask_errors=count_errors=0
    corner_error=0.
    with Image.open(typed.checked(path,f['files']['rgb.png'])) as im:
        parts, dims,_=oracle_parts(im,[],labels,ids,'grids')
        surface=sids[ids]
        for part in parts:
            sid=int(part['id'].split('-')[-1])
            mask_errors+=int(np.count_nonzero(native_mask(part,im.size)!=(surface==sid)))
            cube=np.array(groups[sid])
            face=next(face for face in labels['faces'] if face['surface_id']==sid)
            normal=np.array(face['normal_world']);axis=np.argmax(abs(normal))
            vertices=xyz_cube_face(cube.min(0),axis,normal[axis])
            vertices+=(vertices>cube.min(0)).astype(int)*(cube.max(0)-cube.min(0))
            xy=project(vertices)
            given=np.asarray(part['quad'])
            dist=np.linalg.norm(given[:,None]-xy[None,:],axis=2)
            order=dist.argmin(1)
            assert len(set(order))==4
            corner_error=max(corner_error,float(dist.min(1).max()))
            span=np.abs(np.diff(vertices[order[:3]],axis=0)).sum(1).tolist()
            count_errors+=int(span!=dims[part['id']])
    return dict(id=f['id'],split=split,pixels=int(ids.size),visible_pixels=int((ids!=0).sum()),
                visible_blocks=len(ref_blocks),surfaces=len(parts),faces=len(labels['faces']),
                first_hit_pixel_mismatches=int(np.count_nonzero(direct!=codes[ids])),
                foreground_pixel_mismatches=int(np.count_nonzero((direct!=0)!=(ids!=0))),
                xyz_set_difference=len(direct_blocks^ref_blocks),reference_id_errors=id_errors,
                surface_mask_pixel_mismatches=mask_errors,chart_count_errors=count_errors,
                max_face_corner_error_px=face_error,max_chart_corner_error_px=corner_error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    jobs=list(typed.SUITE.jobs([*typed.SUITE.framed.SIZES,'shelter']))
    with ProcessPoolExecutor(args.workers) as pool:
        cases=list(pool.map(frame,jobs))
    summary={k:(max if k.startswith('max_') else sum)(r[k] for r in cases) for k in cases[0] if k not in ('id','split')}
    typed.compact(OUT/'oracle-coordinate-audit.json',dict(views=len(cases),summary=summary,cases=cases))


if __name__=='__main__':
    main()
