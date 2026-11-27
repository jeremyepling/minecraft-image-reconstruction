"""Collect source-supported outer cube corners from one counted face and a thinner neighbor."""
from functools import lru_cache

import numpy as np

from engine.pillar_seams import native_mask, orientation
from engine.visible_placement import _frontmost_cells, _sample_face
from engine.visible_raster import SAMPLE_STRIDE


def project(xyz, proj):
    uvz = xyz @ proj[:, :3].T + proj[:, 3]
    return uvz[..., :2] / uvz[..., 2:3], uvz[..., 2]


def indices(xy, shape):
    x, y = np.floor(xy).astype(int).T
    return np.clip(y, 0, shape[0]-1), np.clip(x, 0, shape[1]-1)


def source_faces(parts, counts, image, rotation):
    owners = np.zeros((3, image.height, image.width), dtype=np.int16)
    measured = {r['id']:r for r in counts['surfaces']}
    hosts = {}
    for idx, part in enumerate(parts, 1):
        normal = part.get('normal_camera')
        if normal is None or not part.get('support'):
            continue
        direction = rotation.T @ normal
        axis = int(abs(direction).argmax())
        if abs(direction[axis]) < .70:
            continue
        owners[axis][native_mask(part, image.size)] = idx
        row = measured.get(part['id'], {})
        n = row.get('counts')
        if n and any(v is not None and v >= .8 for v in n) and row.get('quad') is not None:
            hosts[idx] = row
    return owners, hosts


def propose(image, parts, counts, camera, cells):
    """No labels or world geometry. Row cell, host/side indices and edge are in estimated-camera coordinates."""
    if not cells or camera.get('P') is None:
        return []
    projection, rotation = orientation(camera, image.size)
    inverse = np.linalg.inv(projection[:, :3])
    eye = -inverse @ projection[:, 3]
    maps, hosts = source_faces(parts, counts, image, rotation)
    if not hosts:
        return []
    offsets = [(0,0,0)] + [tuple(step if j==axis else 0 for j in range(3))
                            for axis in range(3) for step in (-2,-1,1,2)]
    coords = sorted({(x+i,y+j,z+k) for x,y,z in cells for i,j,k in offsets if y+j>=0})
    xyz = np.asarray(coords, dtype=float)
    shape = (image.height//SAMPLE_STRIDE, image.width//SAMPLE_STRIDE)
    old = sorted(cells)
    owner, axis_map = _frontmost_cells(old, projection, eye, shape, face_axes=True)
    footprint=np.bincount(owner[owner>=0],minlength=len(old))
    dep = np.full(shape, np.inf)
    y,x = np.nonzero(owner>=0)
    normal = axis_map[y,x]
    rays = np.column_stack(((x+.5)*SAMPLE_STRIDE,(y+.5)*SAMPLE_STRIDE,np.ones(len(y)))) @ inverse.T
    positions = np.asarray(old)[owner[y,x], normal]
    dep[y,x] = (positions + (eye[normal] > positions+1) - eye[normal]) / rays[np.arange(len(y)),normal]
    u,v=np.meshgrid(np.linspace(.07,.93,5),np.linspace(.07,.93,5))
    samples = np.stack((u.ravel(),v.ravel()),axis=1)
    faces=[]
    for axis in range(3):
        signs = eye[axis] > xyz[:,axis]+1
        base=xyz.copy()
        base[:,axis]+=signs
        delta = np.zeros((len(samples),3))
        free=[i for i in range(3) if i!=axis]
        delta[:, free] = samples
        xy,z=project(base[:,None,:]+delta,projection)
        yy,xx = indices(xy.reshape(-1,2),maps.shape[1:])
        ids=maps[axis,yy,xx].reshape(z.shape)
        nearby = dep[np.minimum(yy//SAMPLE_STRIDE,shape[0]-1),np.minimum(xx//SAMPLE_STRIDE,shape[1]-1)].reshape(z.shape)
        visible = z <= nearby+.08
        winner=np.zeros(len(xyz),int)
        support=np.zeros(len(xyz),int)
        for column in (12, 2, 10, 14, 22, 0, 4, 20, 24):
            label=ids[:,column]
            hits=(ids==label[:,None]).sum(1)
            use=(label>0)&(hits>support)
            winner[use],support[use]=label[use],hits[use]
        width = np.linalg.norm(xy[:,0]-xy[:,4],axis=1)
        height = np.linalg.norm(xy[:,0]-xy[:,20],axis=1)
        area=abs(np.cross(xy[:,4]-xy[:,0],xy[:,20]-xy[:,0]))
        thickness=area/np.maximum(width,height).clip(min=.001)
        seen=(visible & (ids==winner[:,None])).sum(1)
        faces.append((base,signs,winner,support/25,seen/25,thickness,area))
    @lru_cache(maxsize=None)
    def pitch_error(host,axis,level):
        row=hosts[host]
        quad=np.asarray(row['quad'])
        rays=np.column_stack((quad,np.ones(4))) @ inverse.T
        corners=eye+(level-eye[axis])*rays/rays[:,axis,None]
        edges=np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1)
        lengths=(edges[:2]+edges[2:])/2
        known=np.array([v is not None and v >= .8 for v in row['counts']])
        measured=np.array([v or 0 for v in row['counts']])
        return float(np.max(abs(lengths[known] / measured[known]-1)))
    rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
    rows=[]
    for host_axis,(_,hs,hi,hcov,hseen,hw,ha) in enumerate(faces):
        for side_axis,(_,ss,si,scov,sseen,sw,sa) in enumerate(faces):
            if side_axis==host_axis:
                continue
            remaining=3-host_axis-side_axis
            possible=(np.isin(hi,list(hosts)) & (hcov>=.60) & (hseen>=.20) & (si>0)
                      & (scov>=.16) & (sseen>=.08) & (hw>=4.) & (sw>=.8)
                      & ((sw<=.68*hw)|(scov<=.64)) & (sa<1.5*ha))
            for i in np.flatnonzero(possible):
                host=int(hi[i]);side=int(si[i])
                level=int(xyz[i,host_axis]+hs[i])
                error=pitch_error(host,host_axis,level)
                if error>.4:
                    continue
                a=xyz[i].copy()
                a[host_axis]+=hs[i];a[side_axis]+=ss[i]
                b=a.copy(); b[remaining]+=1
                endpoints,_=project(np.array([a,b]),projection)
                vec=endpoints[1]-endpoints[0]
                length=np.linalg.norm(vec)
                if length<4:continue
                line=endpoints[0]+np.linspace(.10,.9,12)[:,None]*vec
                normal=np.array([-vec[1],vec[0]])/length
                face_center=xyz[i]+.5;face_center[host_axis]=level
                host_center,_=project(face_center,projection)
                if (host_center-endpoints.mean(0))@normal<0:normal=-normal
                dist=min(2.5,max(.8,sw[i]*.24))
                def sample(field,offset):
                    yy,xx=indices(line+offset*normal,field.shape)
                    return field[yy,xx]
                paired=(sample(maps[host_axis],dist)==host)&(sample(maps[side_axis],-dist)==side)
                if paired.mean()<.25:continue
                def contrast(offset):
                    return np.abs(sample(rgb,offset+dist)-sample(rgb,offset-dist)).mean(1)
                step=contrast(0)
                relief=step-np.maximum(contrast(3*dist),contrast(-3*dist))
                cell=coords[i]
                neighbors=sum(tuple(x+(d if k==ax else 0) for k,x in enumerate(cell)) in cells
                              for ax in range(3) for d in (-1,1))
                conflict=None
                if cell not in cells:
                    targets=[]
                    wrong=[]
                    for ax,sign,sid in [(host_axis,hs[i],host),(side_axis,ss[i],side)]:
                        hit=_sample_face(projection,cell,ax,1 if sign else -1,shape)
                        if hit is None:continue
                        box,poly,*_=hit
                        yy,xx=np.nonzero(poly)
                        yy+=box[0].start;xx+=box[1].start
                        include=maps[ax,yy*SAMPLE_STRIDE+SAMPLE_STRIDE//2,
                                         xx*SAMPLE_STRIDE+SAMPLE_STRIDE//2]==sid
                        ids=owner[yy[include],xx[include]]
                        targets.extend(ids[ids>=0])
                        err=ids[(ids>=0)&(axis_map[yy[include],xx[include]]!=ax)]
                        wrong.extend(err)
                    if targets:
                        histogram=np.bincount(targets,minlength=len(old))
                        nearby=(np.abs(np.array(old)-xyz[i]).max(1)<=2)
                        target=int(np.argmax(histogram*nearby))
                        if histogram[target] and nearby[target]:
                            conflict=dict(cell=old[target],rays=int(histogram[target]),
                                fraction=round(float(histogram[target]/footprint[target]),4),
                                wrong_axis=round(float(wrong.count(target)/histogram[target]),4))
                rows.append(dict(cell=cell,existing=cell in cells,host=parts[host-1]['id'],
                    side=parts[side-1]['id'],host_axis=host_axis,side_axis=side_axis,
                    edge=endpoints.round(3).tolist(),conflict=conflict,pitch_error=round(error,4),host_fraction=hcov[i],
                    side_fraction=scov[i],host_visible=hseen[i],side_visible=sseen[i],
                    shared=round(float(paired.mean()),4),contrast=round(float(np.median(step)),3),
                    edge_peak=round(float(np.median(relief)),3),neighbors=neighbors,
                    host_width=round(hw[i],2),side_width=round(sw[i],2),
                    count_axes=sum(v is not None and v>=.8 for v in hosts[host]['counts']),
                    known_phase=sum(v is not None for v in hosts[host].get('phase',[]))))
    return rows
