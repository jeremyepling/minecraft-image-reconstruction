"""Geometry evaluation. World coordinates and camera poses enter only after prediction."""
import math
import numpy as np
from PIL import Image, ImageDraw

from benchmark.fixtures import clear_sightline
from calibration.projection import camera_project
from engine.geometry import UNIT,project,transform


def enriched(b):
    """Derive additional truth from declared fixture regions and dimensions (idempotent)."""
    b=dict(b); measurements={m['id']:m for m in b['measurements']}
    probes={p['id']:p for p in b['probes']}
    def add(key,a,c,axis):
        measurements[key]=dict(id=key,a=list(a),b=list(c),axis='xyz'[axis],value=abs(a[axis]-c[axis]))
    lo,hi=b['bounds']
    add('overall_width',[lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],0)
    add('overall_height',[lo[0],lo[1],hi[2]],[lo[0],hi[1],hi[2]],1)
    for region in b['regions']:
        if region['kind']=='recess':
            l,h=region['bounds']; foot=[l[0],l[1],h[2]]
            add(region['id']+'_width',foot,[h[0],l[1],h[2]],0)
            add(region['id']+'_height',foot,[l[0],h[1],h[2]],1)
    if b['kind']=='canopy':
        m=measurements['clear_pillar_spacing'];x,y,z=m['a']
        for i in range(m['value']):
            for j in range(measurements['clear_height']['value']):
                key=f'pillar_gap_{i}_{j}'
                probes[key]=dict(id=key,point=[x+i+.5,lo[1]+j+.5,z-.5],role='pillar_gap',occupied=False)
        for side,i in [('left',-1),('right',m['value'])]:
            for j in range(measurements['clear_height']['value']):
                key=f'front_post_{side}_{j}'
                probes[key]=dict(id=key,point=[x+i+.5,lo[1]+j+.5,z-.5],surface=[x+i+.5,lo[1]+j+.5,z],
                                 role='front_post_surface',occupied=True)
    b.update(measurements=list(measurements.values()),probes=list(probes.values()))
    return b


def polygon(canvas,q,scale):
    p=np.asarray(q,dtype=float)/scale
    if np.isfinite(p).all():
        canvas.polygon([tuple(x) for x in p],fill=1)


def facing(plane,pose):
    return np.dot(np.array(pose['eye_world'])-plane['vertices'][0],plane['normal'])>1e-5


def truth_mask(b,pose,size,scale):
    im=Image.new('1',size); d=ImageDraw.Draw(im)
    for plane in b['planes']:
        if facing(plane,pose):
            polygon(d,camera_project(plane['vertices'],pose),scale)
    return np.asarray(im,dtype=bool)


def covers_point(quads,xy):
    if not len(quads):
        return False
    edge=np.roll(quads,-1,axis=1)-quads
    rel=xy-quads
    signed=edge[:,:,0]*rel[:,:,1]-edge[:,:,1]*rel[:,:,0]
    return bool(np.any((signed>=-1e-7).all(1)|(signed<=1e-7).all(1)))


def metric(truth,value,status='measured'):
    error=None if value is None else float(value-truth)
    return dict(truth=truth,prediction=None if value is None else float(value),error=error,
                abs_error=abs(error) if error is not None else None,status=status)


def projected_outline(face,pose):
    verts=np.array(face['vertices']); q=camera_project(verts,pose)
    # For a vertical wall keep the first grid direction horizontal in the world.
    if face['normal'][1]==0:
        top=np.where(verts[:,1]==verts[:,1].max())[0]
        if q[top[0],0]>q[top[1],0]: top=top[::-1]
        others=[next(i for i in range(4) if i not in top and np.all(verts[i,[0,2]]==verts[t,[0,2]])) for t in top]
        return q[[*top,*others[::-1]]]
    return q


def outline_prediction(rgb,b,pose,occupied,estimator):
    """Oracle face outlines, image-estimated counts. Occluded patches do not produce cells."""
    planes=[];cells=[]
    rgb=np.asarray(rgb)
    eye=pose['eye_world']
    for face in b['planes']:
        if not facing(face,pose):continue
        verts=np.array(face['vertices'],float)
        def sample(u,v):
            return verts[0]+u*(verts[1]-verts[0])+v*(verts[3]-verts[0])
        if not any(clear_sightline(eye,sample(u,v),occupied) for u in (.1,.5,.9) for v in (.1,.5,.9)):continue
        q=projected_outline(face,pose)
        area=abs(np.cross(q[1]-q[0],q[3]-q[0]))
        if area<12 or min(np.linalg.norm(q-np.roll(q,1,axis=0),axis=1))<2:continue
        result=estimator(rgb,q)
        counts=[max(1,int(v if np.isscalar(v) else v[0])) for v in result]
        nx,ny=counts
        h=transform(q)@np.diag([1/nx,1/ny,1])
        inv_source=np.linalg.inv(transform(camera_project(verts,pose)))
        name=face['id'];planes.append(dict(name=name,quad=q.tolist(),columns=nx,rows=ny,grid_to_image=h.tolist()))
        indices=np.indices((nx,ny)).reshape(2,-1).T
        quads=project(h,UNIT+indices[:,None])
        centers=quads.mean(1)
        uv=project(inv_source,centers)
        xyz=verts[0]+uv[:,:1]*(verts[1]-verts[0])+uv[:,1:]*(verts[3]-verts[0])
        inside=np.all((centers>=0)&(centers<pose['image_size']),axis=1)
        for ij,cq,point in zip(indices[inside],quads[inside],xyz[inside]):
            if clear_sightline(eye,point.tolist(),occupied):
                cells.append(dict(plane=name,ij=ij.tolist(),quad=cq.tolist()))
    return planes,cells


def inverse_planes(planes):
    found=[]
    for p in planes:
        try:
            inv=np.linalg.inv(p['grid_to_image'])
        except (np.linalg.LinAlgError,KeyError):
            continue
        found.append((p,inv))
    return found


def locate(inverses,xy,margin=.28):
    matches=[]
    for p,inv in inverses:
        uv=project(inv,xy)
        size=np.array([p['columns'],p['rows']])
        if not np.isfinite(uv).all() or np.any(uv < -margin) or np.any(uv > size+margin):continue
        matches.append((p,inv,uv))
    return matches


def automatic(b,mask,scale,planes):
    """Estimate frontal extents and widest recurring internal silhouette gap with no truth landmarks."""
    rows,cols=mask.nonzero(); lo,hi=b['bounds']
    estimates={'width':None,'height':None,'gap':None}
    dominant=None;best=0
    for p in planes:
        q=np.asarray(p['quad']);u=(q[1]-q[0]+q[2]-q[3])*.5;v=(q[3]-q[0]+q[2]-q[1])*.5
        area=abs(u[0]*v[1]-v[0]*u[1])
        if abs(v[1])>abs(v[0])*1.6 and abs(u[0])>abs(u[1])*1.6 and area>best:
            best=area;dominant=(u,v,p)
    if dominant and len(rows):
        u,v,p=dominant
        px=abs(u[0])/p['columns'];py=abs(v[1])/p['rows']
        estimates.update(width=(cols.max()-cols.min()+1)*scale/px,height=(rows.max()-rows.min()+1)*scale/py)
        # Gate on recurring blank spans, dropping single raster-pixel cracks.
        gaps=[]
        for row in mask[rows.min():rows.max()+1]:
            ids=np.flatnonzero(row)
            if len(ids)>1:
                span=np.diff(ids).max()-1
                if span*scale>max(scale,px*.5):gaps.append(span*scale/px)
        if len(gaps)>=max(3,.12*(rows.max()-rows.min()+1)):
            estimates['gap']=np.median(gaps)
    truth=dict(width=hi[0]-lo[0],height=hi[1]-lo[1],gap=b.get('parameters',{}).get('gap'))
    return {key:metric(val,estimates[key],'measured' if estimates[key] is not None else 'no_visible_extent')
            for key,val in truth.items() if val is not None}


def landmark(m,pose,occupied,inverses):
    a,b=np.asarray(m['a'],float),np.asarray(m['b'],float)
    pts=camera_project([a,b],pose)
    def end(value=None,status='measured'):
        return dict(id=m['id'],axis=m['axis'],**metric(m['value'],value,status))
    if np.any(pts<0) or np.any(pts>=pose['image_size']):return end(status='outside_frame')
    direction=(b-a)/np.linalg.norm(b-a);view=(a+b)/2-pose['eye_world'];view/=np.linalg.norm(view)
    if abs(view@direction)>.93:return end(status='foreshortened')
    if not all(clear_sightline(pose['eye_world'],a+t*(b-a),occupied) for t in (.02,.5,.98)):
        return end(status='occluded')
    choices=[]
    for p,inv,uv in locate(inverses,pts):
        d=np.abs(uv[1]-uv[0])
        if d.max()<.1:continue
        off=d.min()/d.max()
        if off>.32:continue
        choices.append((off,-d.max(),float(d.max())))
    return end(min(choices)[-1]) if choices else end(status='unresolved_plane')


def corner_metrics(b,pose,occupied,inverses):
    if b['kind']!='corner':return {}
    out={}
    for m in b['measurements']:
        if m['id'] not in ('flat_seam_line','geometric_corner_line'):continue
        geometric=m['id']=='geometric_corner_line'
        # Look half a block onto each adjacent face at the middle of its shared vertical edge.
        edge=(np.array(m['a'],float)+m['b'])/2
        samples=[edge+[-.55,0,0],edge+([0,0,-.55] if geometric else [.55,0,0])]
        expected='corner' if geometric else 'coplanar'
        result=dict(expected=expected,prediction=None,correct=None,status='occluded')
        if all(clear_sightline(pose['eye_world'],p,occupied) for p in samples):
            sides=[locate(inverses,xy,margin=.06) for xy in camera_project(samples,pose)]
            result['status']='unresolved_plane'
            if all(len(side)==1 for side in sides):
                p,q=[side[0][0] for side in sides]
                result['status']='measured'
                if p['name']==q['name']:guess='coplanar'
                else:
                    dirs=[]
                    for plane in (p,q):
                        corners=np.array(plane['quad'])
                        tangent=(corners[1]-corners[0])+(corners[2]-corners[3])
                        dirs.append(tangent/max(.001,np.linalg.norm(tangent)))
                    guess='corner' if abs(np.dot(*dirs))<math.cos(math.radians(7)) else 'coplanar'
                result.update(prediction=guess,correct=guess==expected)
        out['geometric_corner' if geometric else 'flat_seam']=result
    return out


def score(building,pose,occupied,planes,cells):
    b=enriched(building)
    scale=3
    size=tuple(math.ceil(d/scale) for d in pose['image_size'])
    actual=truth_mask(b,pose,size,scale)
    im=Image.new('1',size);d=ImageDraw.Draw(im)
    for cell in cells:polygon(d,cell['quad'],scale)
    predicted=np.array(im,dtype=bool); union=(actual|predicted).sum()
    inverses=inverse_planes(planes)
    rays=[];posts=[]
    truth_quads=np.asarray([camera_project(p['vertices'],pose) for p in b['planes'] if facing(p,pose)])
    predicted_quads=np.asarray([c['quad'] for c in cells])
    for p in b['probes']:
        if p['role']=='front_post_surface':
            point=p['surface'];xy=camera_project(point,pose)
            clear=clear_sightline(pose['eye_world'],point,occupied)
            posts.append(dict(id=p['id'],occluded=not clear,filled=covers_point(predicted_quads,xy) if clear else None))
            continue
        if p['role']!='pillar_gap':continue
        xy=camera_project(p['point'],pose)
        gt=covers_point(truth_quads,xy) if np.all((xy>=0)&(xy<pose['image_size'])) else None
        # Behind a clear pillar gap a legitimate wall or roof can fill the pixel.
        clear=clear_sightline(pose['eye_world'],p['point'],occupied)
        eligible=clear and gt is False
        rays.append(dict(id=p['id'],occluded=not clear,background=gt is False,evaluable=eligible,
                         filled=covers_point(predicted_quads,xy) if eligible else None))
    eligible=[p for p in rays if p['evaluable']]
    fp=sum(p['filled'] for p in eligible)
    return dict(silhouette_iou=float((actual&predicted).sum()/union) if union else None,
                air_false_fill=fp/len(eligible) if eligible else None,air_evaluable=len(eligible),
                air_false_positive=fp,air_probes=rays,post_probes=posts,automatic=automatic(b,predicted,scale,planes),
                landmarks=[landmark(m,pose,occupied,inverses) for m in b['measurements']],
                corner=corner_metrics(b,pose,occupied,inverses))
