"""Read building edges, connect their observed spans and anchor wall-to-floor height from full sprites."""
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
try:
    import cv2
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]/'mask-scalelsd/runs/env/Lib/site-packages'))
    import cv2

from benchmark.full_block_lattice import STRIDE, consensus, line_camera, mask_parts, rays, sprite_candidates
from engine.visible_raster import observed_support

cv2.setNumThreads(1)


def outline_lines(image, parts, existing):
    """Native RGB edge spans with nearby observed surface boundaries along their length."""
    w, h = image.size
    shape = h//STRIDE, w//STRIDE
    masks = [observed_support(p['support'], shape, image.size).astype('B') for p in parts if p.get('support')]
    if not masks:
        return [], np.zeros(shape, bool)
    kernel = np.ones((3,3),'B')
    boundary = np.maximum.reduce([cv2.morphologyEx(m, cv2.MORPH_GRADIENT,kernel) for m in masks])
    near = cv2.dilate(boundary, kernel)
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),1.4)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    raw = ([] if detected is None else detected.reshape(-1,4).tolist()) + list(existing)
    records = []
    for line in raw:
        ends = np.asarray(line,float).reshape(2,2)
        d = ends[1]-ends[0]
        length = np.linalg.norm(d)
        if length < max(22, .024*w):
            continue
        samples = np.linspace(*ends,25)
        x,y = np.clip((samples/STRIDE).astype(int),0,np.array(shape[::-1])-1).T
        support = near[y,x].mean()
        if support < .55:
            continue
        records.append((length*support, ends, d/length))
    retained = []
    for _, ends, unit in sorted(records,key=lambda r:-r[0]):
        normal = np.array([-unit[1],unit[0]])
        if any(abs(old_unit@unit)>.998 and abs((old-ends[0])@normal).max()<4
               and np.min(abs((old.mean(0)-ends)@unit)) < np.linalg.norm(old[1]-old[0])*.55
               for old,old_unit in retained):
            continue
        retained.append((ends,unit))
        if len(retained)>=110:
            break
    return [e.ravel().tolist() for e,u in retained], np.maximum.reduce(masks).astype(bool)


def graph(lines, linear):
    """Intersect observed spans at corners and propagate relative depth along their axis."""
    nodes = []
    edges = defaultdict(list)
    tracks = []
    for line in lines:
        p,q = np.reshape(line,(2,2))
        delta = q-p
        length = np.linalg.norm(delta)
        unit = delta/length
        v = linear[:2] - ((p+q)/2)[:,None]*linear[2]
        residual = abs(unit[0]*v[1]-unit[1]*v[0])/np.maximum(1e-9,np.linalg.norm(v,axis=0))
        axis = int(residual.argmin())
        if residual[axis]<.048:
            tracks.append((p,unit,length,axis))
    def node(point):
        for i, xy in enumerate(nodes):
            if np.linalg.norm(point-xy)<=7:
                return i
        nodes.append(point)
        return len(nodes)-1
    stops = [[] for _ in tracks]
    for i,(p,u,length,ax) in enumerate(tracks):
        for j in range(i):
            q,v,other,bx = tracks[j]
            if ax==bx or abs(u[0]*v[1]-u[1]*v[0])<.15:
                continue
            t,s=np.linalg.solve(np.column_stack((u,-v)),q-p)
            if -8<=t<=length+8 and -8<=s<=other+8 and min(abs(t),abs(length-t),abs(s),abs(other-s))<11:
                k=node(p+t*u)
                stops[i].append((t,k))
                stops[j].append((s,k))
    for (p,u,length,axis), hits in zip(tracks,stops):
        if not hits:
            continue
        for t in (0,length):
            if min(abs(t-s) for s,k in hits)>10:
                hits.append((t,node(p+t*u)))
        hits.sort()
        for (_,a),(_,b) in zip(hits,hits[1:]):
            if a!=b:
                edges[a].append((b,axis))
                edges[b].append((a,axis))
    if not nodes:
        return None
    directions=rays(linear,np.array(nodes))
    todo=set(edges)
    best=None
    while todo:
        first=min(todo)
        depths={first:1.}
        queue=deque([first])
        segments=[]
        errors=[]
        while queue:
            a=queue.popleft()
            for b,axis in edges[a]:
                if a<b:
                    segments.append([a,b,axis])
                axis_mask=np.arange(3)!=axis
                da,db=directions[[a,b]][:,axis_mask]
                ratio=(da@db)/max(1e-10,db@db)
                depth=depths[a]*ratio
                if depth<=0:
                    continue
                if b not in depths:
                    depths[b]=depth
                    queue.append(b)
                else:
                    errors.append(abs(depth/depths[b]-1))
        todo.difference_update(depths)
        if len({s[2] for s in segments})<3:
            continue
        segments=[s for s in segments if s[0] in depths and s[1] in depths]
        if best is None or len(segments)>len(best['edges']):
            ids=sorted(depths)
            index={k:i for i,k in enumerate(ids)}
            best=dict(xy=[nodes[i].tolist() for i in ids],
                      xyz=[(directions[i]*depths[i]).tolist() for i in ids],
                      edges=[[index[a],index[b],axis] for a,b,axis in segments],
                      cycle_error=float(max(errors,default=0)))
    return best


def count_scale(row,linear,axis):
    """Use the measured repeat independently verified on held-out native texture bands."""
    evidence=row.get('evidence',{})
    check=evidence.get('texture_check') or {}
    if check.get('held_repeat',0)<.50 or check.get('drift',1)>.12:
        return None
    ray=rays(linear,np.asarray(row['quad']))
    if np.min(abs(ray[:,axis]))<.08 or np.any(ray[:,axis]*ray[0,axis]<=0):
        return None
    on_plane=ray/abs(ray[:,[axis]])
    distances=[]
    for dim,info in enumerate(evidence.get('axes',[])):
        if not info or info.get('source')!='repeated texture' or info.get('count',0)<2.4:
            continue
        a,b=(0,1) if dim==0 else (0,3)
        c,d=(3,2) if dim==0 else (1,2)
        edge=np.linalg.norm(on_plane[b]-on_plane[a])
        opposite=np.linalg.norm(on_plane[d]-on_plane[c])
        if abs(edge-opposite)>.16*max(edge,opposite) or min(edge,opposite)<.005:
            continue
        distances.append(info['count']/((edge+opposite)*.5))
    if not distances or max(distances)>1.15*min(distances):
        return None
    return float(np.sign(ray[0,axis])*np.median(distances))


def count_proposals(parts,counts,scaffold,linear,union,stats,tentative):
    by_id={r['id']:r for r in counts.get('surfaces',[])}
    xy=np.asarray(scaffold['xy'])
    skeleton=np.asarray(scaffold['xyz'])
    bottom=np.where(union,np.arange(union.shape[0])[:,None],-1).max(0)
    incidence=defaultdict(set)
    for a,b,d in scaffold['edges']:
        incidence[a].add(d)
        incidence[b].add(d)
    candidates=[]
    for part in parts:
        axis=part['axis']
        if axis==1 or part['id'] not in by_id:
            continue
        plane=count_scale(by_id[part['id']],linear,axis)
        if plane is None:
            continue
        stats['validated_count_planes']+=1
        wall=cv2.dilate(part['mask'].astype('B'),np.ones((5,5),'B'))
        for a,b,d in scaffold['edges']:
            if d==axis or d==1 or np.linalg.norm(xy[a]-xy[b])<36:
                continue
            samples=np.linspace(xy[a],xy[b],15)
            x,y=np.clip((samples/STRIDE).astype(int),0,np.asarray(union.shape[::-1])-1).T
            if np.mean((abs(y-bottom[x])<=2)&(wall[y,x]>0))<.8:
                continue
            stats['count_ground_span']+=1
            relative=rays(linear,xy[[a,b]])
            xyz=relative*plane/relative[:,[axis]]
            height=-xyz[:,1].mean()
            if not .15<height<120 or np.ptp(xyz[:,1])>.25:
                continue
            eye=-xyz[0]
            eye[1]=height
            P=np.column_stack((linear,-linear@eye))
            span=abs(xyz[0,d]-xyz[1,d])
            scale=plane/skeleton[a,axis]
            metric=skeleton*scale+eye
            upper=[k for k in incidence if k not in (a,b) and len(incidence[k])>=2 and abs(metric[k,1])>1.5]
            detail=dict(edge=[a,b],surface=part['id'],end_families=[len(incidence[a]),len(incidence[b])],
                        measured_base_blocks=float(span),integer_residual=float(abs(span-round(span))),
                        other_elevated_corners=len(upper),
                        closest_corner_residual=min((float(np.max(abs(metric[k]-np.rint(metric[k])))) for k in upper),default=None))
            tentative.append(dict(P=P.tolist(),kind='repeated-wall',contact=detail))
            if min(len(incidence[a]),len(incidence[b]))<2:
                continue
            if span<1.7 or abs(span-round(span))>.19:
                continue
            ok=[k for k in upper if np.max(abs(metric[k]-np.rint(metric[k])))<.22]
            if ok:
                stats['count_held_corner']+=1
                candidates.append((0,abs(span-round(span)),eye,[],[a,b],scale))
    return candidates


def infer(image, source_parts, lines, counts):
    structural,union=outline_lines(image,source_parts,lines)
    linear=line_camera(structural,image.size)
    report=dict(structural_lines=structural,linear=None if linear is None else linear.tolist())
    if linear is None:
        return None,dict(report,reason='no three structural axes')
    scaffold=graph(structural,linear)
    if not scaffold:
        return None,dict(report,reason='no connected three-axis scaffold')
    report['scaffold']=scaffold
    if scaffold['cycle_error']>.10:
        return None,dict(report,reason='inconsistent scaffold loop')
    rgb=np.asarray(image)
    parts=mask_parts(source_parts,image.size,linear)
    bottom=np.where(union,np.arange(union.shape[0])[:,None],-1).max(0)
    xy=np.asarray(scaffold['xy'])
    xyz=np.asarray(scaffold['xyz'])
    from collections import Counter
    stats=Counter()
    tentative=[]
    report.update(gates=stats,tentative=tentative)
    candidates=count_proposals(parts,counts,scaffold,linear,union,stats,tentative)
    for part in [p for p in parts if p['axis']!=1][:8]:
        axis=part['axis']
        anchors=sprite_candidates(part,rgb,linear)
        stats['wall_planes_scanned']+=1
        stats['wall_with_any_sprite']+=bool(anchors)
        selected,phase,_=consensus(anchors)
        if len(selected)<2:
            continue
        # Distinct whole sprites must describe a separated integer displacement on the wall.
        stats["plane_with_two_sprites"]+=1
        centers=np.array([np.mean(a['xyz'],axis=0) for a in selected])
        if np.ptp(centers,axis=0).max()<1.7:
            continue
        stats["separated_repeats"]+=1
        plane=float(np.median(centers[:,axis]))
        wall=cv2.dilate(part['mask'].astype('B'),np.ones((5,5),'B'))
        for a,b,d in scaffold['edges']:
            if d==1 or d==axis:
                continue
            samples=np.linspace(xy[a],xy[b],15)
            x,y=np.clip((samples/STRIDE).astype(int),0,np.asarray(union.shape[::-1])-1).T
            if np.linalg.norm(xy[b]-xy[a])<36 or np.mean((abs(y-bottom[x])<=2)&(wall[y,x]>0))<.8:
                continue
            stats["sprite_ground_span"]+=1
            # Both ends of a wall base must agree in independently inferred floor elevation.
            relative=rays(linear,xy[[a,b]])
            wall_xyz=relative*(plane/relative[:,[axis]])
            height=-wall_xyz[:,1].mean()
            if not .15<height<120 or np.ptp(wall_xyz[:,1])>.25:
                continue
            tentative_eye=np.array(phase)
            tentative_eye[1]=height
            tentative.append(dict(P=np.column_stack((linear,-linear@tentative_eye)).tolist(),kind="native-sprite"))
            floor_err=float(abs((height-phase[1]+.5)%1-.5))
            if floor_err>.22:
                continue
            stats["sprite_ground_phase"]+=1
            # Independently check a turning corner at a junction of at least two directions.
            checks=[]
            for k in (a,b):
                incident={dim for s,t,dim in scaffold['edges'] if k in (s,t)}
                if len(incident)<2:
                    continue
                v=wall_xyz[int(k==b)]+phase
                free=3-axis-1
                checks.append(float(abs((v[free]+.5)%1-.5)))
            if not checks or min(checks)>.20:
                continue
            eye=np.array(phase)
            eye[1]=height
            source_scale=plane/xyz[a,axis]
            if source_scale<=0:
                continue
            candidates.append((len(selected),min(checks)+floor_err,eye,selected,[a,b],source_scale))
    if not candidates:
        return None,dict(report,reason='no verified sprite scale and grounded scaffold junction')
    candidates.sort(key=lambda c:(-c[0],c[1]))
    _,_,eye,anchors,edge,scale=candidates[0]
    if any(abs(c[2][1]-eye[1])>max(.55,.10*eye[1]) for c in candidates):
        return None,dict(report,reason='incompatible wall scales')
    P=np.column_stack((linear,-linear@eye))
    report.update(reason='grounded scaffold and native sprite agreement' if anchors else 'grounded scaffold and verified wall repeats',anchors=anchors,base_edge=edge,scale=float(scale))
    return dict(P=P.tolist(),reason=report['reason'],scale_status='supported',camera_source='structure-scaffold'),report
