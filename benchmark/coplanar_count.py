"""Pass an observed texture pitch and a boundary-anchored lattice across a candidate common plane."""
import numpy as np

from engine.geometry import UNIT, project
from engine.surface_model import lens

SETTINGS = dict(normal_cosine=.975, minimum_repeat=.65, minimum_cycles=2.,
                minimum_inverse_depth=.03, depth_log_tolerance=.16, max_gap_blocks=8.,
                donor_grid_residual=.30, recipient_grid_residual=.28,
                max_pitch_ratio=1.23, max_already_measured_error=.45)


def boundary(part,size):
    y,lo,hi=np.asarray(part['support']['runs']).T
    xy=np.column_stack((np.r_[lo,hi,hi,lo],np.r_[y,y,y+1,y+1]))*np.asarray(size)/part['support']['size']
    widths=hi-lo
    center=np.average(np.c_[(lo+hi)/2,y+.5],axis=0,weights=widths)*np.asarray(size)/part['support']['size']
    return xy,np.r_[center,1.]


def plane_frame(normal,quad,K):
    # Lift the donor's chart directions onto its normal plane to obtain two perpendicular metric axes.
    rays=np.c_[quad,np.ones(4)]@np.linalg.inv(K).T
    incidence=rays@normal
    if max(incidence)>-.10:
        return None
    points=-rays/incidence[:,None]
    across=(points[1]-points[0])+(points[2]-points[3])
    if np.linalg.norm(across)<1.e-6:
        return None
    across/=np.linalg.norm(across)
    down=np.cross(normal,across)
    if down@(points[3]-points[0])<0:
        down=-down
    return K@np.column_stack((across,down,-normal))


def valid_uv(inverse,xy):
    rays=xy@inverse[:,:2].T+inverse[:,2]
    if rays[:,2].min()<=.02:
        return None
    return rays[:,:2]/rays[:,2,None]


def seed(part,sample,base,xy,K,settings):
    if sample is None or part.get('relative_inverse_depth') is None:
        return None
    normal=np.asarray(part.get('normal_camera'),float)
    if normal.shape!=(3,) or np.linalg.norm(normal)<.5:
        return None
    normal=normal/np.linalg.norm(normal)
    quad=np.asarray(sample['quad'])
    frame=plane_frame(normal,quad,K)
    if frame is None:
        return None
    inv=np.linalg.inv(frame)
    extent=valid_uv(inv,xy)
    chart=valid_uv(inv,quad)
    if extent is None or chart is None:
        return None
    lengths=(np.abs(chart[2]-chart[0])+np.abs(chart[1]-chart[3]))/2
    pitches=[]
    strengths=[]
    for axis,record in enumerate(base.get('evidence',{}).get('axes',[])):
        if (record and record['source']=='repeated texture' and record['count']>=settings['minimum_cycles']
                and record['agreement']>=settings['minimum_repeat']):
            strengths.append(record['agreement'])
            pitches.append(lengths[axis]/record['count'])
    if not pitches or min(pitches)<=0 or max(pitches)/min(pitches)>settings['max_pitch_ratio']:
        return None
    pitch=float(np.median(pitches))
    lo,hi=extent.min(0),extent.max(0)
    dims=(hi-lo)/pitch
    if max(abs(dims-dims.round()))>settings['donor_grid_residual'] or dims.min()<.6:
        return None
    block_to_image=frame@np.array([[pitch,0,lo[0]],[0,pitch,lo[1]],[0,0,1.]])
    return dict(donor=part['id'],normal=normal,depth=part['relative_inverse_depth'],grid=block_to_image,
                inverse=np.linalg.inv(block_to_image),dims=dims,confidence=min(strengths))


def predict_frame(photo,parts,samples,baselines,settings=SETTINGS,audit=None):
    K,_=lens([],photo.size)
    Kinv=np.linalg.inv(K)
    bounds=[boundary(p,photo.size) for p in parts]
    seeds=[seed(p,s,b,xy,K,settings) for p,s,b,(xy,_) in zip(parts,samples,baselines,bounds,strict=True)]
    result=[]
    for p,s,base,(xy,centroid) in zip(parts,samples,baselines,bounds,strict=True):
        old=base['counts']
        depth=p.get('relative_inverse_depth')
        if None not in old or depth is None or depth<=settings['minimum_inverse_depth']:
            result.append(base)
            continue
        normal=np.asarray(p.get('normal_camera'),float)
        proposals=[]
        for src,(_,center) in zip(seeds,bounds,strict=True):
            if src is None or src['donor']==p['id'] or normal.shape!=(3,) or normal@src['normal']<settings['normal_cosine']:
                continue
            # A common plane's inverse depth scales with its normal dotted into each pixel ray.
            # The monocular field can have an additive bias, so use this as a veto, never a block distance.
            n=src['normal']
            expected=src['depth']*((Kinv@centroid)@n)/((Kinv@center)@n)
            if expected<=0:
                continue
            residual=abs(np.log(depth/expected))
            if audit is not None:
                audit.append(dict(target=p['id'],donor=src['donor'],depth_residual=float(residual),
                                  depth_pass=bool(residual<=settings['depth_log_tolerance'])))
            if residual>settings['depth_log_tolerance']:
                continue
            uv=valid_uv(src['inverse'],xy)
            if uv is None:
                continue
            lo,hi=uv.min(0),uv.max(0)
            dist=np.linalg.norm(np.maximum(np.maximum(0,-hi),lo-src['dims']))
            if dist>settings['max_gap_blocks']:
                continue
            err=float(np.max(abs(np.r_[lo,hi]-np.r_[lo,hi].round())))
            dims=hi-lo
            if err>settings['recipient_grid_residual'] or min(dims)<.55:
                continue
            quad=project(src['grid'],lo+UNIT*dims).tolist()
            if s is not None:
                original_chart=valid_uv(src['inverse'],np.asarray(s['quad']))
                if original_chart is None:
                    continue
                projected=np.ptp(original_chart,axis=0)
                if any(k is not None and abs(k-a)>settings['max_already_measured_error'] for k,a in zip(old,projected)):
                    continue
            score=err+residual+.01*dist+.15*(1-src['confidence'])
            proposals.append((score,dict(method='coplanar-donor-grid',counts=dims.tolist(),quad=quad,
                phase=np.mod(-lo,1.).tolist(),evidence=dict(transfer=dict(donor=src['donor'],
                recipient_offset=np.rint(lo).astype(int).tolist(), donor_grid_to_image=src['grid'].tolist(),
                depth_log_error=residual,edge_residual=err,gap_blocks=float(dist),raw_bounds=[lo.tolist(),hi.tolist()])))))
        proposals.sort(key=lambda pair:pair[0])
        sizes={tuple(np.rint(pr['counts']).astype(int)) for _,pr in proposals}
        result.append(proposals[0][1] if len(sizes)==1 else base)
    return result
