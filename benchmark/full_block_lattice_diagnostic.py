"""Grader-only known lattice, per-anchor audits, and native visible subface assignment controls."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark import full_block_lattice_run as run
from benchmark.expanded_baseline import SOURCE,frames,read,save
from benchmark.camera_metrics import truth_camera
from benchmark.full_block_lattice import known_anchors, mask_parts, occupancy, sprite_candidates
from benchmark.typed_structures import typed_pack
from engine.geometry import project, transform
from calibration.world import GROUND


def is_partial(kind):
    return kind != 'full_cube'


def audit(anchors, labels, face_ids):
    blocks,faces=labels['blocks'],labels['faces']
    out=[]
    for a in anchors:
        q=np.array(a['quad'])
        t=(np.arange(9)+.5)/9
        xy=project(transform(q),np.stack(np.meshgrid(t,t),axis=-1))
        xx,yy=np.moveaxis(np.floor(xy).astype(int),-1,0)
        sample=face_ids[yy.clip(0,face_ids.shape[0]-1),xx.clip(0,face_ids.shape[1]-1)]
        name,votes=Counter(sample.ravel()).most_common(1)[0]
        row=dict(surface=a['surface'],owner_fraction=round(votes/81,3),full=False,unit=False,exact_id=False)
        if name:
            ref=faces[name-1]
            block=blocks[ref['block']-1]
            actual=np.array(ref['corners_px'])
            error=min(np.linalg.norm(q-np.roll(order,i,axis=0),axis=1).mean()
                      for order in (actual,actual[::-1]) for i in range(4))
            side=np.mean(np.linalg.norm(actual-np.roll(actual,1,axis=0),axis=1))
            row.update(full=not is_partial(block.get('geometry_kind','full_cube')),
                       error_blocks=round(float(error/max(1,side)),4),truth_id=block['block_id'],
                       exact_id='minecraft:'+a.get('block','')==block['block_id'])
            row['unit']=bool(row['full'] and votes/81>=.8 and error/max(1,side)<=.20)
        out.append(row)
    return out


def true_assignment(labels, face_ids, Q):
    # Choose the nearest visible native pixel to each actual subface center. Its signed plane
    # is the only depth input. Register the hit to the half-open occupied cell and decode.
    faces=labels['faces']
    corners=np.array([f['corners_px'] for f in faces])
    centers=corners.mean(1)
    yy,xx=np.nonzero(face_ids)
    ids=face_ids[yy,xx]-1
    delta=(np.column_stack((xx+.5,yy+.5))-centers[ids])**2
    order=np.lexsort((delta.sum(1),ids))
    keys,idx=np.unique(ids[order],return_index=True)
    assert keys.tolist()==list(range(len(faces)))
    x,y=xx[order[idx]],yy[order[idx]]
    normal=np.array([f['normal_world'] for f in faces])
    axis=abs(normal).argmax(1)
    level=np.array([f['corners_world'][0] for f in faces])[keys,axis]
    eye=np.linalg.solve(Q[:,:3],-Q[:,3])
    rays=np.column_stack((x+.5,y+.5,np.ones(len(x))))@np.linalg.inv(Q[:,:3]).T
    hit=eye+rays*((level-eye[axis])/rays[keys,axis])[:,None]
    # Near a face edge one native ray may lie on two candidate cell boundaries. Keep
    # that boundary deterministic by shifting a micron toward the visible face center.
    center=np.array([np.mean(f['corners_world'],axis=0) for f in faces])
    point=np.floor(hit-normal*1e-5+1e-5*(center-hit)).astype(int)
    expected=np.array([f['xyz_world'] for f in faces])
    matches=(point==expected).all(1)
    truth={tuple(b['xyz_world']):b for b in labels['blocks']}
    cells=Counter(map(tuple,point.tolist()))
    pack,decoded=typed_pack({p:'smooth_stone' for p in cells})
    partial={p for p,b in truth.items() if is_partial(b.get('geometry_kind','full_cube'))}
    return dict(faces=len(faces),all_face_owners_exact=int(matches.sum()),occupied=len(cells),correct=len(cells.keys() & truth.keys()),
                missed=len(truth.keys()-cells.keys()),extras=len(cells.keys()-truth.keys()),
                multi_subface_cells=sum(n>=2 for n in cells.values()),
                partial_cells=len(partial),partial_recovered=len(partial & cells.keys()),
                multi_subface_partial_cells=sum(cells[p]>=2 for p in partial)),pack


def one(entry):
    key=entry['id']
    out=run.RUN/entry['dataset']/key
    labels=read(Path(entry['frame'])/'labels.json')
    pose=read(Path(entry['frame'])/'camera.json')
    Q=truth_camera(pose)
    with np.load(Path(entry['frame'])/'labels.npz') as native:
        ids=native['face_id']
    positive,pack=true_assignment(labels,ids,Q)
    (out/'known-visible-subfaces.zip').write_bytes(pack)
    evidence=read(out/'evidence.json.gz')
    anchors=audit(evidence.get('anchors',[]),labels,ids)
    data=dict(id=key,dataset=entry['dataset'],positive=positive,anchors=anchors)
    # Replay actual source observation and occupancy rules with the true metric grid,
    # keeping native face directions/levels/masks unavailable to this arm.
    if entry['dataset']=='decorative-builds-v1' and not run.read(out/'result.json')['baseline']['camera_usable']:
        folder=run.BASE/entry['dataset']/key
        image=Image.open(Path(entry['frame'])/'rgb.png').convert('RGB')
        rgb=np.asarray(image)
        parts=read(folder/'handoffs/placement-input.json.gz')['parts']
        local=np.eye(4)
        local[:3,3]=[0,GROUND+1,0]
        P=Q@local
        P/=np.linalg.norm(P[2,:3])
        linear=P[:,:3]
        eye=np.linalg.solve(linear,-P[:,3])
        scene=mask_parts(parts,image.size,linear)
        proposals=[]
        checked=0
        for part in scene[:8]:
            free=sprite_candidates(part,rgb,linear)
            if free:
                axis=part['axis']
                level=round(eye[axis]+float(np.median([a['xyz'][0][axis] for a in free])))
                proposals.extend(known_anchors(part,rgb,P,eye,range(level-2,level+3)))
                checked+=1
        cells,shared,junctions=occupancy(scene,proposals,eye,linear)
        payload,_=typed_pack(cells)
        (out/'known-camera.zip').write_bytes(payload)
        result,_=run.scorer.evaluate_pack(entry,payload,dict(camera_status='supported',P=P.tolist()),
                                          assignment={p:'fallback' if b=='smooth_stone' else 'source-atlas' for p,b in cells.items()})
        data['known_camera']=dict(geometry=result['geometry'],scores=result['scores'],targets=result['targets'],
                                  checked_planes=checked,anchors=len(proposals),shared=len(shared),junctions=len(junctions),
                                  anchors_audit=audit(proposals,labels,ids))
    save(out/'diagnostic.json',data)
    return key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--summary',action='store_true')
    a=p.parse_args()
    entries=list(frames(SOURCE,run.COHORTS))
    paths=[run.RUN/e['dataset']/e['id'] for e in entries]
    if a.summary:
        rows=[read(d/'diagnostic.json') for d in paths if (d/'diagnostic.json').exists()]
        result={}
        for dataset in run.COHORTS:
            cases=[r for r in rows if r['dataset']==dataset]
            records=[x for row in cases for x in row['anchors']]
            known=[r['known_camera'] for r in cases if 'known_camera' in r]
            result[dataset]=dict(tested=len(cases),true_subface={k:sum(r['positive'][k] for r in cases) for k in cases[0]['positive']} if cases else {},
                anchors=len(records),matched_full_unit=sum(a['unit'] for a in records),
                known_camera_rejected_views=len(known),known_camera_with_export=sum(r['geometry']['strict']['correct']+r['geometry']['strict']['extras']>0 for r in known),
                known_camera={k:run.typed.pooled_scores([r[k] for r in known]) for k in ('geometry','scores')} if known else {})
        save(run.REVIEW/'positive-control.json',result)
        save(run.REVIEW/'anchor-audit.json.gz',rows)
        print(result,flush=True)
        return
    todo=[e for e,d in zip(entries,paths) if (d/'result.json').exists() and not (d/'diagnostic.json').exists()]
    with ProcessPoolExecutor(a.workers) as pool:
        for key in pool.map(one,todo):
            print(key,flush=True)


if __name__=='__main__':
    main()
