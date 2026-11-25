"""Audit whether two supplied query pixels have visible evidence for a depth return."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from benchmark.fixtures import clear_sightline
from calibration.projection import camera_project
from engine.geometry import project


def undirected_angle(a,b):
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(a@b)/max(1e-9,denom),0,1))))


def quad_area(q):
    return abs(np.dot(q[:,0],np.roll(q[:,1],1))-np.dot(q[:,1],np.roll(q[:,0],1)))/2


def grid_basis(p):
    q=np.asarray(p['quad'])
    return [(q[1]-q[0]+q[2]-q[3])/2,(q[3]-q[0]+q[2]-q[1])/2]


def inside_grid(p,points,margin):
    try:
        uv=project(np.linalg.inv(p['grid_to_image']),points)
    except np.linalg.LinAlgError:
        return np.zeros(len(points),bool)
    return ((uv>=-margin)&(uv<=np.array([p['columns'],p['rows']])+margin)).all(axis=1)


def evidence(planes,points):
    """No camera, ground truth, scale, numeric query depth, image labels or fit feedback."""
    points=np.asarray(points,float)
    delta=points[1]-points[0]
    if np.linalg.norm(delta)<3:return 'unknown'
    faces=[p for p in planes if quad_area(np.asarray(p['quad']))>=9]
    upright=[p for p in faces if abs(grid_basis(p)[1][1])>1.25*abs(grid_basis(p)[1][0])]
    if not upright:return 'unknown'
    main=max(upright,key=lambda p:quad_area(np.asarray(p['quad'])))
    front=grid_basis(main)
    if min(undirected_angle(delta,d) for d in front)<12:return 'unknown'
    same=[p for p in faces if all(undirected_angle(a,b)<10 for a,b in zip(front,grid_basis(p)))]
    # A return must join a facade at an endpoint and carry the query along an
    # independently predicted grid direction. Merely straddling a color seam is insufficient.
    if not any(inside_grid(p,points,.35).any() for p in same):return 'unknown'
    interior=points[:1]+np.array([[.20],[.50],[.80]])*delta
    for p in faces:
        basis=grid_basis(p)
        if not inside_grid(p,interior,.08).all():continue
        for d in basis:
            if undirected_angle(delta,d)<=12 and min(undirected_angle(d,f) for f in front)>=15:
                return 'visible'
    return 'unknown'


def truth(measure,pose,occupied):
    a,b=np.asarray(measure['a'],float),np.asarray(measure['b'],float)
    direction=(b-a)/np.linalg.norm(b-a)
    ray=(a+b)/2-pose['eye_world'];ray/=np.linalg.norm(ray)
    xy=camera_project([a,b],pose)
    if np.any(xy<0) or np.any(xy>=pose['image_size']):return 'outside'
    if abs(ray@direction)>.93:return 'foreshortened'
    if not all(clear_sightline(pose['eye_world'],a+t*(b-a),occupied) for t in (.02,.5,.98)):
        return 'occluded'
    return 'visible'


def controls():
    def panel(name,q,nx,ny):
        from engine.geometry import transform
        return dict(name=name,quad=q,columns=nx,rows=ny,
                    grid_to_image=transform(np.asarray(q,float))@np.diag([1/nx,1/ny,1]))
    front=panel('a',[[0,10],[80,10],[80,70],[0,70]],8,6)
    side=panel('b',[[80,10],[104,0],[104,60],[80,70]],3,6)
    assert evidence([front,side],[[80,40],[104,30]])=='visible'
    assert evidence([front,side],[[80,40],[80.5,39.9]])=='unknown'
    assert evidence([front],[[10,40],[70,40]])=='unknown'
    assert evidence([front,side],[[10,40],[35,30]])=='unknown'


def summary(records):
    counts=Counter((r['truth'],r['decision']) for r in records)
    tp=counts['visible','visible']; fn=counts['visible','unknown']
    fp=sum(n for (truth,p),n in counts.items() if truth!='visible' and p=='visible')
    tn=len(records)-tp-fp-fn
    return dict(queries=len(records),true_visible=tp,false_visible=fp,true_unknown=tn,false_unknown=fn,
                visible_precision=tp/(tp+fp) if tp+fp else None,visible_recall=tp/(tp+fn) if tp+fn else None,
                truth_counts=dict(Counter(r['truth'] for r in records)))


def audit(roots,out,selection):
    load=lambda p:json.loads(p.read_text())
    controls()
    winner=load(selection)['selected'];rows=[]
    for root in roots:
        buildings={b['page']:b for b in load(root/'buildings.json')}
        occupied=defaultdict(set)
        for c in load(root/'layout.json')['blocks']:occupied[c['page']].add((c['x'],c['y'],c['z']))
        for v in load(root/'views.json'):
            b=buildings[v['page']]
            if v['split']!='test':continue
            depths=[m for m in b['measurements'] if m['axis']=='z']
            if not depths:continue
            pose=load(root/'captures'/(v['id']+'.json'))
            # All three frozen RGB-only proposal variants are audited. No parameter selection here.
            for method in ['baseline','partition','texel']:
                planes=load(root/'geometry-predictions'/method/(v['id']+'.json'))['planes']
                for m in depths:
                    xy=camera_project([m['a'],m['b']],pose)
                    rows.append(dict(dataset=root.name,building=b['id'],kind=b['kind'],angle=v['angle'],
                                     query=m['id'],method=method,truth=truth(m,pose,occupied[v['page']]),
                                     decision=evidence(planes,xy)))
    by_method={m:summary([r for r in rows if r['method']==m]) for m in sorted({r['method'] for r in rows})}
    selected=[r for r in rows if r['method']==winner]
    report=dict(selected=winner,scope='Held-out buildings, all annotated Z-axis segments and four angles; query endpoints supplied as image pixels.',
                truth_rule='Outside frame, then absolute viewing-axis cosine >0.93, then ray obstruction at 2%, 50%, 98%; all remaining queries visible.',
                cue_rule='Requires a distinct grid direction on a predicted return polygon containing the query, meeting a facade at one endpoint. Thresholds fixed before the test audit.',
                methods=by_method,by_kind={k:summary([r for r in selected if r['kind']==k]) for k in sorted({r['kind'] for r in selected})},
                by_angle={k:summary([r for r in selected if r['angle']==k]) for k in sorted({r['angle'] for r in selected})},
                records_file=out.stem+'-cases.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.with_name(report['records_file']).write_text(json.dumps(rows,indent=1))
    out.write_text(json.dumps(report,indent=1))
    print(winner,by_method[winner])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    opts=p.parse_args();audit(opts.roots,opts.out,opts.selection)
