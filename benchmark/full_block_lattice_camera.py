"""Translation-independent source-camera validation and available partial-fragment pairing opportunities."""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import argparse
import numpy as np
from benchmark import full_block_lattice_run as run
from benchmark.expanded_baseline import SOURCE,frames,read,save
from benchmark.full_block_lattice import line_camera
from benchmark.camera_metrics import registration,truth_camera
from engine.geometry import transform,project
from engine.surface_model import intrinsic


def measure(entry):
    key=entry['id']; ds=entry['dataset']
    out=run.RUN/ds/key
    r=read(out/'result.json')
    if not r['rescued'] and ds!='decorative-builds-v1':
        return
    data=dict(id=key,dataset=ds,split=entry['split'],rescued=r['rescued'])
    ev=read(out/'evidence.json.gz')
    labels=read(Path(entry['frame'])/'labels.json')
    pose=read(Path(entry['frame'])/'camera.json')
    with np.load(Path(entry['frame'])/'labels.npz') as masks:
        raster=masks['face_id']
    if r['rescued']:
        lines=read(run.BASE/ds/key/'handoffs/mlsd.json.gz')['segments']
        linear=line_camera(lines,(1440,900))
        P=np.column_stack((linear,-linear@np.array(ev['eye'])))
        Q=truth_camera(pose)
        T=registration(P,pose)
        def basis(p):
            p=p/np.linalg.norm(p[2,:3])
            focal=np.linalg.norm(p[0,:3]-720*p[2,:3])
            return np.linalg.solve(intrinsic((1440,900),focal),p[:,:3]),focal
        estimate,estimated_f=basis(P)
        reference,true_f=basis(Q)
        cos=(estimate*(reference@T[:3,:3])).sum(0)
        data['orientation_error_deg']=np.degrees(np.arccos(cos.clip(-1,1))).tolist()
        data['focal_ratio']=estimated_f/true_f
        distances=[]
        identified=[]
        predicted=[]
        for a in ev['anchors']:
            q=np.array(a['quad'])
            t=(np.arange(7)+.5)/7
            xy=project(transform(q),np.stack(np.meshgrid(t,t),axis=-1))
            x,y=np.moveaxis(xy.astype(int),-1,0)
            owner,n=Counter(raster[y.clip(0,899),x.clip(0,1439)].ravel()).most_common(1)[0]
            if not owner or n<.8*49:
                continue
            f=labels['faces'][owner-1]; b=labels['blocks'][f['block']-1]
            if b.get('geometry_kind','full_cube')!='full_cube':
                continue
            center=np.array(f['corners_world']).mean(0)
            fitted=np.array(a['xyz']).mean(0)
            distances.append(float(np.linalg.norm(fitted)/np.linalg.norm(center-pose['eye_world'])))
            predicted.append(T[:3,:3]@fitted)
            identified.append(center)
        data['distances']=distances
        extents=[]
        if len(identified)>=2:
            actual=np.ptp(np.array(identified),axis=0); fitted=np.ptp(np.array(predicted),axis=0)
            for axis in np.flatnonzero(actual>=1.5):
                extents.append(dict(axis=int(axis),expected=float(actual[axis]),estimated=float(fitted[axis]),
                                    error_blocks=float(abs(fitted[axis]-actual[axis]))))
        data['matched_anchor_span']=extents
    if ds=='decorative-builds-v1':
        doc=read(out/'grouped-observations.json.gz')
        by=defaultdict(list)
        block_id=np.array([0]+[f['block'] for f in labels['faces']])
        for f in doc['fragments']:
            points=f['native_pixel_indices']
            if not points:
                continue
            native=raster.ravel()[points]
            owners=block_id[native]
            owner,n=Counter(owners.tolist()).most_common(1)[0]
            if not owner or n/len(points)<.65 or labels['blocks'][owner-1].get('geometry_kind','full_cube')=='full_cube':
                continue
            face=Counter(native[owners==owner].tolist()).most_common(1)[0][0]
            by[owner].append((face,f['surface'],tuple(f['local'])))
        diag=Counter()
        for observations in by.values():
            matches=[(a,b) for a,b in combinations(observations,2) if a[0]!=b[0] and a[1]!=b[1]]
            if matches:
                diag['partial_with_independent_faces']+=1
                separation=min(sum(abs(x-y) for x,y in zip(a[2],b[2])) for a,b in matches)
                diag['same_cell' if separation==0 else 'one_cell_apart' if separation==1 else 'two_or_more_apart']+=1
        data['partial']=dict(diag)
    return data


def main():
    with ProcessPoolExecutor(2) as pool:
        rows=[r for r in pool.map(measure,list(frames(SOURCE,run.COHORTS))) if r]
    out={}
    for dataset in run.COHORTS:
        group=[r for r in rows if r['dataset']==dataset]; cases=[r for r in group if r['rescued']]
        degrees=np.array([r['orientation_error_deg'] for r in cases])
        lengths=[e['error_blocks'] for r in cases for e in r['matched_anchor_span']]
        pt=Counter()
        for r in group: pt.update(r.get('partial',{}))
        out[dataset]=dict(rescued=len(cases),median_max_axis_degrees=float(np.median(degrees.max(1))),
            median_focal_ratio=float(np.median([r['focal_ratio'] for r in cases])),
            median_anchor_range_ratio=float(np.median([d for r in cases for d in r['distances']])),
            anchor_range_n=sum(len(r['distances']) for r in cases),
            number_measurable_anchor_spans=len(lengths),median_anchor_span_abs_error_blocks=float(np.median(lengths)) if lengths else None,
            partial_fragment_opportunities=dict(pt))
    save(run.REVIEW/'camera-and-span.json',dict(summary=out,cases=rows))


if __name__=='__main__':
    main()
