"""Forced exports of observed ground proposals plus paired exact-floor-height interventions."""
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from benchmark.camera_silhouette_round import RUN, REVIEW, CTRL, BASE, WORKTREES, SOURCE, read,save,digest
from benchmark.camera_silhouette_ruler import chart
from benchmark.camera_grid_joint import mask
from benchmark.expanded_baseline import frames
from calibration.world import GROUND


def options(row):
    from benchmark.camera_horizon_replay import replay
    import contextlib
    entry=next(e for e in frames(SOURCE,['visible-builds-v3']) if e['id']==row['id'])
    from pathlib import Path
    folder=Path(entry['frame'])
    source=read(RUN/row['id']/'source.json.gz')
    base=BASE/row['dataset']/row['id']
    pose=read(folder/'camera.json')
    faces=read(folder/'labels.json')['faces']
    parts=read(base/'handoffs/placement-input.json.gz')['parts']
    ref=WORKTREES/'camera-recovery-next/benchmark/camera_relative_calibration.py'
    spec=importlib.util.spec_from_file_location('relative_score',ref)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    seen=set()
    results=[]
    seen_rises=set()
    for i,m in enumerate(source['metric']):
        variants={}
        xy=np.array(source['traces'][m['trace']]['xy'])
        axis=source['traces'][m['trace']]['normal']
        linear=np.array(m['linear'])
        if m['contact'] and m['rise'] not in seen_rises:
            seen_rises.add(m['rise'])
            variants['source-ground']=m['ground_P']
        if m['sprite_count']:
            # Diagnostic uses only the height scalar in the recorded-height arm.
            # The paired unverified arm uses the lower edge of the same mask.
            p=next(p for p in parts if p['id']==m['part'])
            visible=mask(p['support'],tuple(pose['image_size']))
            cols=np.flatnonzero(visible.any(0))[::4]
            ys=visible.shape[0]-1-np.argmax(visible[::-1,cols],axis=0)
            ground=chart(np.column_stack((cols+.5,ys+1)),linear,axis)
            heights={'exact-height':pose['eye_world'][1]-(GROUND+1)}
            if ground is not None:
                heights['unverified-mask-base']=-np.median(ground[:,1])*m['plane_depth']
            for name,height in heights.items():
                eye=-chart(xy,linear,axis)[0]*m['plane_depth']
                eye[1]=height
                variants[name]=np.column_stack((linear,-linear@eye)).tolist()
        for name,P in variants.items():
            key=(name,tuple(np.round(np.asarray(P).ravel(),5)))
            if key in seen:
                continue
            seen.add(key)
            r=dict(id=row['id'],split=row['split'],trace=m['trace'],rise=m['rise'],method=name,P=P)
            try:
                r['native']=helper.native_corner_score(P,pose,faces)
            except ValueError as exc:
                r['native_error']=str(exc)
            out=RUN/row['id']/('forced-'+str(i)+'-'+name)
            out.mkdir(parents=True,exist_ok=True)
            cam=dict(P=P,scale_status='supported',camera_source='silhouette-ruler',reason='diagnostic camera')
            with (out/'replay.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                if (out/'build-manifest.json').exists() and (out/'reconstruction.zip').exists():
                    from benchmark.camera_horizon_replay import camera_validation
                    from benchmark.expanded_scores import score_build
                    assert read(out/'build-manifest.json')['placement']['P']==P
                    r['case']=score_build(entry,out)
                    r['case']['camera_validation']=camera_validation(entry,cam)
                else:
                    r['case']=replay(entry,cam,base,out)
            results.append(r)
    save(RUN/row['id']/'forced.json.gz',results)
    return len(results)


if __name__=='__main__':
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MCIR_IMAGE_ONLY='1')
    rows=[r for r in read(CTRL/'independent-matched-cases.json.gz') if not r['camera_usable']]
    with ProcessPoolExecutor(6) as pool:
        print(sum(pool.map(options,rows)))

