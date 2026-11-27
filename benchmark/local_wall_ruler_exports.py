"""Replay strictly supported grid cells with the finite photographed wall-plane choices."""
from benchmark.local_wall_rulers import ARMS, CONTROLS, OUT, WORK, inputs, read
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.camera_horizon_replay import replay
from benchmark.expanded_baseline import COHORTS, SOURCE, frames, save
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
from engine import visible_placement
from pathlib import Path
import argparse
import numpy as np
import os

os.environ['MCIR_IMAGE_ONLY']='1'


def entries():
    return {e['id']:e for e in list(frames(SOURCE,COHORTS))+list(fixed.fixed_frames())}


def options(f):
    rows=read(OUT/(f['id']+'.json'))['records']
    groups={}
    for arm in [*ARMS,'native_mixed']:
        signature=tuple((r['surface'],r['axis'],r['arms'][arm]['level']) for r in rows
                        if r['arms'][arm]['accepted'] and r['arms'][arm]['legal_plane'])
        groups.setdefault(signature,[]).append(arm)
    return groups


def build(task):
    f,e,sig,names=task
    key=f['id']
    out=OUT/'exports'/key/names[0]
    receipt=out/'case.json'
    if receipt.exists():
        return
    site=visible_placement._carriers
    planes={k:(a,l) for k,a,l in sig}
    events=[]
    _,status=inputs(f)
    eye=-np.linalg.solve(np.array(status['P'])[:,:3],np.array(status['P'])[:,3])
    def carriers(image,parts,masks,P):
        result=site(image,parts,masks,P)
        for part,mask in zip(parts,masks,strict=True):
            if part['id'] in planes:
                axis,level=planes[part['id']]
                plane=(axis,level,int(np.sign(eye[axis]-level)))
                old=next((r[1] for r in result if r[0] is mask),None)
                if old==plane:
                    continue
                result=[r for r in result if r[0] is not mask]
                result.insert(0,(mask,plane,'source atlas wall grid'))
                events.append(dict(id=part['id'],old=old,new=plane))
        return result
    out.mkdir(exist_ok=True,parents=True)
    with patch.object(visible_placement,'_carriers',carriers):
        if e['dataset']=='fixed-104':
            result=fixed.replay_placement(fixed.inference_inputs(e))
            (out/'reconstruction.zip').write_bytes(result.pack)
            case,_=fixed.score_result(e,result)
        else:
            camera=dict(P=status['P'],scale_status='supported',reason=status['reason'],camera_source=status['camera_source'])
            case=replay(e,camera,WORK/f['replay']['root'],out)
    save(receipt,dict(arms=names,signature=sig,events=events,case=case))
    print('export',key,names,case['geometry']['strict']['correct'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--ids',nargs='*')
    args=p.parse_args()
    items=entries()
    tasks=[]
    for f in read(CONTROLS)['pilot']['frames']:
        if args.ids and f['id'] not in args.ids:
            continue
        groups=options(f)
        if not any(groups):
            continue
        groups.setdefault((),[]).append('noop')
        for sig,names in groups.items():
            tasks.append((f,items[f['id']],sig,names))
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(build,tasks))
