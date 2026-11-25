"""Evaluate single-view models in absolute world cells and a withheld second view."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from calibration.world import GROUND
from engine.voxel_models import predict, vote
from engine.voxel_space import project, render


def camera(pose, size):
    yaw, pitch = np.radians(pose['yaw_pitch_deg'])
    forward = np.array([-np.sin(yaw)*np.cos(pitch),-np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
    right = np.array([-np.cos(yaw),0,-np.sin(yaw)])
    up = np.cross(right,forward)
    w,h = size
    f = h / (2*np.tan(np.radians(pose['vertical_fov_deg'])/2))
    A = np.stack([w/2*forward+f*right,h/2*forward-f*up,forward])
    return np.column_stack([A, -A@pose['eye_world']])


def registration(P, pose, size):
    # Establish compass and translation with recorded camera ONLY after prediction.
    # Ground plane is known for evaluation, but no alignment to the target voxels.
    A = camera(pose, size)[:,:3]
    eye = np.array(pose['eye_world'])
    xy = project(P, [[0,0,0],[1,0,0],[0,0,1]])
    rays = np.column_stack([xy,np.ones(3)]) @ np.linalg.inv(A).T
    floor = eye+(GROUND+1-eye[1])/rays[:,1,None]*rays
    vectors = (floor[1:]-floor[0])[:,[0,2]]
    axes = [0,1] if abs(vectors[0,0])+abs(vectors[1,1]) >= abs(vectors[0,1])+abs(vectors[1,0]) else [1,0]
    T = np.eye(4)
    T[:3,:3] = 0
    T[1,1] = 1
    for j, axis in enumerate(axes):
        T[[0,2][axis],[0,2][j]] = np.sign(vectors[j,axis])
    T[:3,3] = np.rint(floor[0])
    return T


def iou(a,b):
    return float(np.count_nonzero(a&b) / max(1,np.count_nonzero(a|b)))


def truth_view(coords, P, size):
    low = coords.min(0)
    grid = np.zeros(coords.max(0)-low+1, bool)
    grid[tuple((coords-low).T)] = True
    return np.asarray(render(grid, np.column_stack([P[:,:3],P[:,:3]@low+P[:,3]]),size))>0


def score(grid, scene, poses, coords):
    size = tuple(scene['mask'].shape[::-1])
    T = registration(scene['P'], poses[0], size)
    predicted = np.floor((np.argwhere(grid)+.5)@T[:3,:3].T+T[:3,3]).astype(int)
    a,b = set(map(tuple,predicted)),set(map(tuple,coords))
    common = len(a&b)
    result = dict(voxel=common/max(1,len(a|b)), precision=common/max(1,len(a)),
                  recall=common/len(b), count=len(a), actual=len(b))
    for which,pose in enumerate(poses):
        P = camera(pose,size)
        truth = truth_view(coords,P,size)
        pred = np.asarray(render(grid,scene['P'] if which==0 else P@T,size))>0
        result[['source','novel'][which]] = iou(pred,truth)
    return result


def review(case, scene, poses, grid, out):
    size = tuple(scene['mask'].shape[::-1])
    T = registration(scene['P'],poses[0],size)
    w,h=size
    sheet = Image.new('RGB',(2*w,2*(h+24)),'#19232b')
    draw = ImageDraw.Draw(sheet)
    for row,(key,pose) in enumerate(zip(('image','other'),poses)):
        P = scene['P'] if row==0 else camera(pose,size)@T
        model = render(grid,P,size,True)
        frame = Image.open(case[key]).convert('RGB').resize(size,Image.Resampling.LANCZOS)
        y = row*(h+24)
        for col,(img,title) in enumerate(((frame,'Minecraft input' if row==0 else 'Minecraft withheld camera'),(model,'Voxel fit' if row==0 else 'Same model from withheld camera'))):
            sheet.paste(img,(col*w,y+24))
            draw.text((col*w+6,y+5),title,fill='white')
    sheet.save(out)


def evaluate(inputs, out, methods, committee=None, select=False):
    out.mkdir(parents=True,exist_ok=True)
    cases = json.loads((inputs/'cases.json').read_text())
    rows = []
    for case in cases:
        name = case['id']
        image = Path(case['image'])
        if not (inputs/(name+'.npz')).exists():
            for method in methods + (['vote'] if committee or select else []):
                rows.append(dict(id=name,building=case['building'],method=method,voxel=0,source=0,novel=0,precision=0,recall=0))
            continue
        with np.load(inputs/(name+'.npz')) as data:
            scene = dict(data)
        models = {m:predict(scene,m) for m in methods}
        if committee:
            models['vote'] = vote([models[m] for m in committee])
        # Ground truth and recorded cameras are opened after inference.
        poses = [json.loads(Path(case[k]).with_suffix('.json').read_text()) for k in ('image','other')]
        blocks = json.loads(image.with_name(name+'-blocks.json').read_text())
        coords = np.array([[b['x'],b['y'],b['z']] for b in blocks if b['block_id']!='air'])
        for m,grid in models.items():
            assert grid.shape==scene['hull'].shape
            rows.append(dict(id=name,building=case['building'],method=m,**score(grid,scene,poses,coords)))
            np.savez_compressed(out/(name+'_'+m+'.npz'),grid=grid)
        print(name,flush=True)
    def means():
        summary={}
        for m in sorted({r['method'] for r in rows}):
            rr = [r for r in rows if r['method']==m]
            groups = sorted({r['building'] for r in rr})
            summary[m] = {k:float(np.mean([np.mean([r[k] for r in rr if r['building']==g]) for g in groups]))
                          for k in ('voxel','novel','source','precision','recall')}
        return summary
    summary = means()
    ranking = lambda m: tuple(summary[m][k] for k in ('voxel','novel','source'))
    order = sorted(methods,key=ranking,reverse=True)
    if select:
        committee = order[:3]
        for case in cases:
            name = case['id']
            if not (inputs/(name+'.npz')).exists():
                continue
            scene=dict(np.load(inputs/(name+'.npz')))
            poses=[json.loads(Path(case[k]).with_suffix('.json').read_text()) for k in ('image','other')]
            blocks=json.loads(Path(case['image']).with_name(name+'-blocks.json').read_text())
            coords=np.array([[b['x'],b['y'],b['z']] for b in blocks if b['block_id']!='air'])
            grid=vote([np.load(out/(name+'_'+m+'.npz'))['grid'] for m in committee])
            rows.append(dict(id=name,building=case['building'],method='vote',**score(grid,scene,poses,coords)))
            np.savez_compressed(out/(name+'_vote.npz'),grid=grid)
        summary=means()
    winner = max(summary,key=ranking)
    result=dict(summary=summary, rows=rows, committee=committee, highest=winner)
    (out/'scores.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--methods',nargs='+',required=True)
    parser.add_argument('--committee',nargs='+')
    parser.add_argument('--select',action='store_true')
    args=parser.parse_args()
    evaluate(args.inputs,args.out,args.methods,args.committee,args.select)
