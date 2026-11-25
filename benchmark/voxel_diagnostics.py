"""Inspect frozen single-image predictions and run a separate two-camera control."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from benchmark.voxel_eval import registration, review, truth_view, camera, iou


def diagnostics(inputs, results, selected):
    cases=json.loads((inputs/'cases.json').read_text())
    scored=json.loads((results/'scores.json').read_text())
    groups={}
    probes=[]
    panels=[]
    control=[]
    for case in cases:
        name=case['id']
        path=Path(case['image'])
        metadata=json.loads((path.parents[1]/'buildings.json').read_text())
        building=next(b for b in metadata if b['id']==case['building'])
        poses=[json.loads(Path(case[k]).with_suffix('.json').read_text()) for k in ('image','other')]
        scene=dict(np.load(inputs/(name+'.npz')))
        size=tuple(scene['mask'].shape[::-1])
        T=registration(scene['P'],poses[0],size)
        coords=json.loads(path.with_name(name+'-blocks.json').read_text())
        truth={tuple(b[k] for k in ('x','y','z')) for b in coords if b['block_id']!='air'}
        def world(m):
            cells=np.argwhere(np.load(results/(name+'_'+m+'.npz'))['grid'])
            return set(map(tuple,np.floor((cells+.5)@T[:3,:3].T+T[:3,3]).astype(int)))
        for m in scored['summary']:
            cells=world(m)
            for p in building['probes']:
                xyz=tuple(np.floor(p['point']).astype(int))
                probes.append(dict(id=name,method=m,probe=p['id'],expected=p['occupied'],predicted=xyz in cells))
        groups.setdefault(case['building'],[]).append((world('cone'),truth,poses[0],size))
        if name.endswith('left'):
            grid=np.load(results/(name+'_'+selected+'.npz'))['grid']
            filename=results/(name+'-review.png')
            review(case,scene,poses,grid,filename)
            panels.append((building['kind'],Image.open(filename).copy()))
    for name,((a,truth,p1,size),(b,_,p2,_)) in groups.items():
        both=a&b
        occupancy=len(both&truth)/len(both|truth)
        silhouettes=[]
        for pose in (p1,p2):
            P=camera(pose,size)
            silhouettes.append(iou(truth_view(np.array(list(both)),P,size),truth_view(np.array(list(truth)),P,size)) if both else 0.)
        control.append(dict(building=name,voxel=occupancy,two_source_views=silhouettes))
    w,h=panels[0][1].size
    board=Image.new('RGB',(w*2,(h+28)*2),'#19232b')
    draw=ImageDraw.Draw(board)
    for i,(kind,im) in enumerate(panels):
        x,y=(i%2)*w,(i//2)*(h+28)
        draw.text((x+6,y+7),kind.capitalize(),fill='white')
        board.paste(im,(x,y+28))
    board.save(results/'four-buildings.png')
    (results/'diagnostics.json').write_text(json.dumps(dict(probes=probes,posed_two_view_control=control),indent=2))
    print('Two-view, known camera control:',np.mean([c['voxel'] for c in control]))
    print('Selected method probes:')
    for p in probes:
        if p['method']==selected:
            print(p['id'],p['probe'],p['expected']==p['predicted'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path)
    p.add_argument('results',type=Path)
    p.add_argument('selected')
    a=p.parse_args()
    diagnostics(a.inputs,a.results,a.selected)
