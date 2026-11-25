"""Cache pixel-only rim proposals separately from labels for exact-depth comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark.oblique_recess import infer
from benchmark.recess_review import known_openings
from engine.geometry import project


def cache(root,folder,split,controls):
    folder.mkdir(parents=True,exist_ok=True)
    scenes={s['id']:s for s in json.loads((root/'buildings.json').read_text())}
    for view in json.loads((root/'views.json').read_text()):
        if 'left' not in view['angle'] and 'right' not in view['angle']:
            continue
        if not controls and (view['split']!=split or view['kind'] not in ('heldout-facade','recess')):
            continue
        vid=view['id']
        payload=[]
        def collect(rect,q,vp,distance,pitch,spacing,H):
            payload.append((rect,q,vp,distance,pitch,spacing,H))
        prefix=root/'captures'/vid
        with Image.open(prefix.with_suffix('.png')) as im:
            result=infer(im,collect)
        known={} if controls else known_openings(scenes[view['building_id']],json.loads(prefix.with_suffix('.json').read_text()))
        labels=[]
        for i,(c,(rect,q,vp,distance,pitch,spacing,H)) in enumerate(zip(result['candidates'],payload)):
            if i==0:
                np.save(folder/(vid+'.npy'),rect)
            f=vid+'-'+str(i)
            np.savez(folder/(f+'.npz'),q=q,vp=vp,distance=distance,pitch=pitch,spacing=spacing,H=H)
            xy=np.mean(c['rim'],axis=0)
            target=next(((name,d) for name,(inv,d) in known.items()
                         if ((project(inv,xy)>=0)&(project(inv,xy)<=1)).all()),None)
            labels.append(dict(file=f,target=target,baseline=c['depth'],flat_veto=bool(c['corrected_repeat'] and c['corrected_repeat']['veto'])))
        (folder/(vid+'.json')).write_text(json.dumps(dict(openings={k:v[1] for k,v in known.items()},rows=labels),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','out'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--controls',action='store_true')
    p.add_argument('--split',default='test')
    a=p.parse_args();cache(a.root,a.out,a.split,a.controls)
