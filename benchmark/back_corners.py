"""Compare back-corner estimates with a shared flat-material check and frozen selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark.oblique_recess import internal_repeat,infer
from engine.geometry import project
from engine.corner_edges import predict as edge_depth
from engine.corner_texture import predict as texture_depth
from engine.rim_spacing import texture_detail

METHODS=('baseline','edges','texture','agreement','corroborated')


def alternatives(rect,q,vp,distance,pitch,spacing,baseline,field):
    edge=edge_depth(rect,q,vp,distance,pitch,spacing)
    texture=texture_depth(rect,q,vp,distance,pitch,spacing)
    def checked(proposed):
        if not proposed:
            return None
        observation=internal_repeat(field,q.min(axis=0),q.max(axis=0),vp,distance,proposed,pitch,spacing)
        return None if observation['veto'] else int(proposed)
    e,t=checked(edge['depth']),checked(texture['depth'])
    agreement=e if e==t else None
    votes=[d for d in (baseline,e,t) if d is not None]
    majority=Counter(votes).most_common(1)
    corroborated=majority[0][0] if majority and majority[0][1]>=2 else None
    methods=dict(baseline=baseline,edges=e,texture=t,agreement=agreement,corroborated=corroborated)
    return methods,edge,texture


def evaluate(folder):
    counts={name:Counter(exact=0,wrong_depth=0,false_recess=0) for name in METHODS}
    rows={}
    total=0
    for path in sorted(folder.glob('*.json')):
        truth=json.loads(path.read_text())
        total+=sum(d>0 for d in truth['openings'].values())
        rows[path.stem]=[]
        if not truth['rows']: continue
        rect=np.load(path.with_suffix('.npy'))
        field=texture_detail(rect)
        credited={k:set() for k in METHODS}
        for row in truth['rows']:
            with np.load(folder/(row['file']+'.npz')) as f:
                methods,edge,texture=alternatives(rect,f['q'],f['vp'],float(f['distance']),float(f['pitch']),
                                                  f['spacing'],row['baseline'],field)
            target=row['target']
            for name,d in methods.items():
                if d is None: continue
                if not target or target[1]==0 or target[0] in credited[name]:
                    counts[name]['false_recess']+=1
                else:
                    credited[name].add(target[0])
                    counts[name]['exact' if d==target[1] else 'wrong_depth']+=1
            rows[path.stem].append(dict(target=target,predictions=methods))
    for name,c in counts.items():
        c['unknown']=total-c['exact']-c['wrong_depth']
        c['utility']=c['exact']-c['wrong_depth']-2*c['false_recess']
    return dict(positive_observations=total,methods=counts,views=rows)


def inspect_image(image):
    data=[]
    result=infer(image,lambda *args:data.append(args))
    field=texture_detail(data[0][0]) if data else None
    for c,(rect,q,vp,distance,pitch,spacing,h) in zip(result['candidates'],data):
        methods,edge,texture=alternatives(rect,q,vp,distance,pitch,spacing,c['depth'],field)
        c.update(depth_by_method=methods,texture_evidence=texture,edge_evidence=edge)
        if edge.get('back_corner') is not None:
            c['edge_corner_image_px']=project(h,np.array(edge['back_corner'])/pitch).tolist()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--cache',type=Path)
    source.add_argument('--image',type=Path,help='Inspect a screenshot with all candidate methods')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--selection',type=Path,help='Frozen development report when scoring other views')
    a=p.parse_args()
    if a.image:
        with Image.open(a.image) as im:
            result=inspect_image(im)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
        raise SystemExit(0)
    result=evaluate(a.cache)
    result['selected']=(json.loads(a.selection.read_text())['selected'] if a.selection else
                        max(METHODS,key=lambda n:result['methods'][n]['utility']))
    result['selection_rule']='Development exact minus wrong depth minus twice false recesses; ties prefer the earlier listed method'
    print(result['methods'],result['selected'])
    a.out.write_text(json.dumps(result,indent=2)+'\n')
