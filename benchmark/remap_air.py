"""Compare occupancy support thresholds on a frozen common lattice."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from benchmark.geometry_benchmark import fixture,load
from benchmark.geometry_metrics import score
from engine.geometry import UNIT,project
from engine.remap import structure_mask


def run(roots,split):
    trials=(.50,.60,.70,.80)
    metrics={str(t):[] for t in trials}
    uv=np.stack(np.meshgrid((np.arange(7)+.5)/7,(np.arange(7)+.5)/7),axis=-1)
    for root in roots:
        buildings,voxels=fixture(root)
        for v in load(root/'views.json'):
            if v['split']!=split or v['kind']=='adjacency':
                continue
            name=v['id'];building=buildings[v['building_id']]
            raw=load(root/'geometry-predictions/ground'/(name+'.json'))
            pose=load(root/'captures'/(name+'.json'))
            with Image.open(root/'captures'/(name+'.png')) as im:
                mask=structure_mask(im)
            vals=[]
            for c in raw['cells']:
                q=np.array(c['ij'])
                xy=np.floor(project(np.array(raw['planes'][0]['grid_to_image']),q+uv)).astype(int)
                x,y=xy[:,:,0],xy[:,:,1]
                ok=(x>=0)&(y>=0)&(x<mask.shape[1])&(y<mask.shape[0])
                vals.append(np.mean(ok & mask[y.clip(0,mask.shape[0]-1),x.clip(0,mask.shape[1]-1)]))
            for t in trials:
                cells=[c for c,s in zip(raw['cells'],vals) if s>=t]
                m=score(building,pose,voxels[building['page']],raw['planes'],cells)
                posts=[p for p in m['post_probes'] if not p['occluded']]
                metrics[str(t)].append([m['silhouette_iou'],m['air_false_positive'],m['air_evaluable'],
                                        sum(bool(p['filled']) for p in posts),len(posts)])
    return {k:dict(iou=round(float(np.mean(a,axis=0)[0]),4),false_air=int(a[:,1].sum()),
                   air=int(a[:,2].sum()),posts_kept=int(a[:,3].sum()),posts=int(a[:,4].sum()))
            for k,rows in metrics.items() for a in [np.array(rows)]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',nargs='+',type=Path)
    p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    result=run(args.roots,args.split);print(result)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
