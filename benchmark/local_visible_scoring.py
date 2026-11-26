"""Score decoded provisional and baseline placements under each recorded view."""
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

from benchmark.camera_metrics import truth_camera
from benchmark.local_visible_placement import render
from benchmark.camera_round import read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','out','baseline'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    candidate=read(args.out/'coords.json')
    original=read(args.baseline/'coords.json')
    truth=read(args.baseline/'targets.json')
    rows=[]
    for key, coords in candidate.items():
        folder=args.root/'runs/visible-builds-v1/data/regression/frames'/('legacy-'+key.replace('/','-'))
        expected=set(map(tuple,truth[key]))
        actual=set(map(tuple,coords))
        prior=set(map(tuple,original[key]))
        P=truth_camera(read(folder/'camera.json'))
        visible=np.load(folder/'labels.npz')['face_id'][2::4,2::4]>0
        row=dict(key=key,gained=len((actual-prior)&expected),lost=len((prior-actual)&expected),
                 extra_added=len(actual-prior-expected),extra_removed=len(prior-actual-expected))
        for name, method in [('baseline',prior),('registered',actual)]:
            mask=render(method,P,visible.shape)>=0
            dest=args.out/(name+'-masks')/(key+'.png'); dest.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(mask).save(dest)
            row.update({name+'_overlap':int((mask&visible).sum()),name+'_union':int((mask|visible).sum()),
                        name+'_pixels':int(mask.sum())})
        rows.append(row)
    total=Counter()
    for row in rows:
        total.update({k:v for k,v in row.items() if k!='key'})
    (args.out/'registered-score.json').write_text(json.dumps(dict(total=total,cases=rows),indent=2))
    print(total)


if __name__=='__main__':
    main()
