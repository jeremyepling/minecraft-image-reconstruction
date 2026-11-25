"""Diagnostic ordinal-depth check across labeled occlusion boundaries in one probe frame."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import digest, read, save
from benchmark.surface_metrics import DOWNSAMPLE, truth


def occlusion_pairs(t, inverse_depth):
    ids = t['id_map']
    h,w = ids.shape
    field = np.array(Image.fromarray(inverse_depth.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))
    differences = defaultdict(list)
    for axis in (0,1):
        a,b = (ids[:-1,:],ids[1:,:]) if axis==0 else (ids[:,:-1],ids[:,1:])
        yy,xx = np.nonzero((a!=b)&(a>0)&(b>0))
        dy,dx = (1,0) if axis==0 else (0,1)
        for y,x in zip(yy,xx):
            # Sample 10 native pixels inside each side of this quarter-resolution edge.
            y0,x0,y1,x1 = y-2*dy,x-2*dx,y+3*dy,x+3*dx
            if min(y0,x0)<0 or y1>=h or x1>=w:
                continue
            left,right = int(a[y,x]), int(b[y,x])
            if ids[y0,x0]!=left or ids[y1,x1]!=right:
                continue
            ray = [DOWNSAMPLE*(x+.5+.5*dx),DOWNSAMPLE*(y+.5+.5*dy),1.]
            z = [1/np.linalg.solve(t['charts'][p],ray)[2] for p in (left,right)]
            if min(z)<=0 or abs(z[0]-z[1])<.75:
                continue
            near_first = z[0] < z[1]
            pair = (left,right) if near_first else (right,left)
            diff = float(field[y0,x0]-field[y1,x1])
            differences[pair].append(diff if near_first else -diff)
    pairs = [dict(near_patch=a,far_patch=b,samples=len(values),
                  median_near_minus_far=float(np.median(values)),correct=bool(np.median(values)>0))
             for (a,b),values in sorted(differences.items()) if len(values)>=4]
    return dict(eligible_pairs=len(pairs),correct_order=sum(row['correct'] for row in pairs),pairs=pairs,
                min_true_depth_step_blocks=.75,min_boundary_samples=4,sample_offset_native_px=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture',required=True,type=Path)
    ap.add_argument('--probe',required=True,type=Path,help='Output directory of benchmark.depth_probe')
    ap.add_argument('--out',required=True,type=Path)
    args = ap.parse_args()
    report = read(args.probe/'probe.json')
    f, = [f for f in read(args.fixture/'index.json')['frames'] if f['image_sha256']==report['source_sha256']]
    if digest(args.fixture/f['image'])!=report['source_sha256']:
        raise ValueError('Probe image differs from the fixture')
    t = truth(read(args.fixture/f['pose']),read(args.fixture/f['labels'])['faces'])
    scores = occlusion_pairs(t,np.load(args.probe/'relative-depth.npy',allow_pickle=False))
    save(args.out,dict(frame=f['id'],image_sha256=f['image_sha256'],model_sha256=report['model_sha256'],**scores))


if __name__=='__main__':
    main()
