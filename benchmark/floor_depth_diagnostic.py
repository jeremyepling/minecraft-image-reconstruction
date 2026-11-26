"""Score which grid plane a masked surface actually occupies. Labels enter only here."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.floor_depth_fields import field
from benchmark.texture_camera_eval import DATA,STABLE
from engine.ground_depth import plane_levels
from engine.surface_model import intrinsic
from engine.visible_placement import _estimate_plane_cells
from engine.visible_raster import observed_support, SAMPLE_STRIDE


def frame(cache,split,mode):
    row=json.loads(cache.read_bytes())
    folder=DATA/'data'/split/'frames'/cache.stem
    if not row['parts'] or row['camera']['P'] is None: return []
    with Image.open(folder/'rgb.png') as src: image=src.convert('RGB')
    P=np.asarray(row['camera']['P']);P=P/np.linalg.norm(P[2,:3])
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    shape=tuple((n+SAMPLE_STRIDE-1)//SAMPLE_STRIDE for n in image.size[::-1])
    parts=row['parts']
    masks=[observed_support(p['support'],shape,image.size) for p in parts]
    candidates=_estimate_plane_cells(image,parts,masks,P,eye)
    levels=plane_levels(image,parts,masks,P,eye,field(image,mode),inverse=mode=='inverse')
    # All objects above are image-only. Below this point measure against pixel ownership.
    focal=np.linalg.norm(P[0,:3]-.5*image.width*P[2,:3])
    R=np.linalg.solve(intrinsic(image.size,focal),P[:,:3])
    loc=np.linalg.inv(registration(P,json.loads((folder/'camera.json').read_bytes())))
    fs=json.loads((folder/'labels.json').read_bytes())['faces']
    truths={}
    for f in fs:
        n=loc[:3,:3]@f['normal_world'];axis=np.argmax(abs(n))
        corner=loc@[*f['corners_world'][0],1]; truths[f['id']]=(axis,round(corner[axis]))
    with np.load(folder/'labels.npz') as data:
        ids=data['face_id'][SAMPLE_STRIDE//2::SAMPLE_STRIDE,SAMPLE_STRIDE//2::SAMPLE_STRIDE]
    out=[]
    for part,mask,cells in zip(parts,masks,candidates):
        if not cells or part.get('normal_camera') is None: continue
        axis=int(np.argmax(abs(R.T@part['normal_camera'])))
        base=next(iter(cells))[axis];base+=int(eye[axis]>base+1)
        count=Counter()
        face_ids,pixels=np.unique(ids[mask],return_counts=True)
        for face_id,pix in zip(face_ids,pixels):
            if face_id and truths[face_id][0]==axis:
                count[truths[face_id][1]]+=int(pix)
        if not count: continue
        truth,support=count.most_common(1)[0]
        if support/mask.sum()<.5:continue
        out.append(dict(id=part['id'],frame=cache.stem,family=row['family'],axis=axis,
                        base=int(base),calibrated=levels.get(part['id'],int(base)),truth=int(truth),
                        total=int(mask.sum()),support=support))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='train');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int);p.add_argument('--mode',default='inverse')
    a=p.parse_args()
    files=sorted((STABLE/'runs/visible-depth-rounds/inputs').glob(a.split+'-*.json'))[:a.limit]
    rows=[r for file in files for r in frame(file,a.split,a.mode)]
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(rows))
    for key in ('base','calibrated'):
        wrong=[abs(r[key]-r['truth']) for r in rows]
        print(key,len(rows),sum(v==0 for v in wrong),sum(v<=1 for v in wrong),sum(wrong))


if __name__=='__main__': main()
