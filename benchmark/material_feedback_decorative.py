"""Verify the frozen material-removal feedback on the eight separate decorative source images."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark import material_feedback_geometry as exp
from benchmark import material_seam_confirmation as decorative
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import STABLE, checked, digest, read
from benchmark import typed_structures as typed
import numpy as np
from PIL import Image


def check(task):
    data,builds,key,expected,rule=task
    frame=data/'data'/key.split('-')[0]/'frames'/key
    build=builds/key
    camera=read(build/'build-manifest.json')['placement']
    source=checked(frame/'rgb.png',expected['rgb_sha256'])
    raw=checked(build/'reconstruction.zip',expected['pack_sha256'])
    cells=sorted(decoded_pack(raw))
    atlas=exp.atlas_maps()[0]
    with Image.open(source) as image:
        ids,evidence=exp.placed_materials.assign(image,camera,cells,atlas)
        if decorative.pack_hash(ids)!=expected['baseline_typed_sha256']:
            raise ValueError(f'Updated baseline differs for {key}')
        names={p:('minecraft:'+v,evidence[p]) for p,v in ids.items()}
        proposals=exp.propose(image,camera,cells,names)
        candidate=exp.select(proposals,rule)
        kept=[c for c in cells if candidate is None or list(c)!=candidate['cell']]
        selected,assigned=exp.placed_materials.assign(image,camera,kept,atlas) if candidate else (ids,evidence)
    _,decoded=typed.typed_pack(selected)
    truth,optional=decorative.truth_and_optional(frame)
    truth={p:'minecraft:'+v for p,v in truth.items()}
    coords={}
    if cells:
        transform=registration(np.array(camera['P']),read(frame/'camera.json'))
        xyz=np.floor((np.array(cells)+.5) @ transform[:3,:3].T + transform[:3,3]).astype(int)
        coords=dict(zip(cells,map(tuple,xyz.tolist()),strict=True))
    def score(mapping,origins):
        shift,_=exp.best_translation({coords[p] for p in mapping},set(truth)) if mapping else ((0,0,0),0)
        world={p:tuple(coords[p][i]+shift[i] for i in range(3)) for p in mapping}
        registered={world[p]:name for p,name in mapping.items()}
        valid={world[p] for p in mapping if origins[p]!='fallback'}
        return dict(geometry=typed.geometry_score(registered,truth,optional),
                    result=typed.score(registered,truth,optional,valid),translation=shift)
    return dict(id=key,source_hash=expected['rgb_sha256'],geometry_pack_sha256=expected['pack_sha256'],
                proposals=len(proposals),candidate=candidate,pack_sha256=decorative.pack_hash(selected),
                baseline=score({p:'minecraft:'+v for p,v in ids.items()},evidence),after=score(decoded,assigned))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=STABLE/'runs/decorative-builds-v1')
    parser.add_argument('--builds',type=Path,default=exp.HERE.parent/'material-seam-confirmation/runs/decorative')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    rows=read(exp.HERE/'benchmark/reviews/material-seam-confirmation-01/decorative.json')
    rule=read(exp.OUT/'train.json')['rule']
    tasks=[(args.data,args.builds,r['id'],r,rule) for r in rows]
    with ProcessPoolExecutor(args.workers) as pool:
        measured=list(pool.map(check,tasks))
    typed.compact(exp.OUT/'decorative.json',measured)
