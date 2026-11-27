"""Confirm the wall grid oracle on the four independent gate and shelter photographs."""
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked,read,save
from benchmark.wall_section_oracle import CONTROL,ATLAS
from engine import wall_section_grid as grid
from training.visible_labels import visible_labels


def run(inputs,control,out):
    frames=read(inputs/'screen.json')['selected']
    rows={r['id']:r for r in read(control/'cases.json')}
    cases=[]
    for f in frames:
        key=f['id']
        rgb=inputs/f['directory']/'captures'/(f['source_id']+'.png')
        row=rows[key]
        parts=read(checked(inputs/'upstream'/key/'surfaces.json',row['surface_sha256']))['planes']
        camera=f['camera']
        prior=set(decoded_pack(checked(control/key/'reconstruction.zip',row['pack_sha256'])))
        with Image.open(checked(rgb,f['files']['rgb.png'])) as im:
            image=im.convert('RGB')
        ctx,raw=grid.candidates(image,parts,camera,prior,ATLAS)
        groups=grid.disjoint(raw)
        pose=read(checked(rgb.with_suffix('.json'),f['files']['camera.json']))
        voxels=read(checked(rgb.with_name(f['source_id']+'-blocks.json'),f['files']['world-readback.json']))
        states={tuple(b[ax] for ax in 'xyz'):b for b in voxels}
        labels,_,_=visible_labels(pose,states,(0,0,0))
        truth={tuple(b['xyz_world']) for b in labels['blocks']}
        T=registration(np.asarray(camera['P']),pose)
        def correct(cell):
            xyz=np.floor((np.asarray(cell)+.5)@T[:3,:3].T+T[:3,3]).astype(int)+row['translation']
            return tuple(xyz) in truth
        covered={c for g in groups for o in g for c in o['cells']}
        chosen=[]
        for group in groups:
            good=[o for o in group if all(correct(c) for c in o['cells'])]
            if good:
                chosen.append(max(good,key=lambda o:len(o['cells'])))
        after=grid.materialize(ctx,prior,chosen) if chosen else prior
        assert all(correct(c) for c in after-prior)
        cases.append(dict(id=key,candidates=len(covered),positive=sum(map(correct,covered)),actual=len(after-prior),
                          section_options=sum(map(len,groups)),control_pack_sha256=row['pack_sha256']))
    save(out,cases)


if __name__=='__main__':
    run(CONTROL.parents[2]/'framed-depth-transfer/runs/transfer-input',CONTROL.parent/'transfer-production',
        Path('benchmark/reviews/wall-section-grid-01/shelter-oracle.json'))
