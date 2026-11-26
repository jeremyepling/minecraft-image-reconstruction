"""Prepare and validate the two sealed structures; labels are used only for scoring."""
import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmark import blind_shape_audit, blind_shape_inputs
from benchmark.camera_metrics import score as camera_score
from benchmark.depth_transfer_fixture import ANGLES, PALETTES
from benchmark.texture_camera_eval import ROOT, STABLE, checked, read, save
from training.framing import measure_framing


REVISION = 'c0fc6a8a277433e84365a707bb87b25a7a1ff615'


def prepare(inputs,out,atlas):
    blind_shape_inputs.prepare(inputs,PALETTES,ANGLES,REVISION,atlas)
    previous=read(ROOT/'benchmark/reviews/blind-shapes-01/screen.json')['selected']
    known_shapes={r['shape'] for r in previous}
    known_images={r['files']['rgb.png'] for r in previous}
    out.mkdir(parents=True,exist_ok=True)
    blind_shape_audit.audit(inputs,out/'novelty.json')
    framing=[]
    selected=read(inputs/'screen.json')['selected']
    for f in selected:
        key=f['id']
        assert f['shape'] not in known_shapes and f['files']['rgb.png'] not in known_images
        base=inputs/f['directory']/'captures'
        pose=read(checked(base/(key+'.json'),f['files']['camera.json']))
        raw=read(checked(base/(key+'-blocks.json'),f['files']['world-readback.json']))
        with np.load(checked(inputs/'truth'/key/'labels.npz',f['files']['labels.npz'])) as data:
            margins=measure_framing(pose,raw,f['ground_y'],data['face_id'])
        assert margins['accepted'],(key,margins)
        faces=read(inputs/'native/labels'/(key+'.json'))['faces']
        # This is a camera-only diagnostic; surface proposals are frozen separately.
        result=camera_score(dict(f['camera'],planes=[]),pose,faces)
        framing.append(dict(id=key,files=f['files'],shape=f['shape'],framing=margins,camera=result))
    novelty=read(out/'novelty.json')
    novelty.update(previously_reserved_shapes_checked=len(known_shapes),previously_reserved_rgb_checked=len(known_images))
    save(out/'novelty.json',novelty)
    save(out/'framing.json',framing)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--atlas',type=Path,default=STABLE/'runs/26.3/atlas')
    p.add_argument('--archive',type=Path)
    args=p.parse_args()
    if args.archive:
        blind_shape_inputs.package(args.root,args.archive)
        with ZipFile(args.archive) as archive:
            assert len(archive.namelist())==53 and archive.testzip() is None
            assert all(archive.read(name)==(args.root/name).read_bytes() for name in archive.namelist())
    else:
        prepare(args.root,args.out,args.atlas)
