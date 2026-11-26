"""Measure which surfaces of each held-out canopy roof actually supplied source pixels."""
import argparse
from pathlib import Path

from benchmark import visible_depth_transfer as transfer
from benchmark.texture_camera_eval import checked, read, save
from training.visible_labels import visible_labels


def count(f):
    rgb=transfer.rgb_path(f)
    pose=read(checked(rgb.with_suffix('.json'),f['files']['camera.json']))
    blocks=read(checked(rgb.with_name(f['source_id']+'-blocks.json'),f['files']['world-readback.json']))
    occupied=transfer.structure(blocks,64)
    roof=max(y for x,y,z in occupied)
    labels,_,_=visible_labels(pose,{tuple(b[k] for k in 'xyz'):b for b in blocks},(0,0,0))
    faces=[r for r in labels['faces'] if r['xyz_world'][1]==roof]
    return dict(id=f['id'],shape=f['shape'],course=f['target_block_y'],eye_y_above_base=pose['eye_world'][1]-65,
                visible_top_pixels=sum(r['visible_pixels'] for r in faces if r['normal_world'][1]==1),
                visible_underside_pixels=sum(r['visible_pixels'] for r in faces if r['normal_world'][1]==-1),
                visible_vertical_rim_pixels=sum(r['visible_pixels'] for r in faces if r['normal_world'][1]==0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    save(a.out,{src.stem:[count(f) for f in read(src)['selected']] for src in a.selection})
