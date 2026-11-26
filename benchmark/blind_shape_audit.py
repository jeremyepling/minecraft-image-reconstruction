"""Compare frozen geometry fingerprints to the existing development and test archives."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from benchmark import visible_depth_transfer as transfer
from benchmark.texture_camera_eval import DATA, checked, frames, read, save


def audit(root,out):
    shape_keys, rgb_hashes = set(),set()
    for split in ('train','validation','test','regression'):
        for f in frames(DATA,split):
            rows=read(checked(DATA/'data'/split/f['directory']/'world-readback.json',f['files']['world-readback.json']))
            shape_keys.add(transfer.shape_key(transfer.structure(rows,f['ground_y'])))
            rgb_hashes.add(f['files']['rgb.png'])
    with ZipFile(transfer.ARCHIVE) as z:
        for f in json.loads(z.read('native-inputs/index.json')):
            path = 'native-inputs/'+f['fixture_directory']+'/captures/'+f['id']+'-blocks.json'
            shape_keys.add(transfer.shape_key(transfer.structure(json.loads(z.read(path)),64)))
            rgb_hashes.add(f['sha256'])
    for f in read(transfer.OUT/'screen.json')['screened']:
        shape_keys.add(f['shape'])
        rgb_hashes.add(f['files']['rgb.png'])
    new=read(root/'screen.json')['selected']
    shapes={f['shape'] for f in new}
    images={f['files']['rgb.png'] for f in new}
    assert len(shapes)==2 and len(images)==4 and not shape_keys&shapes and not rgb_hashes&images
    save(out,dict(previous_shape_count=len(shape_keys),previous_unique_rgb=len(rgb_hashes),
                  new_shape_count=len(shapes),new_unique_rgb=len(images),overlapping_shapes=0,overlapping_rgb=0,
                  new_shapes=sorted(shapes)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    audit(args.root,args.out)
