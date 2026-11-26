"""Count exact visible placements by authored part at the whole-build alignment."""
import argparse
from collections import Counter
from pathlib import Path

from benchmark.relative_geometry import best_translation
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.mask_blocks import decoded_pack


def measure(root,predictions,dest):
    layers=read(root/'native/layout.json')['blocks']
    cases={row['id']:row for row in read(predictions/'cases.json')}
    measured=[]
    for frame in read(root/'screen.json')['selected']:
        key=frame['id']
        case=cases[key]
        label=read(checked(root/'truth'/key/'labels.json',frame['files']['labels.json']))
        truth={tuple(b['xyz_world']) for b in label['blocks']}
        roles={tuple(row[k] for k in 'xyz'):row['role'] for row in layers if row['block_id']!='air'}
        folder=predictions/key
        pack=checked(folder/'reconstruction.zip',case['pack_sha256'])
        cells=read(folder/'cells.json')
        assert set(decoded_pack(pack))==set(map(tuple,cells['local']))
        world=set(map(tuple,cells['world']))
        shift,correct=best_translation(world,truth)
        aligned={tuple(v+d for v,d in zip(c,shift)) for c in world}
        covered=Counter(roles[c] for c in truth&aligned)
        targets=Counter(roles[c] for c in truth)
        assert len(truth)==case['relative_geometry']['target'] and correct==case['relative_geometry']['correct']
        measured.append(dict(id=key,translation=shift,visible_positions=case['relative_geometry'],
                             target_by_part=dict(targets),correct_by_part=dict(covered),
                             pack_sha256=digest(pack)))
    save(dest,measured)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    measure(a.root,a.predictions,a.out)
