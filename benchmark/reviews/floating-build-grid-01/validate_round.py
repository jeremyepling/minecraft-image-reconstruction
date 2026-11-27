
"""Independently validate paired inputs and build-coordinate invariance."""
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.expanded_baseline import digest, read, save
from benchmark.mask_blocks import decoded_pack


from benchmark.expanded_baseline import STABLE

RUN = STABLE/'runs/floating-build-grid-01'
EXACT = STABLE.with_name(STABLE.name+'-worktrees')/'floating-build-floor-bypass/runs/floating-build-grid-01-verified'


def world_inventory(folder, origin):
    return {tuple(i+a for i,a in zip(cell,origin,strict=True)): block
            for cell,block in decoded_pack(folder/'reconstruction.zip').items()}


def rows():
    fixtures=read(RUN/'fixtures.json')
    diagnostics=[]
    composition=[]
    for e in fixtures['cases']:
        frame=Path(e['frame'])
        source=np.asarray(Image.open(frame/'rgb.png').convert('RGB'))
        with np.load(frame/'labels.npz',allow_pickle=False) as native:
            mask=native['face_id'] > 0
        labels=read(frame/'labels.json')
        underside=sum(face['normal_world']==[0,-1,0] for face in labels['faces'])
        for bg,variant in e['variants'].items():
            p=Path(variant['image'])
            assert digest(p)==variant['source_sha256'], f'Image changed: {p}'
            current=np.asarray(Image.open(p).convert('RGB'))
            different=np.any(source!=current,axis=2)
            assert int(np.sum(mask & different))==0, f'Foreground changed: {p}'
            composition.append(dict(id=e['id'],background=bg,native_pixels=int(mask.sum()),
                                    background_pixels_changed=int(np.sum(~mask & different)),underside_faces=underside))
            for plane in ('floor','floating'):
                base=EXACT/f'exact-{plane}-base'/bg/e['id']
                roof=EXACT/f'exact-{plane}-roof'/bg/e['id']
                if not (base/'score.json').is_file() or not (roof/'score.json').is_file():
                    continue
                rb=read(base/'score.json')
                rr=read(roof/'score.json')
                assert rb['source_sha256']==rr['source_sha256']==variant['source_sha256']
                assert rb['handoff_sha256']==rr['handoff_sha256']
                basemap=world_inventory(base,rb['projection_checks']['origin_world'])
                roofmap=world_inventory(roof,rr['projection_checks']['origin_world'])
                diagnostics.append(dict(id=e['id'],background=bg,floor_policy=plane,
                     max_corner_px=max(rb['projection_checks']['maximum_native_error_px'],
                                      rr['projection_checks']['maximum_native_error_px']),
                     origin_delta_y=rr['projection_checks']['origin_world'][1]-rb['projection_checks']['origin_world'][1],
                     base_blocks=len(basemap),roof_blocks=len(roofmap),
                     identical_positions=basemap.keys()==roofmap.keys(), identical_ids=basemap==roofmap,
                     only_base=len(basemap.keys()-roofmap.keys()),only_roof=len(roofmap.keys()-basemap.keys()),
                     base_f1=rb['geometry']['strict']['f1'],roof_f1=rr['geometry']['strict']['f1']))
    return composition, diagnostics


if __name__=='__main__':
    composition,diagnostics=rows()
    save(RUN/'independent-checks.json',dict(composition=composition,origins=diagnostics))
    for op in ('floor','floating'):
        obs=[d for d in diagnostics if d['floor_policy']==op]
        print(op,len(obs),'pairs; identical geometry',sum(d['identical_positions'] for d in obs),'; identical IDs',sum(d['identical_ids'] for d in obs))
