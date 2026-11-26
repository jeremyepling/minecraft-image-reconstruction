"""Independently compare decoded packs and horizontal extents on the frozen transfer fixture."""
import argparse
from pathlib import Path

import numpy as np

from benchmark.blind_shape_archive import archive
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.visible_depth_transfer import structure


def report(inputs,runs,out):
    out.mkdir(parents=True,exist_ok=True)
    names=('baseline','final')
    archive(inputs,runs,names,out,prefix='')
    frames=read(inputs/'screen.json')['selected']
    results={name:{r['id']:r for r in read(runs/name/'cases.json')} for name in names}
    rows=[]
    for f in frames:
        key=f['id']
        base=inputs/f['directory']/'captures'
        pose=read(checked(base/(key+'.json'),f['files']['camera.json']))
        occupied=structure(read(checked(base/(key+'-blocks.json'),f['files']['world-readback.json'])),64)
        labels=read(checked(inputs/'truth'/key/'labels.json',f['files']['labels.json']))
        truth={tuple(b['xyz_world']):b['block_id'] for b in labels['blocks']}
        transform=registration(np.asarray(f['camera']['P']),pose)
        xmin,xmax=min(p[0] for p in occupied),max(p[0] for p in occupied)
        zmin,zmax=min(p[2] for p in occupied),max(p[2] for p in occupied)
        measured={}
        for name in names:
            row=results[name][key]
            path=runs/name/key
            pack=checked(path/'reconstruction.zip',row['pack_sha256'])
            xyz=registered(decoded_pack(pack),transform)
            assert xyz==set(map(tuple,read(path/'cells.json')['world']))
            if name=='baseline':
                before=xyz
            score=placement_scores(xyz,before,truth,set(occupied))
            m=score['relative_geometry']
            assert m==row['relative_geometry'] and list(score['translation'])==row['translation']
            dx,_,dz=score['translation']
            extent=[int(max(p[i] for p in xyz)-min(p[i] for p in xyz)+1) for i in (0,2)]
            measured[name]=dict(**m,f1=2*m['correct']/(m['placed']+m['target']),
                                outside_source_xz=sum(not(xmin<=x+dx<=xmax and zmin<=z+dz<=zmax) for x,y,z in xyz),
                                span_xz=extent)
        rows.append(dict(id=key,visible_span_xz=[max(p[i] for p in truth)-min(p[i] for p in truth)+1 for i in (0,2)],
                         **measured))
    save(out/'extents.json',rows)
    checks=[]
    for name,key in [('baseline','holdout-platformshelter-copper-low-left'),('final','holdout-offsetgate-quartz-high-right')]:
        path=runs/('full-'+name+'-check')
        manifest=read(path/'build-manifest.json')
        row=results[name][key]
        assert manifest['image_sha256']==row['rgb_sha256'] and manifest['through']=='placement'
        pack=path/'reconstruction.zip'
        replay=checked(runs/name/key/'reconstruction.zip',row['pack_sha256'])
        assert decoded_pack(pack)==decoded_pack(replay)
        checks.append(dict(method=name,id=key,rgb_sha256=row['rgb_sha256'],atlas_sha256=manifest['atlas']['atlas_sha256'],
                           blocks=len(decoded_pack(pack)),identical_positions_and_materials=True))
    save(out/'cli-checks.json',checks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','runs','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    report(a.inputs,a.runs,a.out)
