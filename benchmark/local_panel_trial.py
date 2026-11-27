"""Export local wall-panel predictions before scoring the image-only and reference choices."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from shutil import copyfile

import numpy as np

from benchmark import framed_depth_eval as framed, local_panel_depth as panels, wall_section_eval as frozen
from benchmark.mask_blocks import _write_read, decoded_pack
from benchmark.texture_camera_eval import checked, read, save, digest

CONTROL=Path('../wall-section-round/runs/control')


def extract(job):
    split,f,inputs,row,out=job
    image,parts,counts,camera=framed.photo(split,f,inputs)
    src=checked(CONTROL/split/f['id']/'reconstruction.zip',row['pack_sha256'])
    prior=set(decoded_pack(src))
    scene,options=panels.trials(image,parts,camera,prior)
    selected=panels.choose(options)
    decided=panels.materialize(image,scene,prior,selected)
    where=out/split/f['id']
    where.mkdir(parents=True,exist_ok=True)
    pack=where/'reconstruction.zip'
    if decided==prior:
        copyfile(src,pack)
    else:
        assert _write_read(pack,decided)==decided
    # Reference labels enter only after saving the model output.
    ref_split='old' if split in ('extra','new') else split
    builds=framed.NEW_INPUTS if split=='new' else framed.transfer.BUILDS
    truth,occ,T,_=framed.references(ref_split,f,framed.NATIVE_INPUTS.get(split),framed.TRAINING_DATA,
                                    framed.DEPTH,dict(native=framed.native.BUILDS,old=builds,fresh=builds))
    def world(c):
        return tuple(np.floor((np.asarray(c)+.5)@T[:3,:3].T+T[:3,3]).astype(int)+row['translation'])
    def hits(cells):
        return sum(world(c) in truth for c in cells)
    for o in options:
        o['true_add']=hits(o['after'])
        o['true_drop']=hits(o['before'])
    clean=[o for o in options if o['true_add']==len(o['after']) and o['true_drop']==0]
    oracle=0
    for o in sorted(clean,key=lambda x:-len(x['after'])):
        if panels.materialize(image,scene,prior,o)!=prior:
            oracle=len(o['after'])
            break
    chosen=options.index(selected) if selected is not None and decided!=prior else None
    save(where/'options.json',dict(options=options,selected=chosen,oracle=oracle))
    return dict(split=split,id=f['id'],rgb_sha256=row['rgb_sha256'],input_hashes=row['input_hashes'],
                reference_hashes=f['files'],control_pack_sha256=row['pack_sha256'],pack_sha256=digest(pack),
                options=len(options),oracle=oracle,eligible=sum(o['texture_gain'] is not None for o in options),
                chosen=chosen,selected_size=len(selected['after']) if chosen is not None else 0,
                gained=hits(decided-prior),lost=hits(prior-decided),
                hidden_added=sum(world(c) in occ-truth for c in decided-prior))


def run(args):
    meta=read(CONTROL/'summary.json')
    rows={(r['split'],r['id']):r for r in read(checked(CONTROL/'cases.json',meta['cases_sha256']))}
    jobs=[]
    for s,f,prior in frozen.cases(args.splits):
        r=rows[s,f['id']]
        assert prior['pack_sha256']==r['pack_sha256'] and prior['rgb_sha256']==r['rgb_sha256']
        jobs.append((s,f,prior,r,args.out))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records=list(pool.map(extract,jobs))
    save(args.out/'cases.json',records)
    save(args.out/'summary.json',dict(cases_sha256=digest(args.out/'cases.json'),
         control_cases_sha256=meta['cases_sha256'],candidate_sha256=digest(Path(panels.__file__))))
    print('cases, options, clean oracle blocks, image gains/losses, hidden',
          len(records),*[sum(r[k] for r in records) for k in ('options','oracle','gained','lost','hidden_added')])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',nargs='+',default=list(framed.SIZES),choices=framed.SIZES)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workers',default=4,type=int)
    run(p.parse_args())
