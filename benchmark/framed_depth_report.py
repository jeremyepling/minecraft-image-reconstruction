"""Archive checked exports and measure uncertainty across complete building shapes."""
import argparse
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from shutil import copyfile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from benchmark.framed_depth_eval import (DEPTH, EVALUATION, NATIVE_INPUTS,
    SIZES, TRAINING_DATA, cases, native, references, transfer)
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked, digest, read, save


def f1(rows):
    return 2*rows[...,0]/(rows[...,1]+rows[...,2])


def bootstrap(totals):
    # Resample whole occupancy shapes, keeping all frames and palettes of a shape together.
    rows=np.asarray(list(totals.values()))
    rng=np.random.default_rng(61126)
    sample=rows[rng.integers(len(rows),size=(10000,len(rows)))].sum(axis=1)
    differences=100*(f1(sample[:,1])-f1(sample[:,0]))
    individual=f1(rows[:,1])-f1(rows[:,0])
    return dict(shapes=len(rows),improved=int((individual>0).sum()),tied=int((individual==0).sum()),
                regressed=int((individual<0).sum()),paired_f1_delta_percentile_95_pp=np.quantile(differences,[.025,.975]).tolist())


def report(source,out):
    out.mkdir(exist_ok=True,parents=True)
    summary=read(source/'summary.json')
    scored=read(checked(source/'cases.json',summary['cases_sha256']))
    index={(r['split'],r['id']):r for r in scored}
    shapes={name:defaultdict(lambda:np.zeros((2,3),dtype=int)) for name in ('fixed','extra')}
    assert len(index)==len(scored)==sum(SIZES.values())
    with ZipFile(out/'scores.zip','w',ZIP_DEFLATED) as archive:
        for split,f,control in cases(SIZES):
            key=f['id']
            row=index[split,key]
            folder=source/split/key
            pack=checked(folder/'reconstruction.zip',row['pack_sha256'])
            assert set(decoded_pack(BytesIO(pack.read_bytes())))==set(map(tuple,read(folder/'cells.json')['local']))
            for path in (pack,folder/'cells.json'):
                archive.write(path,path.relative_to(source))
            cohort='fixed' if split in EVALUATION else split
            if cohort not in shapes:
                continue
            ref_split='old' if split=='extra' else split
            _,occupied,_,_=references(ref_split,f,NATIVE_INPUTS.get(split),TRAINING_DATA,DEPTH,
                                      dict(native=native.BUILDS,old=transfer.BUILDS,fresh=transfer.BUILDS))
            shape=transfer.shape_key(occupied)
            for i,model in enumerate((control,row)):
                m=model['relative_geometry']
                shapes[cohort][shape][i]+=[m['correct'],m['target'],m['placed']]
        for name in ('cases.json','summary.json'):
            archive.write(source/name,name)
            copyfile(source/name,out/name)
    with ZipFile(out/'scores.zip') as archive:
        assert archive.testzip() is None
    save(out/'shape-check.json',dict(cases_sha256=summary['cases_sha256'],archive_sha256=digest(out/'scores.zip'),
                                     groups={k:bootstrap(v) for k,v in shapes.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report(args.predictions,args.out)
