"""Measure the added placement cost on three fixed photos, with identical warm model caches."""
import argparse
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

from benchmark.framed_depth_eval import cases, members, photo
from benchmark.framed_depth_sheet import VIEWS
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import STABLE, save
from engine import atlas_plane_depth


def measure(predictions, out):
    ids=set(VIEWS)|{'holdout-switchback-brick-high-right'}
    result=[]
    for split,f,row in cases(('old','extra','new')):
        key=f['id']
        if key not in ids:
            continue
        image,parts,counts,camera=photo(split,f,row)
        archive,folder=members(split)
        expected={'previous': set(decoded_pack(BytesIO(archive.read(f'{folder}/{key}/reconstruction.zip')))),
                  'final': set(decoded_pack(predictions/split/key/'reconstruction.zip'))}
        timings={'previous':[],'final':[]}
        for n in range(3):
            for name in (('previous','final') if n%2==0 else ('final','previous')):
                def keep(image,parts,camera,prior,atlas,atlas_path):
                    return prior,{}
                disable=patch.object(atlas_plane_depth,'refine_framed_depth',keep) if name=='previous' else nullcontext()
                with disable:
                    t=perf_counter()
                    cells,_=atlas_plane_depth.place_ranked(image,'26.3',parts,counts,camera,
                                                           atlas_path=STABLE/'runs/26.3/atlas')
                    elapsed=perf_counter()-t
                assert cells==expected[name],(name,key)
                if n:
                    timings[name].append(elapsed)
        result.append(dict(id=key,seconds=timings,medians={k:median(v) for k,v in timings.items()}))
    assert len(result)==3
    save(out,dict(stage='placement from frozen image-derived camera, masks and counts',rows=result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    measure(args.predictions,args.out)
