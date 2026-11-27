"""Verify the shipped mask and local-placement helpers together on all frozen RGB inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from benchmark import facade_inset_round as suite
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked, digest, read, save
from engine import inset_returns
from engine.visible_inset_cells import inset_edits


LOCAL = suite.ROOT/'benchmark/reviews/facade-local-placement-01'
REPORT = suite.ROOT/'benchmark/reviews/facade-inset-round-01'


@lru_cache(maxsize=1)
def local_release():
    meta=read(LOCAL/'summary.json')
    rows={r['id']:r for r in read(checked(LOCAL/'cases.json',meta['cases_sha256']))}
    return rows, ZipFile(checked(LOCAL/'exports.zip',meta['exports_sha256']))


def verify(job):
    split,frame,prior,control=job
    im,parts,_,camera=suite.ordinary_inputs(*job)
    revised,returns,phase,outlines,_=inset_returns.detect(im,parts,'26.3',camera=camera)
    key=frame['id']
    if split=='shelter':
        with ZipFile(suite.ARCHIVE) as z:
            packed=z.read('production/transfer/'+key+'/reconstruction.zip')
        assert sha256(packed).hexdigest()==control['pack_sha256']
        original=decoded_pack(BytesIO(packed))
    else:
        original=decoded_pack(checked(suite.CONTROL/split/key/'reconstruction.zip',control['pack_sha256']))
    gained,removed=inset_edits(im.size,camera,revised,returns,original)
    rows,z=local_release()
    row=rows[key]
    assert row['rgb_sha256']==frame['files']['rgb.png'] and row['control_pack_sha256']==control['pack_sha256']
    expected=original
    if row['methods']:
        recorded=row['methods']['clear_front']
        packed=z.read('clear_front/'+split+'/'+key+'/reconstruction.zip')
        assert sha256(packed).hexdigest()==recorded['pack_sha256']
        expected=decoded_pack(BytesIO(packed))
    built={p:v for p,v in original.items() if p not in removed}
    built.update({p:'minecraft:smooth_stone' for p in gained})
    # Datapack parser uses namespaced block IDs.
    assert built==expected, (key,len(set(built)^set(expected)),set(built.values()),set(expected.values()))
    item=dict(id=key,split=split,rgb_sha256=row['rgb_sha256'],returns=len(returns),texture=len(phase),outlines=len(outlines),
              added=len(gained),removed=len(removed),golden_pack_equal=True)
    if not (returns or phase or outlines):
        assert revised is parts, key
        return item
    target=suite.visible_map(split,frame)
    old=suite.surface_stats(parts,target,im.size)
    new=suite.surface_stats(revised,target,im.size)
    item.update(surface_gains=sorted(set(new['matched'])-set(old['matched'])),
                surface_losses=sorted(set(old['matched'])-set(new['matched'])),
                unmatched_delta=new['unmatched_predictions']-old['unmatched_predictions'],
                background_delta=new['background_pixels']-old['background_pixels'],
                paired_delta=new['paired_pixels']-old['paired_pixels'])
    return item


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=REPORT/'combined.json')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    jobs=list(suite.jobs([*suite.framed.SIZES,'shelter']))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(verify,jobs))
    changed=[r for r in rows if r['returns'] or r['texture'] or r['outlines']]
    output=dict(frames=len(rows),changed=len(changed),return_matches=sum(r['returns'] for r in rows),
                texture_matches=sum(r['texture'] for r in rows),outline_matches=sum(r['outlines'] for r in rows),gained=sum(len(r['surface_gains']) for r in changed),
                lost=sum(len(r['surface_losses']) for r in changed),unmatched_delta=sum(r['unmatched_delta'] for r in changed),
                background_delta=sum(r['background_delta'] for r in changed),
                paired_delta=sum(r['paired_delta'] for r in changed),rows=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    save(args.out,output)
    print({k:v for k,v in output.items() if k!='rows'})


if __name__=='__main__':
    main()
