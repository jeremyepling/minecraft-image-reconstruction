"""Measure the opportunity in proposed spans, opening reference labels after prediction."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from benchmark import framed_depth_eval as data
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import best_translation
from benchmark.texture_camera_eval import ROOT, save
from engine import roof_span_contacts as span
from engine.visible_horizontal_faces import _surface_masks
from engine.visible_placement import _frontmost_cells


def case(job):
    split,f,row=job
    image,parts,counts,camera=data.photo(split,f,row)
    with ZipFile(data.REVIEW/'framed-depth-01/scores.zip') as z:
        prior=set(decoded_pack(BytesIO(z.read(f"{split}/{f['id']}/reconstruction.zip"))))
    candidates=set()
    witnesses=Counter()
    def record(image,parts,P,native,eye,rot,base,proposals):
        candidates.update(set.union(*(cells for _,_,cells in proposals)))
        cells=sorted(base|candidates)
        ids={cell:i for i,cell in enumerate(cells)}
        front,axes=_frontmost_cells(cells,native,eye,image.size[::-1],face_axes=True)
        for axis,part,new in proposals:
            mask=_surface_masks([part],image.size,np.arange(image.width)+.5,
                                np.arange(image.height)+.5,rot)[0]
            hits=np.bincount(front[mask & (axes==axis)],minlength=len(cells))
            for cell in new:
                witnesses[cell]=max(witnesses[cell],hits[ids[cell]])
        return base
    with patch.object(span,'verify_rectangles',record):
        span.recover(image,parts,counts,camera,prior)
    truth,occupied,T,_=data.references(split,f,None,data.TRAINING_DATA,data.DEPTH,{})
    before=registered(prior,T)
    shift,_=best_translation(before,truth)
    visible=set()
    for cell in candidates:
        w=next(iter(registered({cell},T)))
        if tuple(np.asarray(w)+shift) in truth:
            visible.add(cell)
    return dict(id=f['id'],family=f['family'],proposed=len(candidates),visible=len(visible),
                visible_without_12_pixels=int(sum(witnesses[c]<12 for c in visible)),
                visible_no_pixel=int(sum(witnesses[c]==0 for c in visible)))


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(case,data.cases(['train'])))
    folder=ROOT/'runs/roof-span-contacts'
    folder.mkdir(exist_ok=True,parents=True)
    save(folder/'ceiling.json',rows)
    print({k:sum(r[k] for r in rows) for k in rows[0] if k not in ('family','id')})
