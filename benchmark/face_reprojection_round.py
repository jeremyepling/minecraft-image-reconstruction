"""Compare same-orientation visible pixels and one occlusion retry after plane choice."""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from engine.atlas_plane_depth import _atlas,options,choose
from engine.plane_offset_features import ranked_choice
from engine.visible_placement import place_with_camera


@lru_cache(maxsize=2)
def weights(path,fold):
    record=json.loads(Path(path).read_bytes())
    return record['folds'][fold] if fold else record['weights']


def run(image,version,parts,counts,camera,policy):
    rgb=np.asarray(image.convert('RGB'))
    atlas=_atlas(version,None)
    name=os.environ.get('MCIR_OFFSET_FIT')
    w=weights(name,os.environ.get('MCIR_OFFSET_EXCLUDE_FAMILY')) if name else None
    def select(part,mask,P,eye,rays,axis,baseline,dist):
        rows=options(mask,P,eye,rays,axis,baseline,rgb,atlas,dist)
        return ranked_choice(rows,baseline,w) if w is not None else choose(rows,baseline,'combined')
    partial=None
    if policy=='loose':
        partial=lambda *args:True
    if policy=='art':
        from engine.partial_atlas_face import verify
        partial=lambda *args:verify(rgb,atlas,*args)
    return place_with_camera(image,version,parts,counts,camera,select,policy,partial)


def same_face(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'same-face')


def retry(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'retry')


def loose(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'loose')


def art(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'art')


def plane(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,None)


def apply_gate(image,parts,camera,result,mode):
    # Load the independently frozen cell-gate experiment for this composition.
    import engine,benchmark
    source=Path(os.environ['MCIR_CELL_GATE_ROOT'])
    for package in (engine,benchmark):
        path=str(source/package.__name__)
        if path not in package.__path__:package.__path__=[*package.__path__,path]
    from benchmark.visible_cell_decisions import after_placement
    cells,status=result
    result=after_placement(image,parts,camera,cells,mode)
    return result,{**status,'cell_gate':mode,'placed_blocks':len(result)}


def filtered(image,version,parts,counts,camera):
    return apply_gate(image,parts,camera,plane(image,version,parts,counts,camera),'tree')


def art_filtered(image,version,parts,counts,camera):
    return apply_gate(image,parts,camera,art(image,version,parts,counts,camera),'tree')
