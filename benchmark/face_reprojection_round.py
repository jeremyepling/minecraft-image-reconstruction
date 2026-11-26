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
    return place_with_camera(image,version,parts,counts,camera,select,policy)


def same_face(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'same-face')


def retry(image,version,parts,counts,camera):
    return run(image,version,parts,counts,camera,'retry')
