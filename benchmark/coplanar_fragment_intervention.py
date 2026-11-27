"""Evaluation-only intervention: give the merger recorded group assignments, then recount and place."""
import hashlib
from functools import lru_cache
from pathlib import Path

from benchmark.coplanar_fragment_join import join_regions
from benchmark.texture_camera_eval import checked,read
from benchmark.wall_texture_trial import rebuild


@lru_cache(maxsize=1)
def assignments():
    review=Path(__file__).resolve().parent/'reviews/coplanar-fragments-01'
    meta=read(review/'summary.json')
    return {r['photo_hash']:[g['mask_ids'] for g in r['thresholds']['standard']['groups']]
            for r in read(checked(review/'cases.json',meta['cases_sha256']))}


def ideal(image,parts,counts,camera,prior):
    groups=assignments()[hashlib.sha256(image.tobytes()).hexdigest()]
    if not groups:
        return prior,dict(groups=[])
    cells,detail=rebuild(image,counts,camera,join_regions(image,parts,groups))
    return cells,dict(groups=groups,**detail)
