"""Benchmark an offset ranker with its weights or development fold fixed at launch."""
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from engine.atlas_plane_depth import place


@lru_cache(maxsize=1)
def weights():
    fit=json.loads(Path(os.environ['MCIR_OFFSET_FIT']).read_bytes())
    fold=os.environ.get('MCIR_OFFSET_EXCLUDE_FAMILY')
    return np.array(fit['folds'][fold] if fold else fit['weights'])


def predict(image,version,parts,counts,camera):
    return place(image,version,parts,counts,camera,'ranked',weights=weights())
