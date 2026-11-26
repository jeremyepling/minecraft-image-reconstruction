"""Offline adapter for calibrated relative-depth placement. Only RGB indexes the model cache."""
import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from engine import ground_depth, visible_placement
from engine.surface_fields import infer


def depth(image):
    path = os.environ.get('MCIR_DEPTH_EXPERIMENT_CACHE')
    if path:
        cache = Path(path) / (hashlib.sha256(image.tobytes()).hexdigest()+'.npy')
        if cache.exists():
            return np.load(cache, allow_pickle=False)
    return infer(image,'depth')


def predict(image,version,parts,counts,camera):
    original = visible_placement._estimate_plane_cells
    def estimated(img,parts,masks,P,eye):
        levels = ground_depth.plane_levels(img,parts,masks,P,eye,depth(img))
        return original(img,parts,masks,P,eye,levels)
    with patch.object(visible_placement,'seam_camera',return_value=camera), \
         patch.object(visible_placement,'_estimate_plane_cells',side_effect=estimated):
        return visible_placement.place(image,version,parts,counts=counts)
