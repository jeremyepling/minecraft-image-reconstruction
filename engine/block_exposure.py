"""Compare a visible face against rendered examples under bounded exposure changes."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from calibration.retrieval import features, rank
from engine.reconstruction import exemplar_bank


class ExposureMatcher:
    sample_size = 16

    def __init__(self, atlas_path, library):
        atlas_path = Path(atlas_path)
        catalog = json.loads((atlas_path/'placement-index.json').read_bytes())
        self.version = catalog['version']
        self.mapping = defaultdict(set)
        for side in catalog['choices'].values():
            for texture, block_ids in side.items():
                self.mapping[texture].update(block_ids)
        tiles, labels, self.method = exemplar_bank(atlas_path, library, catalog)
        eligible = np.isin(labels, list(self.mapping))
        self.labels = labels[eligible]
        self.references = features(tiles[eligible])

    def rank_patches(self, patches):
        for textures, distances, margin in rank(np.asarray(patches), self.references, self.labels, self.method, True):
            ids = [min(self.mapping[t], key=lambda b:(b != t,b)) for t in textures]
            yield dict(texture=textures[0], block_id=ids[0], ranked_textures=textures,
                       ranked_ids=ids, distances=distances, score_margin=margin)

    def rank_patch(self, patch):
        return next(self.rank_patches([patch]))
