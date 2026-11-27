"""Source-only SAM click interface; callers supply their own prompt policy."""
import subprocess
import sys

import numpy as np
import torch

from benchmark.sky_crop_models import IMPLEMENTATION, IMPLEMENTATION_REV, MODELS, WORKTREES, digest


def make_predictor():
    revision = subprocess.check_output(['git', '-C', str(IMPLEMENTATION), 'rev-parse', 'HEAD'], text=True).strip()
    key, _, checkpoint_hash = MODELS['sam-b']
    checkpoint = WORKTREES / 'sky-crop-models/runs/sky-crop-models/sam_vit_b_01ec64.pth'
    if revision != IMPLEMENTATION_REV or digest(checkpoint) != checkpoint_hash:
        raise ValueError('Unexpected SAM weights or implementation')
    sys.path.insert(0, str(IMPLEMENTATION))
    from mobile_sam import SamPredictor, sam_model_registry
    model = sam_model_registry[key]()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
    return SamPredictor(model.eval().to('cuda'))


@torch.inference_mode()
def predict_clicks(predictor, box_xyxy_inclusive, points_xy, labels):
    masks, scores, _ = predictor.predict(box=np.asarray(box_xyxy_inclusive),
                                         point_coords=np.asarray(points_xy),
                                         point_labels=np.asarray(labels), multimask_output=True)
    return masks, scores
