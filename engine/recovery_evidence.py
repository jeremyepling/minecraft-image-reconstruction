"""Single-screenshot ScaleLSD and MobileSAM evidence for the opt-in Wall Recovery method."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _model_input(variable, default):
    path = Path(os.environ.get(variable) or ROOT/'.deps'/default).resolve()
    if not path.exists():
        raise FileNotFoundError(f'Provide the pinned model input via {variable} or {path}')
    return path


def infer(source, walls):
    """Use the configured model interpreter while retaining the ordinary builder's numerical environment."""
    interpreter = os.environ.get('MCIR_RECOVERY_PYTHON') or sys.executable
    with TemporaryDirectory() as private:
        folder = Path(private)
        (folder/'walls.json').write_text(json.dumps([{'support': p['support']} for p in walls]), encoding='utf8')
        subprocess.run([interpreter, '-B', '-m', 'engine.recovery_evidence', str(Path(source).resolve()), private],
                       cwd=ROOT, check=True)
        result = json.loads((folder/'evidence.json').read_bytes())
        with np.load(folder/'masks.npz', allow_pickle=False) as payload:
            masks = np.unpackbits(payload['packed'], count=int(payload['width']), axis=2).astype(bool)
    return result, masks


def _compute(source, folder):
    os.environ.setdefault('MPLCONFIGDIR', str(folder/'matplotlib'))
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(folder/'torch-extensions'))
    from importlib.metadata import version
    import cv2
    import torch
    from benchmark import mobile_region_probe as mobile
    from benchmark import scalelsd_fields as scale

    device = os.environ.get('MCIR_RECOVERY_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
    scale_source = _model_input('MCIR_SCALELSD_SOURCE', 'scalelsd')
    checkpoint = _model_input('MCIR_SCALELSD_CHECKPOINT', scale.MODEL_NAME)
    mobile_source = _model_input('MCIR_MOBILESAM_SOURCE', 'MobileSAM')
    if scale.revision(scale_source) != scale.UPSTREAM_REV or scale.digest(checkpoint) != scale.MODEL_SHA256:
        raise ValueError('ScaleLSD source revision or v2 checkpoint differs from the frozen experiment')
    for checkout in (mobile_source, scale_source):
        if subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no']):
            raise ValueError(f'Expected pristine model source at {checkout}')
    with Image.open(source) as src:
        image = src.convert('RGB')
    walls = json.loads((folder/'walls.json').read_bytes())
    predictor = mobile.get_predictor(mobile_source, device)
    points, records, selected, masks, elapsed = mobile.make_proposals(predictor, image, walls)
    del predictor
    np.savez_compressed(folder/'masks.npz', packed=np.packbits(masks, axis=2), width=masks.shape[2])

    sys.path.insert(0, str(scale_source))
    from scalelsd.ssl.misc.train_utils import fix_seeds, load_scalelsd_model
    fix_seeds(scale.SETTINGS['seed'])
    model = load_scalelsd_model(str(checkpoint), device)
    model.junction_threshold_hm = scale.SETTINGS['junction_heatmap_threshold']
    model.num_junctions_inference = scale.SETTINGS['max_junctions']
    # Decode in the detector's native grayscale convention, including non-ASCII paths on Windows.
    gray = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape[::-1] != image.size:
        raise ValueError('OpenCV could not decode the screenshot at its native dimensions')
    with torch.inference_mode():
        lines = scale.capture(model, gray, device, verify_transform=False)
    provenance = dict(device=device, model_environment={p:version(p) for p in ('torch','torchvision','timm','numpy','Pillow')},
        scalelsd=dict(source_revision=scale.UPSTREAM_REV, checkpoint_revision=scale.MODEL_REV,
                      checkpoint_sha256=scale.MODEL_SHA256, settings=scale.SETTINGS, line_count=len(lines['segments']),
                      native_transform=lines['transform'], model_seconds=lines['inference_seconds']),
        mobilesam=dict(source_revision=mobile.SOURCE, checkpoint_sha256=mobile.WEIGHTS, settings=mobile.SETTINGS,
                       prompts_from='wall-depth-anything', prompt_count=len(points), candidates=len(records),
                       filtered=len(selected), model_seconds=elapsed))
    result = dict(lines=lines, records=records, selected=selected, provenance=provenance)
    (folder/'evidence.json').write_text(json.dumps(result, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    _compute(args.source, args.folder)
