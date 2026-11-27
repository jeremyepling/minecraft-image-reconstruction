"""Cache pre-inset supports from frozen RGB fields on camera-supported expanded views."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
from unittest.mock import patch

from benchmark.expanded_baseline import ROOT, SOURCE, COHORTS, frames, read, save, replay_source_fields

BASE = ROOT.parent / 'full-expanded-controls/runs/expanded-baseline-01'
RUN = ROOT / 'runs/visible-boundary-direction-01'
CASES = BASE / 'scores/cases.json'


def configure():
    models = ROOT.parent / 'surface-strip-entry/.deps'
    for key, name in [('MINECRAFT_MOGE_ONNX','moge-2-vits.onnx'), ('MINECRAFT_DEPTH_ONNX','depth-v2-small-uint8.onnx'),
                      ('MINECRAFT_MLSD_TFLITE','M-LSD_512_tiny_fp16.tflite')]:
        os.environ[key] = str(models / name)
    os.environ.update(MCIR_RECOVERY_DEVICE='cuda', MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')


class Captured(Exception):
    pass


def capture(entry):
    from engine import surface_model, inset_returns
    folder = RUN / entry['dataset'] / entry['id']
    path = folder / 'pre-inset.json.gz'
    if path.exists():
        return entry['id']
    folder.mkdir(parents=True,exist_ok=True)
    image = Path(entry['frame']) / 'rgb.png'
    def grab(image, parts, *args, **kwargs):
        save(path, parts)
        raise Captured
    with replay_source_fields(image, BASE / entry['dataset'] / entry['id']), patch.object(inset_returns, 'detect', grab):
        try:
            surface_model.build(image, '26.3', folder, through='placement')
        except Captured:
            return entry['id']
    raise ValueError('Missing pre-inset stage')


def main():
    configure()
    usable = {(r['dataset'], r['id']) for r in read(CASES) if r['camera_usable']}
    entries = [e for e in frames(SOURCE, COHORTS) if (e['dataset'],e['id']) in usable]
    with ProcessPoolExecutor(3, initializer=configure) as pool:
        for result in pool.map(capture, entries):
            print(result,flush=True)


if __name__=='__main__':
    main()
