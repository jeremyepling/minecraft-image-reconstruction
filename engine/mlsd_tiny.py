"""Native screenshot line segments from the NAVER M-LSD Tiny 512 FP16 checkpoint."""
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from engine.atlas import digest


NAME = 'NAVER M-LSD Tiny 512 FP16'
MODEL = ('MINECRAFT_MLSD_TFLITE', 'M-LSD_512_tiny_fp16.tflite',
         'e869f90601ffa263f21ac47e09ff0d65dc936d1aeb47126b8499d5504adfb100')
SETTINGS = dict(runtime='ai-edge-litert 2.4.0', num_threads=1,
                preprocessing='Pillow BOX resize to 512 square, float32 RGB plus 1.0 alpha',
                score_threshold=.10, min_length_output_pixels=20)


def predict(source, checkpoint):
    """Run the original probe's preprocessing and decoder. Return scored source-space endpoints."""
    try:
        release = version('ai-edge-litert')
        from ai_edge_litert.interpreter import Interpreter
    except (PackageNotFoundError, ImportError) as error:
        raise RuntimeError('Install ai-edge-litert==2.4.0 in the Python environment for M-LSD Tiny.') from error
    start = perf_counter()
    runtime = Interpreter(model_path=str(checkpoint), num_threads=SETTINGS['num_threads'])
    runtime.allocate_tensors()
    spec, = runtime.get_input_details()
    if tuple(spec['shape']) != (1, 512, 512, 4) or spec['dtype'] != np.float32:
        raise ValueError('Expected the NAVER M-LSD 512 float-input TFLite model')
    batch = np.ones((1, 512, 512, 4), dtype=np.float32)
    batch[0, :, :, :3] = np.asarray(source.convert('RGB').resize((512, 512), Image.Resampling.BOX))
    runtime.set_tensor(spec['index'], batch)
    ready = perf_counter()
    runtime.invoke()
    done = perf_counter()
    centers, scores, offsets = [runtime.get_tensor(d['index'])[0] for d in runtime.get_output_details()]
    if centers.shape != (200, 2) or scores.shape != (200,) or offsets.shape != (256, 256, 4):
        raise ValueError('Unexpected M-LSD output signature')
    scale = np.array(source.size) / np.array(offsets.shape[:2][::-1])
    segments = []
    for (y, x), score in zip(centers, scores):
        ends = offsets[y, x].reshape(2, 2)
        if score <= SETTINGS['score_threshold'] or np.linalg.norm(ends[0] - ends[1]) <= SETTINGS['min_length_output_pixels']:
            continue
        p, q = (ends + [x, y]) * scale
        segments.append(dict(p=p.tolist(), q=q.tolist(), score=float(score)))
    return segments, dict(runtime=f'ai-edge-litert {release}', first_invoke_ms=round(1000*(done-ready), 3),
                          load_through_decode_ms=round(1000*(perf_counter()-start), 3))


def detect(source):
    env, name, sha = MODEL
    checkpoint = Path(os.environ.get(env) or Path(__file__).resolve().parents[1]/'.deps'/name)
    if not checkpoint.is_file() or digest(checkpoint) != sha:
        raise ValueError(f'Provide the pinned M-LSD Tiny model via {env} or run python -m benchmark.normal_assets --mlsd --download')
    segments, info = predict(source, checkpoint)
    if info['runtime'] != SETTINGS['runtime']:
        raise ValueError(f"The strip finder uses {SETTINGS['runtime']}. Install ai-edge-litert==2.4.0.")
    return dict(image_size=list(source.size), junctions=[], segments=[s['p']+s['q'] for s in segments])
