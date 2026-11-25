"""Run a local NAVER M-LSD Tiny 512 TFLite checkpoint on one screenshot."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def probe(checkpoint, image, out):
    from ai_edge_litert.interpreter import Interpreter

    start = perf_counter()
    runtime = Interpreter(model_path=str(checkpoint), num_threads=1)
    runtime.allocate_tensors()
    spec, = runtime.get_input_details()
    if tuple(spec['shape']) != (1, 512, 512, 4) or spec['dtype'] != np.float32:
        raise ValueError('Expected the NAVER M-LSD 512 float-input TFLite model')
    with Image.open(image) as file:
        source = file.convert('RGB')
    batch = np.ones((1, 512, 512, 4), dtype=np.float32)
    batch[0, :, :, :3] = np.asarray(source.resize((512, 512), Image.Resampling.BOX))
    runtime.set_tensor(spec['index'], batch)
    ready = perf_counter()
    runtime.invoke()
    done = perf_counter()

    # Center indices, center scores and endpoint displacements follow NAVER's
    # published pred_lines interface. Its score and length defaults apply here.
    centers, scores, offsets = [runtime.get_tensor(d['index'])[0] for d in runtime.get_output_details()]
    if centers.shape != (200, 2) or scores.shape != (200,) or offsets.shape != (256, 256, 4):
        raise ValueError('Unexpected M-LSD output signature')
    scale = np.array(source.size) / np.array(offsets.shape[:2][::-1])
    segments = []
    preview = source.copy()
    pen = ImageDraw.Draw(preview)
    for (y, x), score in zip(centers, scores):
        ends = offsets[y, x].reshape(2, 2)
        if score <= .10 or np.linalg.norm(ends[0] - ends[1]) <= 20:
            continue
        p, q = (ends + [x, y]) * scale
        segments.append(dict(p=p.tolist(), q=q.tolist(), score=float(score)))
        pen.line([tuple(p), tuple(q)], fill='#25ead5', width=3)
    end = perf_counter()

    out.mkdir(parents=True, exist_ok=False)
    preview.save(out/'lines.png')
    result = dict(model='NAVER M-LSD Tiny 512 FP16', model_file=checkpoint.name,
                  model_sha256=sha256(checkpoint), model_bytes=checkpoint.stat().st_size,
                  image=image.name, image_sha256=sha256(image), size=source.size,
                  runtime=f'ai-edge-litert {version("ai-edge-litert")}', num_threads=1,
                  preprocessing='Pillow BOX resize to 512 square, float32 RGB plus 1.0 alpha',
                  score_threshold=.10, min_length_output_pixels=20, segment_count=len(segments),
                  first_invoke_ms=round(1000*(done-ready), 3),
                  load_through_decode_ms=round(1000*(end-start), 3), segments=segments)
    (out/'segments.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'{len(segments)} segments, first CPU invoke {result["first_invoke_ms"]} ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'image', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    probe(args.checkpoint, args.image, args.out)
