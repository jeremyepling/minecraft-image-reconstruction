"""Run a local NAVER M-LSD Tiny 512 TFLite checkpoint on one screenshot."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from engine.mlsd_tiny import NAME, SETTINGS, predict


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def probe(checkpoint, image, out):
    with Image.open(image) as file:
        source = file.convert('RGB')
    segments, info = predict(source, checkpoint)
    preview = source.copy()
    pen = ImageDraw.Draw(preview)
    for s in segments:
        pen.line([tuple(s['p']), tuple(s['q'])], fill='#25ead5', width=3)

    out.mkdir(parents=True, exist_ok=False)
    preview.save(out/'lines.png')
    result = dict(model=NAME, model_file=checkpoint.name,
                  model_sha256=sha256(checkpoint), model_bytes=checkpoint.stat().st_size,
                  image=image.name, image_sha256=sha256(image), size=source.size,
                  segment_count=len(segments), segments=segments)
    result.update(SETTINGS)
    result.update(info)
    (out/'segments.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'{len(segments)} segments, first CPU invoke {result["first_invoke_ms"]} ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'image', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    probe(args.checkpoint, args.image, args.out)
