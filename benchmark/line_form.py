"""Review the large-scale outline and local relief evidence in image coordinates."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from engine.line_outline import contour, group_lines, silhouette
from engine.structural_lines import detect


def inspect(image, local=True):
    lines = detect(image)
    mask = silhouette(image, lines)
    grouped = group_lines(lines, mask)
    panels = []
    if local and mask.getbbox():
        from engine.line_relief import analyze
        panels = analyze(image, grouped, mask)
    return dict(size=list(image.size), lines=grouped, panels=panels), mask


def render(image, result, mask):
    source = image.convert('RGB')
    base = np.array(Image.blend(source, Image.new('RGB', image.size, '#0c1520'), .32))
    base[contour(mask)] = [255, 209, 90]
    outline = Image.fromarray(base)
    pen = ImageDraw.Draw(outline)
    for line in result['lines']:
        if line['scale'] != 'coarse' or line['role'] == 'background':
            continue
        color = '#ffd15a' if line['role'] == 'outline' else '#78cfef'
        pen.line([tuple(line['p']), tuple(line['q'])], fill=color, width=2)
    relief = source.copy()
    pen = ImageDraw.Draw(relief)
    colors = dict(recess='#25f3cc', protrusion='#fd9ee1', flat='#90c9ff', unknown='#f7d380')
    for panel in result['panels']:
        points = [tuple(p) for p in panel['quad']]
        color = colors[panel['label']]
        pen.line(points + points[:1], fill=color, width=2)
        if panel.get('connector'):
            pen.line([tuple(p) for p in panel['connector']], fill='white', width=3)
        pen.text(points[0], panel['label'], fill=color, stroke_width=1, stroke_fill='black')
    return outline, relief


def inspect_file(path, out):
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).copy()
    result, mask = inspect(image)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'form.json').write_text(json.dumps(result, indent=2) + '\n')
    mask.save(out/'mask.png')
    outline, relief = render(image, result, mask)
    outline.save(out/'outline.png')
    relief.save(out/'relief.png')
    print(f"{len(result['lines'])} segments, {len(result['panels'])} panel proposals in {out}")
