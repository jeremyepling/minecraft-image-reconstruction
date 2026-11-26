"""Review the large-scale outline and local relief evidence in image coordinates."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from engine.line_outline import contour, cube_outline, group_lines, silhouette
from engine.structural_lines import detect


def inspect(image, local=True, cube_turns=False):
    lines = detect(image)
    mask = silhouette(image, lines)
    evidence = {}
    axes = []
    if cube_turns:
        mask, evidence['cube_turns'] = cube_outline(image, lines, mask)
        axes = evidence['cube_turns']['axes']
    grouped = group_lines(lines, mask)
    panels = []
    if local and mask.getbbox():
        from engine.line_relief import analyze
        panels = analyze(image, grouped, mask, cube_axes=axes)
    return dict(size=list(image.size), lines=grouped, panels=panels, **evidence), mask


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


def render_cube_turns(image, result):
    """Show the observed outline runs that were admitted by the cube model."""
    preview = Image.blend(image.convert('RGB'),Image.new('RGB',image.size,'#0c1520'),.28)
    pen = ImageDraw.Draw(preview)
    evidence = result['cube_turns']['contour']
    colors = ('#59d8fb','#ffcc66','#f5a6dd')
    width = max(2,round(max(image.size)/500))
    for run in evidence['runs']:
        pen.line([tuple(p) for p in run['points']],fill=colors[run['axis']],width=width)
    for turn in evidence['turns']:
        x,y=turn['point']
        pen.ellipse((x-width,y-width,x+width,y+width),fill='white',outline='#15202c')
    return preview


def inspect_file(path, out, cube_turns=False):
    with Image.open(path) as im:
        image = ImageOps.exif_transpose(im).copy()
    result, mask = inspect(image, cube_turns=cube_turns)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'form.json').write_text(json.dumps(result, indent=2) + '\n')
    mask.save(out/'mask.png')
    outline, relief = render(image, result, mask)
    outline.save(out/'outline.png')
    relief.save(out/'relief.png')
    if cube_turns:
        render_cube_turns(image,result).save(out/'cube-turns.png')
    print(f"{len(result['lines'])} segments, {len(result['panels'])} panel proposals in {out}")
