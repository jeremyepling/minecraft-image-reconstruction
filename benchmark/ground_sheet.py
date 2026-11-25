"""One controlled visual comparison of the frozen camera method on two floor materials."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.camera_sheet import INK, MUTED, PAPER, RULE, crop_box, font, label, line_on, proposal_lines

FLOORS = ('smooth_stone', 'light_gray_concrete')
NAMES = ('Smooth stone', 'Light gray concrete')
TEAL, GOLD = '#39ebe4', '#ffd058'
COL, MARGIN, GAP = 640, 38, 24
WIDTH = 2*COL + 2*MARGIN + GAP


def _corners(case):
    return np.array([case[key] for key in ('corner_px', 'width_end_px', 'height_end_px')], dtype=float)


def _pairs(example):
    # An endpoint triplet has the same world and image coordinates in a pose pair.
    # Round pixel positions only to match serialization differences between captures.
    def lookup(floor):
        cases = example['metrics'][floor]['endpoint_diagnostic']['cases']
        return {tuple(_corners(case).round(4).ravel()): case for case in cases}
    left, right = map(lookup, FLOORS)
    return [(left[k], right[k]) for k in sorted(left.keys() & right.keys())]


def _select(examples):
    held = [e for e in examples if e['split'] == 'holdout'] or examples
    full = [e for e in held if 'close-front' not in e['id']] or held
    eligible = []
    for e in full:
        for pair in _pairs(e):
            corners = _corners(pair[0])
            leg = np.linalg.norm(corners[1:]-corners[0], axis=1).min()
            predictions = e['predictions']
            loses_scale = predictions[FLOORS[0]]['P'] is not None and predictions[FLOORS[1]]['P'] is None
            # Show a metric-coverage difference, then a measurable ratio on
            # the plain floor, then the longest legible common endpoint pair.
            key = (loses_scale, pair[1]['predicted_ratio'] is not None, leg, e['id'])
            eligible.append((key, e, pair))
    if eligible:
        _, primary, pair = max(eligible, key=lambda row: row[0])
    else:
        primary, pair = full[0], None
    close = sorted((e for e in held if 'close-front' in e['id']), key=lambda e: e['id'])
    return primary, pair, next((e for e in close if e['id'] != primary['id']), None)


def _panel(source, box, panel_height, *, prediction=None, points=None, focus=None):
    region = source.crop(box)
    size = (COL, panel_height)
    scale = min(COL/region.width, panel_height/region.height)
    fitted = tuple(round(v*scale) for v in region.size)
    inset = np.array([(n-v)//2 for n, v in zip(size, fitted)])
    result = Image.new('RGB', size, '#dfe6e1')
    result.paste(region.resize(fitted, Image.Resampling.LANCZOS if scale <= 1 else Image.Resampling.NEAREST), tuple(inset))
    d = ImageDraw.Draw(result)

    def convert(xy):
        return (np.asarray(xy)-box[:2])*scale+inset

    if prediction is not None:
        for xy, boundary in proposal_lines(prediction):
            line_on(d, convert(xy), TEAL, 3 if boundary else 1)
        axes = prediction.get('axes_homogeneous', [])
        if len(axes) == 3:
            # Homogeneous directions are unoriented. Evaluating at the same
            # image location makes a direction-only fit visible on either floor.
            center = np.array([.76*source.width, .49*source.height])
            for vp in np.asarray(axes):
                delta = vp[:2]-vp[2]*center
                norm = np.linalg.norm(delta)
                if norm > 1e-8:
                    delta = 34/scale*delta/norm
                    line_on(d, convert([center-delta, center+delta]), '#ffffff', 2)
    if focus is not None:
        xy = convert(np.array(focus).reshape(2, 2))
        d.rectangle([tuple(p) for p in xy], outline=GOLD, width=2)
    if points is not None:
        c, w, h = convert(points)
        for p, color, letter in ((w, TEAL, 'W'), (h, GOLD, 'H')):
            line_on(d, [c, p], color, 3)
            # Center the letter inside the endpoint, including at the image edge.
            x, y = np.clip(p, [14, 14], np.array(size)-15)
            d.ellipse((x-12, y-12, x+12, y+12), fill=INK, outline=color, width=2)
            d.text((x, y), letter, font=font(14, True), fill='#ffffff', anchor='mm')
        d.ellipse((c[0]-5, c[1]-5, c[0]+5, c[1]+5), fill='white', outline=INK, width=2)
    return result


def _status(pred, scores):
    metric = 'Metric grid returned' if pred['P'] is not None else 'Absolute scale uncertain'
    axes = len(pred.get('axes_homogeneous', []))
    pix = scores.get('mean_px')
    suffix = f'   ·   {pix:.2f} px mean corner error' if pix is not None else ''
    return f'{metric}   ·   {axes} directions{suffix}'


def sheet(out: Path, examples: list):
    """Write comparison.png and return named feedback boxes in native source pixels."""
    if not examples:
        raise ValueError('The paired-floor review needs at least one capture pair')
    primary, pair, close = _select(examples)
    canvas = Image.new('RGB', (WIDTH, 1770), PAPER)
    d = ImageDraw.Draw(canvas)
    xs = [MARGIN, MARGIN+COL+GAP]
    regions = []

    def plot(example, boxes, y, height, name, **options):
        for j, floor in enumerate(FLOORS):
            with Image.open(example['images'][floor]) as raw:
                source = raw.convert('RGB')
            box = boxes[j] if boxes else (0, 0, *source.size)
            opts = {key: values[j] for key, values in options.items()}
            canvas.paste(_panel(source, box, height, **opts), (xs[j], y))
            regions.append(dict(image=example['id'], floor=floor, stage='camera', name=name,
                                box=list(map(int, box))))

    def rule(y):
        d.line((MARGIN, y, WIDTH-MARGIN, y), fill=RULE, width=2)

    run_id = Path(out).name
    methods = {e['predictions'][f]['method'] for e in examples for f in FLOORS}
    if len(methods) != 1:
        raise ValueError('The paired-floor review requires one frozen camera method on every image')
    label(d, (MARGIN, 25), f'JAVA 26.3    /    {run_id}    /    CAMERA METHOD   {methods.pop()}', 18, MUTED, True)
    label(d, (MARGIN, 62), 'Camera calibration on tiled and plain floors', 43, bold=True)
    label(d, (MARGIN, 127), 'Each pair uses the same building, palette, camera pose and field of view. Each prediction receives one screenshot.', 20)
    scope = [e for e in examples if e['split'] == primary['split']]
    for j, floor in enumerate(FLOORS):
        count = sum(e['predictions'][floor]['P'] is not None for e in scope)
        total_axes = sum(len(e['predictions'][floor].get('axes_homogeneous', [])) == 3 for e in scope)
        label(d, (xs[j], 181), NAMES[j], 28, bold=True)
        label(d, (xs[j], 223), f'{primary["split"].capitalize()}   ·   Metric camera {count}/{len(scope)}   ·   Three axes {total_axes}/{len(scope)}', 18, MUTED)
    rule(260)
    label(d, (MARGIN, 282), 'Full frame', 27, bold=True)
    label(d, (MARGIN+158, 288), primary['id'], 18, MUTED)
    sources = []
    for floor in FLOORS:
        with Image.open(primary['images'][floor]) as im:
            sources.append(im.size)
    if sources[0] != sources[1]:
        raise ValueError('Paired review images must have identical dimensions')
    points = [_corners(c) for c in pair] if pair else [None, None]
    detail = None
    if pair:
        lo, hi = points[0].min(0), points[0].max(0)
        span = hi-lo
        width = max(80, span[0]+45, (span[1]+45)*COL/247)
        detail = crop_box((lo+hi)/2, width, width*247/COL, sources[0])
    plot(primary, None, 332, 365, 'full-frame', prediction=[primary['predictions'][f] for f in FLOORS], focus=[detail, detail])
    for j, floor in enumerate(FLOORS):
        label(d, (xs[j], 716), _status(primary['predictions'][floor], primary['metrics'][floor]), 18, bold=True)
    label(d, (MARGIN, 750), 'Cyan grids show proposed wall planes. White lines show estimated directions at a shared pixel. Gold outlines mark the detail below.', 17, MUTED)
    rule(785)
    label(d, (MARGIN, 807), 'Width ÷ height on one visible wall', 27, bold=True)
    label(d, (MARGIN, 852), 'Diagnostic inputs are the recorded corner and two edge endpoints. Both floors use the same measurement.', 19, MUTED)
    if pair:
        plot(primary, [detail, detail], 892, 247, 'ratio-endpoints', points=points)
        for j, c in enumerate(pair):
            ratio, truth = c['predicted_ratio'], c['expected_ratio']
            estimate = f'Estimated {ratio:.3f}' if ratio is not None else 'Estimate unavailable'
            error = '' if ratio is None else f'   ·   {c["abs_error_pct"]:.1f}% error'
            label(d, (xs[j], 1153), f'{estimate}   ·   Recorded {truth:.3f}{error}', 19, bold=True)
    else:
        for x in xs:
            label(d, (x, 907), 'No shared labeled endpoint triplet', 19, MUTED)
    bottom = 1190 if pair else 952
    if close:
        rule(bottom+6)
        label(d, (MARGIN, bottom+24), 'Facade-only control', 27, bold=True)
        label(d, (MARGIN+286, bottom+31), close['id'], 18, MUTED)
        label(d, (MARGIN, bottom+64), 'These are native close views. The floor lies outside both frames.', 18, MUTED)
        # The central strip preserves the texture's pixel scale as the sheet is enlarged.
        with Image.open(close['images'][FLOORS[0]]) as im:
            strip = crop_box(np.array(im.size)/2, min(940, im.width), min(212, im.height), im.size)
        plot(close, [strip, strip], bottom+101, 154, 'facade-only-detail')
        close_pairs = _pairs(close)
        close_pair = next((p for p in close_pairs if p[1]['predicted_ratio'] is not None),
                          close_pairs[0] if close_pairs else None)
        for j, floor in enumerate(FLOORS):
            label(d, (xs[j], bottom+272), _status(close['predictions'][floor], close['metrics'][floor]), 18)
            if close_pair:
                ratio = close_pair[j]['predicted_ratio']
                estimate = f'{ratio:.3f}' if ratio is not None else 'unavailable'
                label(d, (xs[j], bottom+304), f'Full-frame endpoint ratio   {estimate}   ·   Recorded {close_pair[j]["expected_ratio"]:.3f}', 18, bold=True)
        bottom += 350 if close_pair else 322
    rule(bottom+4)
    label(d, (MARGIN, bottom+24), f'{primary["split"].capitalize()} ratio coverage', 25, bold=True)
    for j, floor in enumerate(FLOORS):
        for k, (key, name) in enumerate((('endpoint_diagnostic', 'Given the endpoints'), ('automatic_charts', 'Automatic wall planes'))):
            metrics = [e['metrics'][floor] for e in scope]
            rows = [m[key] for m in metrics]
            count = sum(row['measured'] for row in rows)
            total = sum(row['total'] for row in rows)
            per_layout = []
            for layout in sorted({m['layout'] for m in metrics}):
                errors = [m[key]['mean_abs_error_pct'] for m in metrics
                          if m['layout'] == layout and m[key]['mean_abs_error_pct'] is not None]
                if errors:
                    per_layout.append(np.mean(errors))
            error = f'{np.mean(per_layout):.2f}% mean error' if per_layout else 'Error unavailable'
            label(d, (xs[j], bottom+65+k*31), f'{name}   ·   {count}/{total} measured   ·   {error}', 18, bold=k==0)
    label(d, (MARGIN, bottom+137), 'Mean absolute percentage error weights layouts equally on returned estimates. Counts include every eligible ratio.', 17, MUTED)
    canvas = canvas.crop((0, 0, WIDTH, bottom+178))
    canvas.save(Path(out)/'comparison.png')
    return regions
