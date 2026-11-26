"""Draw learned second-pass predictions and their numerical inputs at the same source coordinates."""
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from benchmark.camera_round import digest
from benchmark.ledge_join_sheet import render_comparison
from benchmark.surface_detection_sheet import paint
from benchmark.surface_reference import recorded
from benchmark.planarity_afterpass import raw_diagnostic


NAMES = {'current':'Previous finder', 'planarity':'Pixel-wise Planarity', 'ect':'ECT edge model'}
RAW = {
    'planarity': [
        ('planes', 'Plane proposals', 'Each color identifies a plane in the raw model segmentation.'),
    ],
    'ect': [
        ('shape-edges', 'Shape edges', 'Brighter blue marks stronger evidence for a depth step or a turn.'),
        ('appearance-edges', 'Texture and light', 'Brighter purple marks stronger evidence for a material or lighting change.'),
    ],
}


def redraw(run, out):
    from benchmark.afterpass_round import FIELDS, validate_fields
    from benchmark.camera_round import read
    from benchmark.normal_round import groups
    from benchmark.surface_reference import ARCHIVE
    from benchmark.wall_join_test import checked
    manifest = read(run/'manifest.json')
    checked(ARCHIVE, manifest['reference_sha256'])
    cohorts, _, fixtures = groups()
    if manifest['fixtures'] != fixtures:
        raise ValueError('Different fixture set')
    captures = {f'{c.floor}/{c.id}':c for entries in cohorts.values() for c in entries}
    rows = read(checked(run/'cases.json', manifest['artifacts']['cases.json']))
    scores = read(checked(run/'scores.json', manifest['artifacts']['scores.json']))
    for row in rows:
        checked(run/row['predictions'], manifest['artifacts'][row['predictions']])
    roster = {r['key']:r['sha256'] for r in rows}
    paths = {m:FIELDS/m for m in manifest['candidates']}
    for m, path in paths.items():
        checked(path/'manifest.json', manifest['candidates'][m]['fields_manifest_sha256'])
        validate_fields(path, fixtures, roster)
    winner = run/'candidate-runs/planarity/manifest.json'
    meta = read(checked(winner, manifest['candidates']['planarity']['manifest_sha256']))
    out.mkdir(parents=True, exist_ok=False)
    return render(out, rows, captures, scores, fields=paths, labels_key=meta['sheet_labels_key'], source=run)


def plane_overlay(photo, labels):
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Expected a 2D array of nonnegative plane IDs')
    ids = np.asarray(Image.fromarray(labels.astype(np.int32)).resize(photo.size, Image.Resampling.NEAREST))
    colors = {}
    for i in np.unique(ids):
        if i:
            rgb = colorsys.hsv_to_rgb((int(i) * .61803399) % 1., .77, .98)
            colors[int(i)] = '#%02x%02x%02x' % tuple(round(c * 255) for c in rgb)
    return paint(photo, ids, colors)


def edge_overlay(photo, a, b, color):
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('Expected two aligned 2D edge maps')
    values = np.maximum(a, b)
    if values.dtype == np.uint8:
        probability = Image.fromarray(values)
    elif (np.issubdtype(values.dtype, np.floating) and np.isfinite(values).all()
          and values.min() >= 0 and values.max() <= 1):
        probability = Image.fromarray(np.rint(values * 255).astype(np.uint8))
    else:
        raise ValueError('Expected uint8 or 0–1 edge strengths')
    strength = probability.resize(photo.size, Image.Resampling.BILINEAR)
    backdrop = ImageOps.colorize(photo.convert('L'), (14, 19, 22), (125, 133, 136))
    return Image.composite(Image.new('RGB', photo.size, color), backdrop, strength)


def render(out, rows, captures, table, *, fields=None, labels_key='labels', source=None):
    """Field directories contain source-aligned <image sha256>.npz files.

    The planarity candidate manifest may specify sheet_labels_key for its chosen segmentation.
    Without maps the comparison still shows the scored predictions for every method that ran.
    """
    out = Path(out)
    ranked = table['all']
    names = {m: name for m, name in NAMES.items() if m in ranked}
    if len(names) < 2 or 'current' not in names:
        raise ValueError('Need the current finder and a measured afterpass')
    if fields is None:
        fields = {m: out/'candidate-runs'/m/'fields' for m in names if m != 'current'}
        fields = {m: p for m, p in fields.items() if p.is_dir()}
    paths, stages = {}, []
    for m, path in fields.items():
        if m not in names:
            raise ValueError(f'Fields supplied for unscored method {m}')
        path = Path(path)
        if (path/'fields').is_dir():
            path = path/'fields'
        if not path.is_dir():
            raise ValueError(f'Missing field directory: {path}')
        paths[m] = path
        stages.extend(RAW[m])
    def draw_fields(put, photo, row, box, prefix, focus=None):
        sha = row['sha256']
        used = {}
        for m, directory in paths.items():
            path = directory/(sha+'.npz')
            with np.load(path, allow_pickle=False) as arrays:
                if m == 'planarity':
                    labels = arrays[labels_key]
                    overlay = plane_overlay(photo, labels)
                    caption = 'Colors identify separate proposals'
                    if focus is not None:
                        _, ref = recorded(captures[row['key']])
                        best, = raw_diagnostic(labels, ref, [focus])
                        caption = f"Best raw outline for R{focus}: {best['best_iou']:.0%} overlap"
                    put(prefix+'planes', overlay.crop(box), 'Raw plane regions', caption)
                    shape = labels.shape
                    keys = [labels_key]
                else:
                    keys = ['depth', 'normal', 'illumination', 'reflectance']
                    depth, normal, light, material = (arrays[k] for k in keys)
                    shape = depth.shape
                    if light.shape != shape:
                        raise ValueError(f'Differently sized cause maps for {sha}')
                    for key, a, b, color, caption in (
                        ('shape-edges', depth, normal, '#37d6ff', 'Depth steps and surface turns'),
                        ('appearance-edges', light, material, '#da97ff', 'Lighting and material changes'),
                    ):
                        overlay = edge_overlay(photo, a, b, color)
                        put(prefix+key, overlay.crop(box), caption, 'Brightness shows model response')
            used[m] = dict(file=path.name, sha256=digest(path), channels=keys, grid_hw=list(shape),
                           source_wh=list(photo.size), source_image_sha256=sha)
        return used

    return render_comparison(out, rows, captures, table, names, 'Which second pass recovers the missing surfaces?',
                             raw_stages=stages, raw_panels=draw_fields if paths else None, source=source)
