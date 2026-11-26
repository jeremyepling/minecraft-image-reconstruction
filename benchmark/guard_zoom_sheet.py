"""Draw matched surface outlines and the texture grids downstream of the same proposals."""
import numpy as np
from PIL import Image

from benchmark.afterpass_sheet import plane_overlay
from benchmark.camera_round import digest, read
from benchmark.count_metrics import reference_planes
from benchmark.guard_zoom_round import NAMES
from benchmark.ledge_join_sheet import describe, render_comparison
from benchmark.mask_metrics import footprint
from benchmark.plane_acceptance_round import BASE, PLANES
from benchmark.planarity_afterpass import raw_diagnostic
from benchmark.surface_pipeline_sheet import _straight_photo
from benchmark.surface_reference import recorded
from benchmark.surface_topology import opening_mask
from benchmark.wall_join_test import read_gzip
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_sample import sample_surface


def opening_panels(put, out, row, capture):
    with Image.open(capture.path('image')) as original:
        photo = original.convert('RGB')
    _, ref = recorded(capture)
    openings = opening_mask(ref['id_map'])
    parts = read_gzip(out / row['predictions'])
    filled = {}
    for m in NAMES:
        occupied = np.zeros_like(openings, dtype=bool)
        for part in parts[m]:
            occupied |= footprint(part, openings.shape, photo.size, True)
        filled[m] = openings & occupied
        if int(filled[m].sum()) != row['scores'][m]['opening_filled']:
            raise ValueError('Opening overlay disagrees with measured coverage')
    focus = filled['zoom'] & ~filled['current']
    if not focus.any():
        focus = openings
    y, x = np.nonzero(focus)
    box = (0, 0, *photo.size)
    if x.size:
        sx, sy = photo.width/openings.shape[1], photo.height/openings.shape[0]
        box = (max(0, int(x.min()*sx)-50), max(0, int(y.min()*sy)-50),
               min(photo.width, int((x.max()+1)*sx)+50), min(photo.height, int((y.max()+1)*sy)+50))
    def tint(mask, color):
        scaled = Image.fromarray(mask).resize(photo.size, Image.Resampling.NEAREST)
        colored = Image.blend(photo, Image.new('RGB', photo.size, color), .66)
        return Image.composite(colored, photo, scaled).crop(box)
    put('opening-source', photo.crop(box), 'Original opening detail', describe(row))
    put('opening-truth', tint(openings, '#ffd050'), 'Gold marks scored open space', f"{openings.sum()} opening samples in this view")
    for m, label in NAMES.items():
        n = int(filled[m].sum())
        put('opening-'+m, tint(filled[m], '#fc4381'), label, f'{n} opening samples covered in the view')
    return dict(frame=row['key'], crop_xyxy=box, opening_samples=int(openings.sum()),
                filled_samples={m:int(mask.sum()) for m, mask in filled.items()})


def render(out, rows, captures, table, zoom):
    from benchmark.zoom_plane_fields import render_crop_proposals
    raw_stages = [('whole', 'Whole-image planes', 'Colored outlines show regions from the original Planarity pass.'),
                  ('crops', 'Closer-view planes', 'Outlines come from the same checkpoint on overlapping crops.')]
    downstream = []
    for method, title in NAMES.items():
        downstream.extend([
            ('sample-' + method, title, 'Straightened sample', 'The same selected guess, turned face-on. Gaps stay masked.', 200),
            ('count-' + method, title, 'Block Counter', 'Yellow marks measured grid lines. A question mark means unresolved.', 200)])
    downstream.extend([
        ('opening-source', 'Openings up close', 'Original', 'For each building, use the view with the largest increase in covered opening samples.', 210),
        ('opening-truth', 'Open space', 'Scored reference', 'Gold marks background between or beneath building surfaces.', 210)])
    downstream.extend(('opening-'+m, title, 'Scored prediction', 'Pink marks predicted surface pixels inside a scored opening.', 210)
                      for m, title in NAMES.items())
    def opening_rank(r):
        zoom_filled = r['scores']['zoom']['opening_filled']
        return zoom_filled-r['scores']['current']['opening_filled'], zoom_filled, r['scores']['zoom']['opening_pixels']
    opening_views = {layout:max((r for r in rows if r['layout']==layout), key=opening_rank)
                     for layout in {r['layout'] for r in rows}}

    def raw_panels(put, photo, row, box, prefix, focus=None):
        field = PLANES / 'fields' / (row['sha256'] + '.npz')
        with np.load(field, allow_pickle=False) as archive:
            labels = archive['relaxed']
        caption = 'Original whole-image prediction'
        if focus is not None:
            _, ref = recorded(captures[row['key']])
            match, = raw_diagnostic(labels, ref, [focus])
            caption = f"Best raw overlap for R{focus}: {match['best_iou']:.0%}"
        put(prefix+'whole', plane_overlay(photo, labels).crop(box), 'Model output before acceptance', caption)
        crop_field = zoom / 'crop-fields' / 'fields' / (row['sha256'] + '.npz')
        crop_overlay = render_crop_proposals(photo, crop_field, box=box)
        put(prefix+'crops', crop_overlay, 'Crop predictions in source coordinates', 'Same original pixels and checkpoint')
        return {'whole_image_fields_sha256': digest(field), 'crop_fields_sha256': digest(crop_field)}

    def extra_details(put, data, row, selection):
        photo, ref, _, grades = data
        rid = selection['focus_reference']
        expected = reference_planes(ref, read(captures[row['key']].path('labels'))['faces'])[rid]['size']
        out_status = {}
        for method, grade in grades.items():
            near = grade['lookup'][rid]
            index = near['index']
            part = grade['parts'][index-1] if index else None
            sample = sample_surface(photo, part) if part is not None else None
            if sample is None:
                clean = grid = Image.new('RGB', (180, 96), '#293233')
                spans = [None, None]
                label = 'No straightened sample'
            else:
                reading = count(sample)
                clean, grid, _ = _straight_photo(sample, reading)
                spans = dimensions(part, sample['quad'], photo.size, reading['counts'])['blocks']
                label = 'Matched surface' if near['found'] else 'Closest guess to missed surface'
            put('sample-'+method, clean, label, f"R{rid} reference · {near['overlap']:.0%} outline overlap")
            guess = ' × '.join('?' if x is None else str(x) for x in spans)
            put('count-'+method, grid, 'Predicted ' + guess, f'Recorded R{rid}: {expected[0]} × {expected[1]} blocks')
            out_status[method] = dict(reference=rid, expected=expected, found=near['found'],
                                      proposal=part['id'] if part else None, blocks=spans)
        opening = opening_views[row['layout']]
        return dict(surfaces=out_status, openings=opening_panels(put, out, opening, captures[opening['key']]))

    return render_comparison(out, rows, captures, table, NAMES,
        'Can protected wall edges or closer views find more surfaces?',
        raw_stages=raw_stages, raw_panels=raw_panels, detail_stages=downstream, detail_renderer=extra_details,
        prior_detail_sheet=BASE.parent/'plane-acceptance-01/sheet-views.json')
