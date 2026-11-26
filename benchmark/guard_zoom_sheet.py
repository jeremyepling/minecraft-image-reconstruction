"""Draw matched surface outlines and the texture grids downstream of the same proposals."""
import numpy as np
from PIL import Image, ImageChops, ImageFilter

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
from benchmark.wall_join_test import checked, read_gzip
from engine.masked_count import count
from engine.surface_dimensions import dimensions
from engine.surface_perimeter import _native_alpha
from engine.surface_sample import sample_surface


def opening_panels(put, out, row, capture, names, box=None, earlier_zoom=None):
    with Image.open(capture.path('image')) as original:
        photo = original.convert('RGB')
    _, ref = recorded(capture)
    openings = opening_mask(ref['id_map'])
    parts = read_gzip(out / row['predictions'])
    filled = {}
    for m in names:
        occupied = np.zeros_like(openings, dtype=bool)
        for part in parts[m]:
            occupied |= footprint(part, openings.shape, photo.size, True)
        filled[m] = openings & occupied
        if int(filled[m].sum()) != row['scores'][m]['opening_filled']:
            raise ValueError('Opening overlay disagrees with measured coverage')
    if box is None:
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
    for m, label in names.items():
        n = int(filled[m].sum())
        put('opening-'+m, tint(filled[m], '#fc4381'), label, f'{n} opening samples covered in the view')
    counts = {m:int(mask.sum()) for m, mask in filled.items()}
    if earlier_zoom is not None:
        old_parts, expected_count = earlier_zoom
        old_filled = np.zeros_like(openings, dtype=bool)
        for part in old_parts:
            old_filled |= openings & footprint(part, openings.shape, photo.size, True)
        counts['previous_zoom'] = int(old_filled.sum())
        if counts['previous_zoom'] != expected_count:
            raise ValueError('Historical zoom opening count differs from the prior sheet')
        put('opening-previous_zoom', tint(old_filled, '#fc4381'), 'Previous zoom with additions',
            f'{expected_count} opening samples covered in the view')
    return dict(frame=row['key'], crop_xyxy=box, opening_samples=int(openings.sum()), filled_samples=counts)


def render(out, rows, captures, table, zoom):
    return render_zoom_review(out, rows, captures, table, zoom, NAMES,
                              'Can protected wall edges or closer views find more surfaces?',
                              BASE.parent/'plane-acceptance-01/sheet-views.json')


def render_crop_split(out, rows, captures, table):
    prior = BASE.parent / 'guard-zoom-01'
    sheet = checked(prior / 'sheet-views.json', read(prior/'manifest.json')['artifacts']['sheet-views.json'])
    opening_details = {view['layout']:view['close_up']['downstream']['openings'] for view in read(sheet)['views']}
    # These reference IDs select illustrations after prediction. The preceding review identified both targets.
    targets = {'shopfront': ('smooth_stone/dev-shopfront-plaster-low-left', 15),
               'workshop': ('light_gray_concrete/holdout-workshop-masonry-low-left', 1)}
    return render_zoom_review(out, rows, captures, table, prior/'candidate-runs/zoom',
                              {'current':'Main default', 'split':'Split-only zoom'},
                              'Can zoom separate walls while keeping openings clear?', sheet,
                              opening_details=opening_details, detail_targets=targets)


def render_contained(out, rows, captures, table):
    prior = BASE.parent/'guard-zoom-01'
    sheet = checked(prior/'sheet-views.json', read(prior/'manifest.json')['artifacts']['sheet-views.json'])
    openings = {v['layout']:v['close_up']['downstream']['openings'] for v in read(sheet)['views']}
    return render_zoom_review(out, rows, captures, table, prior/'candidate-runs/zoom',
                              {'current':'Split-only zoom', 'contained':'One-wall split'},
                              'Can one wall be split without cutting its neighbor?', sheet,
                              opening_details=openings, guard_inputs=prior,
                              detail_targets={'shopfront': ('light_gray_concrete/dev-shopfront-plaster-low-left', 13)})


def render_zoom_review(out, rows, captures, table, zoom, names, title, prior_sheet, *,
                       opening_details=None, detail_targets=None, guard_inputs=None):
    from benchmark.zoom_plane_fields import render_crop_proposals
    raw_stages = [('whole', 'Whole-image planes', 'Colored outlines show regions from the original Planarity pass.'),
                  ('crops', 'Closer-view planes', 'Outlines come from the same checkpoint on overlapping crops.')]
    if guard_inputs:
        guard_hashes = read(guard_inputs/'manifest.json')['artifacts']
        raw_stages.append(('guard', 'Proposed cuts', 'White outlines are earlier protected-edge proposals considered for a split.'))
    downstream = []
    for method, name in names.items():
        downstream.extend([
            ('sample-' + method, name, 'Straightened sample', 'The same selected guess, turned face-on. Gaps stay masked.', 200),
            ('count-' + method, name, 'Block Counter', 'Yellow marks measured grid lines. A question mark means unresolved.', 200)])
    opening_note = ('Same views and crops used to score open space in the previous zoom test.' if opening_details else
                    'For each building, use the view with the largest increase in covered opening samples.')
    downstream.extend([
        ('opening-source', 'Openings up close', 'Original', opening_note, 210),
        ('opening-truth', 'Open space', 'Scored reference', 'Gold marks background between or beneath building surfaces.', 210)])
    downstream.extend(('opening-'+m, title, 'Scored prediction', 'Pink marks predicted surface pixels inside a scored opening.', 210)
                      for m, title in names.items())
    historical_openings = opening_details and not guard_inputs
    if historical_openings:
        prior = prior_sheet.parent
        prior_artifacts = read(prior/'manifest.json')['artifacts']
        downstream.append(('opening-previous_zoom', 'Previous zoom', 'Archived comparison',
                           'This earlier method could add surfaces outside a wall. Pink marks open space it covered.', 210))
    def opening_rank(r):
        zoom_filled = r['scores']['zoom']['opening_filled']
        return zoom_filled-r['scores']['current']['opening_filled'], zoom_filled, r['scores']['zoom']['opening_pixels']
    if opening_details:
        by_key = {r['key']:r for r in rows}
        opening_views = {layout:by_key[rec['frame']] for layout, rec in opening_details.items()}
    else:
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
        sources = {'whole_image_fields_sha256': digest(field), 'crop_fields_sha256': digest(crop_field)}
        if guard_inputs:
            path = row['predictions']
            guard = read_gzip(checked(guard_inputs/path, guard_hashes[path]))['guard']
            candidates = [p for p in guard if p['id'].startswith('plane-consensus-')]
            img = photo.crop(box)
            for part in candidates:
                native = _native_alpha(part['support'], photo.size)
                if native:
                    alpha, bounds = native
                    registered = Image.new('L', img.size)
                    registered.paste(alpha, (bounds[0]-box[0], bounds[1]-box[1]))
                    edge = ImageChops.subtract(registered, registered.filter(ImageFilter.MinFilter(5)))
                    img.paste('white', mask=edge)
            accepted = [d for d in row['decisions'] if d['reason']=='accepted']
            note = f'{len(candidates)} proposed · {len(accepted)} accepted in this view'
            if focus is not None:
                target = next((d for d in row['reference_diagnostics'] if d['reference']==focus), {})
                for related in target.get('supporting_decisions', []):
                    decision = related['decision']
                    if related['relationship']=='direct protected-edge match' and decision['reason']=='unsupported-seam':
                        note = 'Edge unconfirmed'
                        if 'depth_agreement' in decision:
                            note += f" · {decision['depth_agreement']:.0%} depth agreement"
                        break
                if accepted:
                    n = sum(len(d['children']) for d in accepted)
                    note = f'{len(accepted)} accepted in view · {n} pieces returned'
            put(prefix+'guard', img, 'Protected-edge proposals', note)
            sources['guard_predictions_sha256'] = guard_hashes[path]
        return sources

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
        box = opening_details[row['layout']]['crop_xyxy'] if opening_details else None
        historical = None
        if historical_openings:
            path = opening['predictions']
            old_parts = read_gzip(checked(prior/path, prior_artifacts[path]))['zoom']
            historical = old_parts, opening_details[row['layout']]['filled_samples']['zoom']
        return dict(surfaces=out_status, openings=opening_panels(put, out, opening, captures[opening['key']],
                                                                 names, box, historical))

    return render_comparison(out, rows, captures, table, names, title,
        raw_stages=raw_stages, raw_panels=raw_panels, detail_stages=downstream, detail_renderer=extra_details,
        prior_detail_sheet=prior_sheet, detail_targets=detail_targets)
