"""Render fixed-coordinate crop sequences and the native pixel edits between passes."""

from html import escape

import numpy as np
from PIL import Image, ImageDraw

from benchmark.sky_crop_reference import native_mask
from benchmark.sky_crop_reference_review import copy_crop, figure, padded_box, page, percent


TRAJECTORY_STYLE = """
.timeline { display:flex; flex-wrap:wrap; align-items:flex-start; gap:14px }
.timeline figure { flex:1 1 0; max-width:340px; min-width:120px }
.timeline-note { flex: 1 1 250px; max-width:380px; margin:0 }
.scroll { overflow-x:auto; margin:12px 0 }
.exact-add { color:#bef56e } .extra-add { color:#f4a2f7 }
.exact-remove { color:#fb895f } .extra-remove { color:#82dcff }
"""


def review_page(title, contents):
    return page(title, contents).replace('</style>', TRAJECTORY_STYLE + '</style>', 1)


def score_line(row):
    return (
        f'{percent(row["iou"])} IoU · {row["missed"]:,} missed px · '
        f'{row["extra"]:,} extra px'
    )


def table(rows):
    return f"""
<div class="scroll"><table><thead><tr><th scope="col">Pass</th><th scope="col">Crops</th>
<th scope="col">Mean IoU</th><th scope="col">Mean IoU for matched crops</th>
<th scope="col">Pixel-pooled IoU</th><th scope="col">IoU ≥ 95%</th>
<th scope="col">Edited</th><th scope="col">Improved</th><th scope="col">Regressed</th>
<th scope="col">Target px added</th><th scope="col">Off-target px added</th>
<th scope="col">Target px removed</th><th scope="col">Off-target px removed</th>
<th scope="col">Cube pixels touched</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
"""


def row_cells(summary, token):
    columns = [
        summary['active_masks'], percent(summary['macro_iou']),
        percent(summary['matched_macro_iou']), percent(summary['pixel_weighted_iou']),
        summary['masks_iou_at_least_95'],
        summary.get('changed_this_round', '—'), summary.get('improved_this_round', '—'),
        summary.get('regressed_this_round', '—'),
        summary.get('added_target', '—'), summary.get('added_off_target', '—'),
        summary.get('removed_target', '—'), summary.get('removed_off_target', '—'),
        percent(summary['full_cube_coverage']),
    ]
    numbers = ''.join(f'<td>{value:,}</td>' if isinstance(value, int) else f'<td>{value}</td>'
                      for value in columns)
    return f'<tr><th scope="row">{escape(pass_name(token))}</th>{numbers}</tr>'


def pass_name(token):
    if token == '0':
        return 'Initial crop'
    if token.isdigit():
        return f'Pass {token}'
    return 'Training-selected stop' if token == 'selected' else 'Producer final'


def write_index(output, report):
    items = [
        f'<li><a href="{escape(case["id"])}/index.html">{escape(case["id"])}</a> ({escape(case["split"])})</li>'
        for case in report['cases']
    ]
    blocks = []
    for name, groups in report['trajectories'].items():
        panels = []
        for group in ('all', 'training', 'held-out', 'validation', 'test'):
            if group in groups:
                scores = groups[group]
                rows = [row_cells(metrics, token) for token, metrics in scores.items()]
                panels.append(f'<h3>{escape(group.title())}</h3>{table(rows)}')
        last_token = next(token for token in reversed(groups['all']) if token.isdigit())
        failures = ''.join(f'<li>{escape(category)} · {count}</li>'
                           for category, count in groups['all'][last_token]['failures_by_category'].items())
        blocks.append(f'<section><h2>{escape(name)}</h2>{"".join(panels)}'
                      f'<h3>Remaining failures after pass {last_token}</h3><ul>{failures}</ul></section>')
    html = f"""
<h1>Sky crop correction passes</h1>
<p>Successive masks use the same native source pixels and keep the saved Surface Finder crop’s reference surface.
Scores and pixel edits are recomputed from the saved masks. Prompt or policy receipts describe why
the producer made an attempt. SAM reference feedback uses the exact surface to locate errors and select
edits. The source-only policy receives the screenshot, geometry fields and its current crop. The exact
mask appears with each example for comparison. The chooser-initialized condition starts from its
selected mask. Its row at pass 0 measures the difference from the saved Surface Finder crop.</p>
<h2>Building sequences</h2>
<ul>{"".join(items)}</ul>
<p>Training includes Pillars and Viaduct. Held-out combines the two validation and two test buildings.
The 95% threshold is crop IoU. “Cube pixels touched” counts all visible cube pixels under at least
one crop, including other planes.</p>
{"".join(blocks)}
<p><a href="report.json">All proposal scores, edit receipts and input hashes</a></p>
"""
    (output / 'index.html').write_text(review_page('Sky crop correction passes', html), encoding='utf-8')


def transition_image(local, before, after, target, path, receipts):
    added, removed = after & ~before, before & ~after
    overlay = np.asarray(local).copy()
    for area, color in (
        (added & target, (188, 248, 77)),
        (added & ~target, (244, 105, 245)),
        (removed & target, (250, 99, 50)),
        (removed & ~target, (67, 196, 249)),
    ):
        overlay[area] = (.28 * overlay[area] + .72 * np.asarray(color)).astype(np.uint8)
    result = Image.fromarray(overlay)
    draw = ImageDraw.Draw(result)
    for point in receipts:
        if all(key in point for key in ('x', 'y', 'label')):
            x, y = point['x'], point['y']
            color = '#bef56e' if point['label'] == 1 else '#82dcff'
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline='black', width=3)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=1)
    result.save(path)


def short_report(row):
    bits = [
        f'<span class="exact-add">+{row["added_target"]:,} target</span>',
        f'<span class="extra-add">+{row["added_off_target"]:,} off-target</span>',
        f'<span class="exact-remove">−{row["removed_target"]:,} target</span>',
        f'<span class="extra-remove">−{row["removed_off_target"]:,} off-target</span>',
    ]
    text = f'{score_line(row)} ({row["delta_iou"]:+.1%})<br>{" · ".join(bits)}'
    return text


def unchanged_passes(first, last):
    span = f'pass {first}' if first == last else f'passes {first} through {last}'
    return f'<p>Crop unchanged during {span}.</p>'


def attempt_note(receipts):
    lines = []
    for record in receipts:
        if 'diagnosis' in record:
            verb = 'Accepted' if record.get('accepted') else 'Kept previous crop'
            operation = 'positive point' if record.get('label') == 1 else 'negative point'
            diagnosis = str(record['diagnosis']).replace('_', ' ').replace('-', ' ')
            pos = f' ({record["x"]}, {record["y"]})' if 'x' in record and 'y' in record else ''
            lines.append(f'{verb} after {operation}{pos} · {escape(diagnosis)}')
        elif 'added' in record and 'removed' in record:
            lines.append(f'Policy recorded {record["added"]:,} px added and {record["removed"]:,} px removed.')
    return '<br>'.join(lines)


def examples(arms, tokens):
    first, last = [arms[token]['proposals'] for token in (tokens[0], tokens[-1])]
    pairs = list(zip(first, last, strict=True))
    eligible = [index for index, (before, _) in enumerate(pairs) if before['target'] is not None]
    if not eligible:
        return []
    gain = lambda index: last[index]['iou'] - first[index]['iou']
    edited = lambda index: sum(arms[token]['proposals'][index]['changed_this_round'] for token in tokens[1:])
    best = max(eligible, key=lambda index: (edited(index) > 1, gain(index), edited(index)))
    labels = ['Improvement' if gain(best) > 0 else 'Most active crop']
    selected = [best]
    unfinished = [index for index in eligible if index != best and last[index]['iou'] < .95]
    if unfinished:
        fail = max(unfinished, key=lambda index: (gain(index) < 0, edited(index) > 0,
                                                  last[index]['missed'] + last[index]['extra']))
        selected.append(fail)
        labels.append('Regression' if gain(fail) < 0 else 'Remaining boundary error')
    return list(zip(selected, labels, strict=True))


def write_example(folder, image, reference, visual_row, arms, trajectory, tokens, index, tag):
    label, part, initial, paired = visual_row
    ident = folder.name
    prefix = f'{trajectory["index"]}-{label}'
    target = reference.labels == paired if paired else np.zeros_like(initial)
    masks = []
    for token in tokens:
        support = trajectory['packets'][token]['frames'].get(ident, {}).get(part['id'])
        mask = initial if support is None else native_mask(support, image.size)
        masks.append(mask)
    union = target | initial
    for mask in masks:
        union |= mask
    left, top, right, bottom = padded_box(union, image.size)
    region = np.s_[top:bottom, left:right]
    local = image.crop((left, top, right, bottom))
    truth = target[region]
    local.save(folder / f'{prefix}-source.png')
    copy_crop(local, truth, folder / f'{prefix}-exact.png')
    copy_crop(local, masks[0][region], folder / f'{prefix}-initial.png')
    first = arms[tokens[0]]['proposals'][index]
    last = arms[tokens[-1]]['proposals'][index]
    related = [
        figure(f'{prefix}-source.png', 'Source image detail'),
        figure(f'{prefix}-initial.png', f'Initial crop · {score_line(first)}'),
        figure(f'{prefix}-exact.png', f'Reference · {first["target"]} · {first["target_pixels"]:,} px'),
    ]
    if not np.array_equal(initial, masks[0]):
        copy_crop(local, initial[region], folder / f'{prefix}-ordinary.png')
        related.insert(1, figure(f'{prefix}-ordinary.png', 'Saved Surface Finder crop'))

    updates = []
    hold_start = None
    for counter, token in enumerate(tokens[1:], 1):
        row = arms[token]['proposals'][index]
        receipts = row['edit_receipts']
        if not row['changed_this_round'] and not receipts:
            if hold_start is None:
                hold_start = token
            continue
        if hold_start is not None:
            updates.append(unchanged_passes(hold_start, tokens[counter - 1]))
            hold_start = None
        mask = masks[counter][region]
        previous = masks[counter - 1][region]
        details = attempt_note(receipts)
        before_error = arms[tokens[counter - 1]]['proposals'][index]['failure_category']
        if before_error:
            details = f'Previous main error · {escape(before_error)}.<br>{details}'
        local_receipts = [dict(point, x=point['x'] - left, y=point['y'] - top)
                          for point in receipts if 'x' in point and 'y' in point]
        transition_image(local, previous, mask, truth, folder / f'{prefix}-{token}-edit.png', local_receipts)
        copy_crop(local, mask, folder / f'{prefix}-{token}-crop.png')
        updates.append(f"""
<h4>{escape(pass_name(token))}{" · crop unchanged" if not row['changed_this_round'] else ""}</h4>
<div class="timeline">
{figure(f'{prefix}-{token}-crop.png', 'Crop after this pass')}
{figure(f'{prefix}-{token}-edit.png', 'Pixel edits and prompt point if supplied')}
<p class="timeline-note">{short_report(row)}<br>{details}</p>
</div>
""")
    if hold_start is not None:
        updates.append(unchanged_passes(hold_start, tokens[-1]))
    final_reason = last['failure_category'] or 'Reached at least 95% IoU'
    return f"""
<section id="{prefix}">
<h3>{escape(label)} · {escape(part["id"])} · {escape(tag)}</h3>
<p>{percent(first["iou"])} → {percent(last["iou"])} IoU. Final result · {escape(final_reason)}.</p>
<div class="timeline">{"".join(related)}</div>
{"".join(updates)}
</section>
"""


def write_case(output, case, visual_data, trajectories):
    image, _, reference, visual_rows = visual_data
    ident = case['id']
    folder = output / ident
    folder.mkdir(parents=True, exist_ok=True)
    image.save(folder / 'source.png')
    sections = []
    for index, trajectory in enumerate(trajectories):
        arms = {token: case['arms'][f'{index}#{token}'] for token in trajectory['packets']}
        tokens = [token for token in arms if token.isdigit()]
        sections.append(f'<h2>{escape(trajectory["name"])}</h2>')
        rows = [row_cells(arm['summary'], token) for token, arm in arms.items()]
        sections.append(table(rows))
        for selected, tag in examples(arms, tokens):
            sections.append(write_example(folder, image, reference, visual_rows[selected],
                                          arms, dict(trajectory, index=index), tokens, selected, tag))
    title = f'{ident} · crop corrections'
    content = f"""
<p><a href="../index.html">All buildings</a></p><h1>{escape(title)}</h1>
<p><a href="source.png">Full native source image</a></p>
<p>Edits are measured against the preceding crop.
<span class="exact-add">Green adds target pixels</span>,
<span class="extra-add">pink adds off-target pixels</span>,
<span class="exact-remove">orange removes target pixels</span>,
and <span class="extra-remove">blue removes off-target pixels</span>.</p>
{"".join(sections)}
"""
    (folder / 'index.html').write_text(review_page(title, content), encoding='utf-8')
