"""Write the same-coordinate crop and exact native mask review."""

from html import escape

import numpy as np
from PIL import Image

from benchmark.sky_crop_reference import ratio


STYLE = """
:root { color-scheme:dark; font:16px/1.45 system-ui,sans-serif; color:#e5e9ef; background:#141920 }
body { max-width: 1450px; margin:auto; padding:24px }
h1,h2,h3 { line-height:1.2 }
a { color:#97cdfb }
section { margin: 16px 0; border:1px solid #3a4554; border-radius:8px; padding:16px }
dl { display:flex; flex-wrap:wrap; gap: 4px 24px }
dt { font-weight:bold } dd { margin:0 }
img { display:block; max-width:100%; object-fit:contain; border-radius:3px }
.source { max-width:min(100%,900px) }
.panels { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:12px }
figure { margin:0; min-width:0 }
figure img { width:100%; background:#303843; image-rendering:pixelated }
figcaption { margin-top:4px; color:#b1bfce; font-size:.9rem }
table { border-collapse:collapse; font-variant-numeric:tabular-nums }
th,td { text-align:right; padding:6px 12px; border-bottom:1px solid #3a4554 }
th:first-child,td:first-child { text-align:left }
.stats { color:#b1bfce; font-variant-numeric:tabular-nums }
.done { color:#b7eb73 } .extra { color:#fc916c } .missed { color:#83dbff }
details { margin-top:14px }
@media (max-width:800px) { .panels { grid-template-columns:repeat(2, minmax(0, 1fr)) } }
"""


def page(title, contents):
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{escape(title)}</title><style>{STYLE}</style><body>'
        f'{contents}</body></html>\n'
    )


def percent(value):
    return f'{value:.1%}'


def metric_cells(summary):
    return (
        f'<td>{summary["active_masks"]}</td><td>{percent(summary["macro_iou"])}</td>'
        f'<td>{percent(summary["pixel_weighted_iou"])}</td><td>{summary["masks_iou_at_least_50"]}</td>'
        f'<td>{summary["intersection"]:,}</td><td>{summary["extra"]:,}</td>'
        f'<td>{percent(summary["full_cube_coverage"])}</td><td>{summary["targets_without_primary"]}</td>'
        f'<td>{summary["targets_without_pixels"]}</td><td>{summary["union_background_pixels"]:,}</td>'
    )


TABLE_HEAD = """
<thead><tr><th scope="col">Run</th><th scope="col">Masks</th><th scope="col">Mean IoU</th>
<th scope="col">Pixel-pooled IoU</th><th scope="col">Masks ≥ .50 IoU</th>
<th scope="col">Paired overlap px</th><th scope="col">Paired extra px</th>
<th scope="col">Cube pixels touched</th><th scope="col">Targets without a primary mask</th>
<th scope="col">Targets without overlap</th><th scope="col">Unique background px</th></tr></thead>
"""


def metric_table(rows):
    return f'<table>{TABLE_HEAD}<tbody>{"".join(rows)}</tbody></table>'


def write_index(output, report):
    baseline = report['pooled']['baseline']
    rows = [
        f'<tr><td><a href="{escape(case["id"])}/index.html">{escape(case["id"])}</a> '
        f'({escape(case["split"])})</td>{metric_cells(case["arms"]["baseline"]["summary"])}</tr>'
        for case in report['cases']
    ]
    rows.append(f'<tr><th scope="row">All buildings</th>{metric_cells(baseline)}</tr>')
    candidate_rows = [
        f'<tr><th scope="row">{escape(name)}</th>{metric_cells(summary)}</tr>'
        for name, summary in report['pooled'].items()
    ]
    comparison = ''
    if len(candidate_rows) > 1:
        comparison = f'<h2>Alternative masks on the same proposals</h2>{metric_table(candidate_rows)}'
    contents = f"""
<h1>Sky surface crop review</h1>
<p>{baseline['active_masks']} active Surface Finder masks on {len(report['cases'])} saved screenshots.
Each crop is paired with the highest-IoU connected visible full-cube surface it touches. All candidate
arms keep that pairing. Target coverage also counts visible cube surfaces without a primary mask.</p>
{metric_table(rows)}
{comparison}
<p>Mean IoU weights proposals equally. Pixel-pooled IoU adds the intersections and unions of the individual
pairs. Cube pixels touched uses the union of all masks, even when a proposal touches several surfaces.
Partial blocks and decorative model faces have known native pixels and appear separately in the raw scores.</p>
<p><a href="report.json">Scoring records and input hashes</a></p>
"""
    (output / 'index.html').write_text(page('Sky surface crop review', contents), encoding='utf-8')


def padded_box(mask, size, margin=8):
    left, top, right, bottom = Image.fromarray(mask).getbbox()
    width, height = size
    return max(left - margin, 0), max(top - margin, 0), min(right + margin, width), min(bottom + margin, height)


def figure(path, caption):
    return (
        f'<figure><a href="{escape(path)}"><img src="{escape(path)}" '
        f'alt="{escape(caption)}" loading="lazy"></a><figcaption>{escape(caption)}</figcaption></figure>'
    )


def copy_crop(image, mask, filename):
    crop = image.convert('RGBA')
    crop.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
    crop.save(filename)


def write_panels(output, source, handoff, part, label, predicted, target):
    box = padded_box(predicted | target, source.size)
    left, top, right, bottom = box
    region = np.s_[top:bottom, left:right]
    local = source.crop(box)
    guess, exact = predicted[region], target[region]
    local.save(output / f'{label}-source.png')
    actual_path = part['source_crop']['path']
    actual_box = part['source_crop']['xyxy']
    with Image.open(handoff / actual_path) as actual:
        actual_panel = Image.new('RGBA', local.size)
        actual_panel.paste(actual, (actual_box[0] - left, actual_box[1] - top))
    actual_panel.save(output / f'{label}-actual.png')
    copy_crop(local, exact, output / f'{label}-exact.png')

    overlay = np.asarray(local).copy()
    for region_mask, color in (
        (guess & exact, (183, 235, 115)),
        (guess & ~exact, (252, 113, 73)),
        (~guess & exact, (93, 204, 250)),
    ):
        overlay[region_mask] = (.36 * overlay[region_mask] + .64 * np.array(color)).astype(np.uint8)
    Image.fromarray(overlay).save(output / f'{label}-overlap.png')

    names = (
        ('source', 'Source image detail'),
        ('actual', 'Saved Surface Finder crop'),
        ('exact', 'Exact frontmost-pixel crop'),
        ('overlap', 'Native pixel overlay'),
    )
    return ''.join(figure(f'{label}-{name}.png', caption) for name, caption in names)


def write_unmatched(output, source, reference, targets):
    missing = [target for target in targets if target['primary_predictions'] == 0]
    items = []
    by_id = {target['id']: i for i, target in enumerate(reference.targets, 1)}
    for target in sorted(missing, key=lambda target: (-target['pixels'], target['id'])):
        mask = reference.labels == by_id[target['id']]
        box = padded_box(mask, source.size)
        left, top, right, bottom = box
        name = f'{target["id"]}-unmatched.png'
        copy_crop(source.crop(box), mask[top:bottom, left:right], output / name)
        coverage = percent(ratio(target['covered_pixels'], target['pixels']))
        items.append(figure(name, f'{target["id"]} · {target["direction"]} · {target["pixels"]:,} px · {coverage} touched'))
    return (
        f'<details id="unmatched"><summary>{len(missing)} reference surfaces without a primary crop</summary>'
        f'<div class="panels">{"".join(items)}</div></details>'
    )


def write_case(output, report, source, handoff, reference, visuals):
    ident = report['id']
    folder = output / ident
    folder.mkdir(parents=True, exist_ok=True)
    source.save(folder / 'source.png')
    baseline = report['arms']['baseline']
    items = []
    for (label, part, predicted, target_index), row in zip(visuals, baseline['proposals'], strict=True):
        target = reference.labels == target_index if target_index else np.zeros_like(predicted)
        target_name = 'No full-cube target'
        if target_index:
            record = reference.targets[target_index - 1]
            direction = escape(record['direction'])
            target_name = f'{record["id"]} · {direction} · plane {record["plane_offset"]}'
        panels = write_panels(folder, source, handoff, part, label, predicted, target)
        items.append(f"""
<section id="{label}">
<h3>{label} · {escape(row['id'])} · {target_name}</h3>
<p class="stats">IoU {percent(row['iou'])} · Precision {percent(row['precision'])} ·
Recall {percent(row['recall'])} · {row['intersection']:,} matched px ·
{row['missed']:,} missed px · {row['extra']:,} extra px
({row['other_cube_pixels']:,} other cube, {row['non_cube_pixels']:,} decorative,
{row['background_pixels']:,} background)</p>
<div class="panels">{panels}</div>
</section>
""")
    links = ' '.join(f'<a href="#{row["label"]}">{row["label"]}</a>' for row in baseline['proposals'])
    unmatched = write_unmatched(folder, source, reference, baseline['targets'])
    title = ident
    rows = [f'<tr><th scope="row">{escape(name)}</th>{metric_cells(arm["summary"])}</tr>'
            for name, arm in report['arms'].items()]
    contents = f"""
<p><a href="../index.html">All buildings</a></p><h1>{escape(title)}</h1>
<a href="source.png"><img class="source" src="source.png" alt="Full frozen Sky source image"></a>
{metric_table(rows)}
{unmatched}
<h2>Active crops</h2>
<nav aria-label="Crops">{links}</nav>
<p><span class="done">Green overlap</span> · <span class="extra">Orange crop only</span> ·
<span class="missed">Blue target only</span>. Each row uses the same native source window in all four panels.</p>
{"".join(items)}
"""
    (folder / 'index.html').write_text(page(title, contents), encoding='utf-8')
