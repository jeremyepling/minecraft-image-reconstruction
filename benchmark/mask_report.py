"""Write a self-contained, offline reading view for the frozen surface experiment."""
import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
from urllib.parse import quote


NAMES = {'current-b':'Current B', 'mask-first':'Model regions first',
         'scalelsd':'Regions + ScaleLSD v2', 'deeplsd':'Regions + DeepLSD'}
STYLE = """
:root{color-scheme:light;font:16px/1.45 "Segoe UI",Arial,sans-serif;color:#203332;background:#faf9f6}
body{max-width:82rem;margin:0 auto;padding:1.5rem}h1,h2,h3{line-height:1.2;color:#173737}
h2{margin-top:2rem}p{max-width:72rem}a{color:#075b70}nav{display:flex;flex-wrap:wrap;gap:.3rem 1rem}
details{margin:.6rem 0;border:1px solid #c3cfc9;background:white;padding:.6rem .85rem;border-radius:4px}
summary{cursor:pointer;font-weight:600}summary small{font-weight:400;margin-left:.5rem;color:#48605d}
.scroll{overflow-x:auto;margin:.75rem 0 1.5rem}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;line-height:1.3}
td,th{padding:.4rem .6rem;text-align:right;border-bottom:1px solid #d7dfdc}
th:first-child{text-align:left}thead th{background:#244844;color:white}
tbody th{font-weight:500}tr.group th{background:#edf2ee;color:#244844;font-weight:700}
tbody tr:nth-child(even):not(.group){background:#f7faf8}code{overflow-wrap:anywhere}
@media(max-width:48rem){body{padding:.6rem}td,th{padding:.3rem;font-size:.86rem}}
"""


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def number(value):
    return f'{value:,}'


def share(a,b):
    return f'{a:,} / {b:,} ({100*a/b:.1f}%)' if b else f'{a:,} / 0 (—)'


def distance(s, name):
    n=s[name+'_distance_count']
    return f"{s[name+'_distance_sum']/n:.1f} px" if n else '—'


def measures(s):
    yield 'Surfaces', [
        ('Reference surfaces found',share(s['matched'],s['eligible'])),
        ('Missed surfaces',number(s['eligible']-s['matched'])),
        ('Unmatched proposals / all proposals',share(s['proposed']-s['matched'],s['proposed']))]
    if 'unmatched_overlaps_small_surface' in s:
        yield 'What the unmatched proposals covered', [
            ('Matched a real surface below the scoring size',number(s['unmatched_overlaps_small_surface'])),
            ('Covered mostly background',number(s['unmatched_mostly_background'])),
            ('Covered a fragment or a mixture of surfaces',number(s['unmatched_fragment_or_mixed']))]
    yield 'Surface pixels', [
        ('Assigned to the correct surface',share(s['correct_pixels'],s['eligible_pixels'])),
        ('Assigned to the wrong surface',share(s['wrong_plane_pixels'],s['eligible_pixels'])),
        ('Left unclaimed',share(s['unclaimed_pixels'],s['eligible_pixels'])),
        ('Background samples covered',number(s['background_claimed']))]
    yield 'Edges and corners', [
        ('Mean boundary error on matched surfaces',distance(s,'edge')),
        ('Reference boundary samples found within 8 px',share(s['edge_true_near'],s['edge_true'])),
        ('Predicted boundary samples within 8 px',share(s['edge_proposed_near'],s['edge_proposed'])),
        ('Mean corner error on matched surfaces',distance(s,'corner')),
        ('Reference corners found within 12 px',share(s['corner_true_near'],s['corner_true'])),
        ('Predicted corners within 12 px',share(s['corner_proposed_near'],s['corner_proposed']))]
    yield 'Openings and joins', [
        ('Opening pixels covered',share(s['opening_filled'],s['opening_pixels'])),
        ('Gaps at least half filled',share(s['gaps_half_filled'],s['gaps'])),
        ('Touching outlines that recovered a real connection',share(s['correct_connections'],s['real_connections'])),
        ('Extra connections / all proposed connections',
         share(s['proposed_connections']-s['correct_connections'],s['proposed_connections']))]


def block_measures(entry):
    score=entry['score']
    diag=entry['diagnostic']
    yield 'Blocks written to the datapack · image-only prediction', [
        ('Correct coordinates / visible reference blocks',share(score['correct_visible'],score['visible_truth'])),
        ('Predicted coordinates written',number(score['serialized'])),
        ('Extra coordinates',number(score['extra'])), ('Missed coordinates',number(score['missed']))]
    yield 'Blocked steps · number of surface regions', [
        ('No matching planar chart',number(entry['regions'].get('unresolved_chart',0))),
        ('Camera unavailable',number(entry['regions'].get('unresolved_camera',0))),
        ('Unknown block scale',number(entry['regions'].get('unresolved_scale',0))),
        ('Unknown plane depth',number(entry['regions'].get('unresolved_depth',0))),
        ('No fully supported block face',number(entry['regions'].get('no_fully_supported_cell',0)))]
    yield 'Diagnostic · given the recorded camera and surface plane', [
        ('Correct coordinates / visible reference blocks',share(diag['correct_visible'],diag['visible_truth'])),
        ('Predicted coordinates written',number(diag['serialized'])),
        ('Extra coordinates',number(diag['extra'])), ('Missed coordinates',number(diag['missed']))]


def table(columns, entries, metric):
    sections=[list(metric(entry)) for entry in entries]
    content=['<div class="scroll"><table><thead><tr><th scope="col">Measure</th>']
    content.extend('<th scope="col">'+escape(name)+'</th>' for name in columns)
    content.append('</tr></thead><tbody>')
    for j,(section, rows) in enumerate(sections[0]):
        content.append(f'<tr class="group"><th colspan="{len(columns)+1}">{escape(section)}</th></tr>')
        for i,(label,_) in enumerate(rows):
            content.append('<tr><th scope="row">'+escape(label)+'</th>')
            content.extend('<td>'+escape(s[j][1][i][1])+'</td>' for s in sections)
            content.append('</tr>')
    content.append('</tbody></table></div>')
    return ''.join(content)


def score_table(scores):
    return table([NAMES[m] for m in scores],scores.values(),measures)


def block_table(blocks, selected):
    text=table(['Current B',NAMES[selected]+' (selected)'],[blocks['baseline'],blocks['selected']],block_measures)
    if 'selected_with_B_charts' in blocks:
        control=blocks['selected_with_B_charts']['score']
        text+=('<p><strong>Fixed depth and scale check.</strong> When the selected method used B’s planar charts, '
               f"{control['correct_visible']:,} of {control['visible_truth']:,} visible coordinates were correct, "
               f"with {control['extra']:,} extra coordinates across {control['serialized']:,} written.</p>")
    return text


def link(href,title):
    return f'<a href="{escape(quote(str(href).replace(chr(92),"/"),safe="/:.-_#"))}">{escape(title)}</a>'


def core_summary(scores):
    return ' · '.join(f'{NAMES[m]} {s["matched"]}/{s["eligible"]}' for m,s in scores.items())


def headline(scores):
    base,mask=scores['current-b'],scores['mask-first']
    lines=(f"<p>Region masks recovered <strong>{mask['matched']:,} of {mask['eligible']:,}</strong> surfaces. "
           f"B found <strong>{base['matched']:,}</strong>. The masks covered "
           f"{share(mask['opening_filled'],mask['opening_pixels'])} of opening samples, compared with "
           f"{share(base['opening_filled'],base['opening_pixels'])} for B.</p>")
    if 'scalelsd' in scores and 'deeplsd' in scores:
        lines+=(f"<p>Adding ScaleLSD found {scores['scalelsd']['matched']:,} surfaces. "
                f"Adding DeepLSD found {scores['deeplsd']['matched']:,}. The table shows what each gained or lost in boundary accuracy.</p>")
    return lines


def timing(row):
    names={'model_fields_cached':'Read shared numerical model outputs', 'model_fields_fresh':'Generate missing shared model outputs',
           'normal_timed_separately':'MoGe-2 direction inference on separate timed pass',
           'depth_timed_separately':'Depth Anything V2 inference on separate timed pass',
           'current-b':'Current B processing', 'mask-first':'Model regions first processing',
           'scalelsd_refine':'ScaleLSD refinement from cached lines','deeplsd_refine':'DeepLSD refinement from cached lines',
           'scalelsd_inference_recorded':'ScaleLSD inference on its original run',
           'deeplsd_inference_recorded':'DeepLSD inference on its original run'}
    cells=''.join(f'<tr><th scope="row">{label}</th><td>{row[key]:.3f} s</td></tr>' for key,label in names.items() if key in row)
    return '<table><tbody>'+cells+'</tbody></table>'


def render_report(out):
    out=Path(out)
    rows=load(out/'cases.json')
    scores=load(out/'scores.json')
    selected=load(out/'selection.json')['selected']
    blocks=load(out/'blocks.json') if (out/'blocks.json').is_file() else {}
    timing_file=Path(__file__).resolve().parent/'data/mask-field-timing-v1.json'
    original_times=load(timing_file)['per_image'] if timing_file.is_file() else {}
    for row in rows:
        for kind,record in original_times.get(row['sha256'],{}).items():
            row['timing_seconds'][kind+'_timed_separately']=record['seconds']
    buildings=defaultdict(list)
    for row in rows:
        buildings[row['layout']].append(row)
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           '<title>Surface recovery by picture</title><style>'+STYLE+'</style><main id="top">',
           '<h1>Which visible surfaces did each method find?</h1>',
           f'<p>{len(rows)} saved views of {len(buildings)} Minecraft buildings. The development comparison selected <strong>{escape(NAMES[selected])}</strong>.</p>',
           '<nav aria-label="Run files">'+ ' '.join(link(p,t) for p,t in [
               ('comparison.jpg','Comparison sheet'),('manifest.json','Run manifest'),('cases.json','Per-picture data'),
               ('scores.json','Aggregate data'),('selection.json','Selection rule')])+'</nav>',
           '<h2>All pictures</h2><p>Counts here add up across pictures. A wall seen in four views contributes once in each view. '
           'A surface is a connected flat patch. A proposal counts as a match when its intersection covers at least half of its union with that patch. '
           'Each proposal and each patch can be matched once. The scored surfaces contain at least four visible block faces and 24 scoring samples.</p>',
           headline(scores['all']),score_table(scores['all'])]
    if blocks:
        parts+=['<nav>'+link('loss-trace/stages.jpg','Six examples of where B lost surfaces')+
                ' '+link('recovered-by-stage.json','Recoveries by B failure stage')+'</nav>',
                '<h3>Did that improve the blocks?</h3><p>The exporter wrote smooth-stone blocks for this geometry check and the scorer read their '
                'coordinates back from the pack. The diagnostic supplies the correct camera and surface plane. In-game placement was outside this replay.</p>',
                block_table(blocks['all'],selected),'<nav>'+link('blocks.json','Block scores')+
                ' '+link('block-packs.zip','Exported datapacks')+'</nav>']
    parts+=['<p>Pixel counts use the scoring grid, sampled every four image pixels in each direction. Distances use original screenshot pixels. '
            'Mean edge and corner errors cover matched surfaces with measurable boundaries. Their coverage rows also count the missed surfaces. '
            'An opening sample is background with building visible above and to both sides. '
            'The connections check asks whether regions that touch in the picture correspond to recorded planes meeting in 3D. '
            'A dash means there was no denominator.</p>']
    total_times=defaultdict(float)
    seen=set()
    for row in rows:
        for k,v in row['timing_seconds'].items():
            if not k.endswith(('_inference_recorded','_timed_separately')) or row['sha256'] not in seen:
                total_times[k]+=v
        seen.add(row['sha256'])
    parts+=['<details><summary>Runtime across these pictures</summary><p>Processing time includes each replayed picture. '
            'Separately measured model inference counts each distinct screenshot hash once.</p>',timing(total_times),'</details>']
    for key,label in [('development','Development buildings'),('previously_reviewed_regression','Previously reviewed comparison buildings')]:
        if key in scores:
            parts += ['<details><summary>'+label+'</summary>',score_table(scores[key]),'</details>']
    parts.append('<h2 id="pictures">Scores by picture</h2><nav aria-label="Buildings">')
    for layout, cases in buildings.items():
        parts.append(link('#'+layout,f'{layout.title()} ({len(cases)})'))
    parts.append('</nav>')
    for layout,cases in buildings.items():
        group='Development' if cases[0]['cohort']=='development' else 'Reviewed regression'
        parts.append(f'<section id="{escape(layout)}"><h3>{escape(layout.title())} · {group}</h3>')
        if layout in scores.get('layouts',{}):
            parts+=['<details><summary>All views of this building</summary>',score_table(scores['layouts'][layout]),'</details>']
        for row in cases:
            parts+=['<details><summary>'+escape(row['id'])+' · '+escape(row['floor'].replace('_',' '))+
                    '<small>'+escape(core_summary(row['scores']))+'</small></summary>',
                    '<p>'+link(row['source'],'Original screenshot')+' · '+link(row['predictions'],'Saved predictions')+'</p>',
                    score_table(row['scores'])]
            if row.get('blocks'):
                parts.append(block_table(row['blocks'],selected))
            parts+=['<details><summary>Runtime for this picture</summary>',timing(row['timing_seconds']),'</details></details>']
        parts+=['<p>'+link('#pictures','Back to building list')+'</p></section>']
    parts.append('</main></html>')
    (out/'report.html').write_text('\n'.join(parts),encoding='utf8')
    return 'report.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    print(render_report(parser.parse_args().out))
