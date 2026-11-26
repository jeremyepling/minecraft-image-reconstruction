"""Index the current Surface Finder's misses using archived predictions and recorded geometry."""
from collections import Counter
from itertools import groupby

from benchmark.mask_metrics import measure
from benchmark.recovery_summary import remaining, verified
from benchmark.remaining_surface_audit import kind
from benchmark.single_wall_sheet import load_gzip
from benchmark.surface_reference import recorded


METHOD = 'single-v2'
DIAGNOSTICS = ('merged', 'parallel_step', 'in_raw_sam', 'in_filtered_sam')
ORIENTATIONS = ('vertical', 'top', 'underside')


def _id_list(ids):
    return ', '.join(f'R{rid}' for rid in ids) or '—'


def _picture_table(frames, totals):
    lines = [
        '# Missed surfaces by screenshot', '',
        f"The Surface Finder matched {totals['found']} of {totals['eligible']} scored surfaces "
        f"across {len(frames)} screenshots. {totals['missed']} did not match.", '',
        'An R number identifies one reference surface in that picture. A match requires at least '
        '50% overlap, measured as the intersection divided by the union. Each prediction can match one surface.', '',
        'A merge means one prediction covered at least half of two scored surfaces. A depth step is '
        'a merge between surfaces facing the same direction at different depths. The last two columns '
        'show where a saved MobileSAM outline could match the missed surface before and after filtering. '
        'These checks can overlap.', '',
    ]
    for layout, pictures in groupby(frames, key=lambda frame: frame['layout']):
        lines.extend([
            f'## {layout.title()}', '',
            '| Screenshot | Found / scored | Missed surfaces | Merged | Depth step | Raw MobileSAM | After filter |',
            '| --- | ---: | --- | --- | --- | --- | --- |',
        ])
        for frame in pictures:
            missed = ', '.join(f"R{rid} ({frame['missed_orientation'][rid]})" for rid in frame['missed'])
            cells = [f"`{frame['key']}`", f"{frame['found']} / {frame['eligible']}", missed or 'None']
            cells.extend(_id_list(frame[key]) for key in DIAGNOSTICS)
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')
    return '\n'.join(lines)


def build_index(rows, cases, source, hashes, mobile_rows):
    """Return per-picture records, pooled counts and a Markdown index. Reference data is for scoring."""
    ordered = sorted(rows, key=lambda row: (row['layout'], row['key']))
    keys = {row['key'] for row in ordered}
    if len(keys) != len(ordered) or keys != cases.keys() or keys != mobile_rows.keys():
        raise ValueError('Screenshot IDs differ between the Surface Finder, captures and MobileSAM')

    frames = []
    counts = Counter()
    orientation_counts = Counter()
    for row in ordered:
        key = row['key']
        case, mobile = cases[key], mobile_rows[key]
        if mobile['sha256'] != row['sha256'] or mobile['layout'] != row['layout']:
            raise ValueError(f'MobileSAM used a different input for {key}')
        verified(case.path('image'), row['sha256'])
        parts = load_gzip(source, row['predictions'], hashes)[METHOD]
        _, ref = recorded(case)
        score, pairs = measure(parts, ref, use_mask=True)
        matched = sorted(rid for _, rid in pairs)
        saved = row['scores'][METHOD]
        if (score['eligible'] != saved['eligible'] or len(matched) != saved['matched']
                or matched != sorted(row['scopes'][METHOD]['matched'])):
            raise ValueError(f'Regraded surface IDs or counts differ from the saved scores for {key}')
        sam = {name: set(map(int, mobile[name])) for name in ('selected_matches', 'pool_matches')}
        detail = remaining(case, parts, matched, sam)
        missed_orientation = {rid: kind(ref['patches'][rid]) for rid in detail['missed']}
        orientations = Counter(missed_orientation.values())
        frames.append(dict(key=key, sha256=row['sha256'], layout=row['layout'], eligible=score['eligible'],
                           found=len(matched), matched_ids=matched, missed_orientation=missed_orientation,
                           orientation={name: orientations[name] for name in ORIENTATIONS}, **detail))
        counts.update(eligible=score['eligible'], found=len(matched), missed=len(detail['missed']))
        counts.update({name: len(detail[name]) for name in DIAGNOSTICS})
        orientation_counts.update(orientations)
    totals = dict(counts, orientation={name: orientation_counts[name] for name in ORIENTATIONS})
    return frames, totals, _picture_table(frames, totals)
