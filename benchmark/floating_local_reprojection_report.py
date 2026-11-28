"""Archive the decoded scores and draw frontmost reference-only placement deltas."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from benchmark.expanded_baseline import read, save


ARMS = ('incumbent', 'depth_anything', 'moge', 'reprojection', 'combined', 'reference_camera', 'oracle')


def group_scores(rows, arm):
    result = dict(views=len(rows), switched=sum(row['choices'][arm] != 'incumbent' for row in rows))
    changes = Counter()
    for row in rows:
        changes.update(row['options'][row['choices'][arm]]['paired'])
    result['paired'] = dict(changes)
    for metric in ('geometry', 'scores'):
        for regime in ('strict', 'facade'):
            selected = [row['options'][row['choices'][arm]]['score'][metric][regime] for row in rows]
            counts = {key: sum(score[key] for score in selected) for key in ('correct', 'missed', 'extras')}
            n = 2 * counts['correct']
            result[f'{metric}_{regime}'] = dict(counts, f1=n / (n + counts['missed'] + counts['extras']))
    return result


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


def pixels(row, source):
    choice = row['oracle']
    frame = Path(row['frame'])
    labels = read(frame / 'labels.json')
    shown = {tuple(face['xyz_world']): [] for face in labels['faces']}
    for face in labels['faces']:
        shown[tuple(face['xyz_world'])].append(face['id'])
    present = []
    for arm in ('incumbent', choice):
        pack = Path(row['options'][arm]['pack'])
        placement = read(pack.with_name('placements.json'))
        present.append({tuple(xyz) for xyz, _ in placement['aligned']})
    before, after = present
    with np.load(frame / 'labels.npz') as channels:
        owner = channels['face_id']
    rgb = np.asarray(Image.open(source['image']).convert('RGB')).copy()
    for changes, color in ((after - before, (20, 240, 135)), (before - after, (255, 155, 55))):
        active = np.isin(owner, [fid for coord in changes for fid in shown.get(coord, ())])
        rgb[active] = .24 * rgb[active] + .76 * np.array(color)
    return Image.fromarray(rgb)


def sheet(examples, output):
    width, height, gap = 640, 384, 24
    card = Image.new('RGB', (2 * width + 3 * gap, len(examples) * (height + 68) + 86), '#15191d')
    draw = ImageDraw.Draw(card)
    title = ImageFont.load_default(size=26)
    body = ImageFont.load_default(size=19)
    draw.text((gap, 15), 'One-block wall depth choices on frozen exports', fill='white', font=title)
    draw.text((gap, 52), 'Left  Source image     Right  Recorded visible faces of gained and lost positions',
              fill='#bfcad3', font=body)
    for index, (row, source, after) in enumerate(examples):
        y = 86 + index * (height + 68)
        text = row['id'].removeprefix('train-').removeprefix('test-').replace('-', ' ').title()
        change = row['options'][row['oracle']]['paired']
        draw.text((gap, y), text, fill='white', font=body)
        draw.text((width + 2 * gap, y), f"Perfect chooser  +{change['gained']} gained / {change['lost']} lost",
                  fill='white', font=body)
        with Image.open(source['image']) as original:
            for j, panel in enumerate((original, after)):
                fitted = ImageOps.contain(panel, (width, height))
                card.paste(fitted, (gap + j * (width + gap), y + 33))
    card.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/local-reprojection-options'))
    parser.add_argument('--out', type=Path, default=Path('benchmark/reviews/floating-local-reprojection-01'))
    args = parser.parse_args()
    root, output = args.run, args.out
    output.mkdir(parents=True, exist_ok=True)
    cohort = read(root / 'cohort.json')
    examples, rows = [], []
    for case in cohort['cases']:
        directory = root / case['panel'] / case['id']
        result = read(directory / 'results.json')
        source = read(directory / 'source.json')
        decisions = read(directory / 'depth.json')
        if result['oracle'] != 'incumbent':
            examples.append((result, source, pixels(result, source)))
        result.pop('frame')
        for entry in result['options'].values():
            entry.pop('pack')
        for entry in decisions.values():
            entry.get('provenance', {}).pop('field', None)
        result['patches'] = source['patches']
        result['options_offsets'] = {key: row['offsets'] for key, row in source['options'].items()}
        result['choices'] = dict(incumbent='incumbent', oracle=result['oracle'],
                                 **{key: row['choice'] for key, row in decisions.items()})
        result['selectors'] = decisions
        rows.append(result)
    groups = {group: [row for row in rows if row['group'] == group] for group in
              ('development', 'held-out', 'six-sky')}
    groups['total'] = rows
    implementation = Path(__file__).parent
    archived = implementation.resolve().parents[1] / 'floating-depth-refit' / 'benchmark'
    scorer = ('relative_face_scoring.py', 'expanded_scores.py', 'typed_structures.py',
              'expanded_score_validation.py', 'strip_visibility_audit.py', 'mask_blocks.py')
    metadata = dict(cohort, frozen_code_sha256={name: hash_file(implementation / name) for name in (
        'floating_local_reprojection.py', 'floating_local_reprojection_projection.py')},
        scorer_sha256={name: hash_file(archived / name) for name in scorer})
    save(output / 'conditions.json.gz', dict(metadata=metadata, rows=rows))
    save(output / 'summary.json', {group: {arm: group_scores(values, arm) for arm in ARMS}
                                    for group, values in groups.items()})
    sheet(examples, output / 'placement-deltas.png')


if __name__ == '__main__':
    main()
