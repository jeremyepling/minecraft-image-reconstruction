"""Test the frozen guarded attachment on four saved Factory views."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.floating_expanded_wall_attachment import (
    PREFIX, ROOT, WORKTREES, compare, data, digest, options, read, replay, save,
)
from benchmark.mask_blocks import decoded_pack


ARCHIVE = WORKTREES / 'floating-depth-refit/benchmark/reviews/floating-depth-refit-01/conditions.json.gz'
SOURCE = WORKTREES / 'floating-extra-isolation/runs/extra-isolation'
OUT = ROOT / 'runs/attachment-transfer'
REVIEW = ROOT / 'benchmark/reviews/floating-pregrid-attachment-transfer-01'
ARM = 'tangent-preserve-split'
LAYOUTS = frozenset({'factory-02', 'factory-03'})


def run(row):
    panel, name = row['panel'], row['id']
    case = next(c for c in data(panel)[3] if c['id'] == name)
    root = SOURCE / panel / 'large-contraction'
    chart = root / 'charts/sky' / name / 'charts.json'
    folder = root / 'assembly' / PREFIX / name
    prior = read(folder / 'score.json')
    packet = read(chart)
    image = Path(case['variants']['sky']['image'])
    assert digest(image) == prior['source_sha256'] == packet['source_sha256'] == row['source_sha256']
    assert digest(chart) == prior['chart_sha256']
    assert digest(folder / 'reconstruction.zip') == prior['pack_sha256']
    frame = Path(case['frame'])
    assert all(digest(frame / file) == value for file, value in prior['reference_sha256'].items())
    origins = prior['selected_origins']
    with Image.open(image) as opened:
        rgb = np.asarray(opened.convert('RGB'))
    # These archived charts never went through the adopted pink guarding pass.
    # Use the same rule, with no partition membership supplied on these inputs.
    additions, fits = options(packet, origins, rgb, {})
    truth = {tuple(b['xyz_world']): b['block_id'] for b in read(frame / 'labels.json')['blocks']}
    saved = read(folder / 'placements.json')
    assert decoded_pack(folder / 'reconstruction.zip') == {tuple(p): m for p, m in saved['local']}
    baseline = {tuple(p): m for p, m in saved['aligned']}
    arms = {}
    for arm, changes in (('noop', []), (ARM, additions[ARM])):
        dest = OUT / panel / arm
        replay(panel, case, root, arm, dest, changes, origins)
        result = dest / PREFIX / name
        receipt = read(result / 'score.json')
        placements = read(result / 'placements.json')
        assert decoded_pack(result / 'reconstruction.zip') == {tuple(p): m for p, m in placements['local']}
        if arm == 'noop':
            assert placements == saved and receipt['score'] == prior['score']
        aligned = {tuple(p): m for p, m in placements['aligned']}
        score = receipt['score']['fixed'] or receipt['score']['relaxed_yaw']
        arms[arm] = dict(score=score, paired=compare(baseline, aligned, truth), pack_sha256=receipt['pack_sha256'])
    return dict(panel=panel, id=name, layout=row['layout'], offers=fits, insertion_options=len(additions[ARM]),
                source_sha256=digest(image), chart_sha256=digest(chart),
                reference_sha256=prior['reference_sha256'], arms=arms)


def aggregate(rows, arm):
    result = {}
    for metric in ('geometry', 'scores'):
        for regime in ('strict', 'facade'):
            count = Counter()
            for row in rows:
                score = row['arms'][arm]['score'][metric][regime]
                count.update({key: score[key] for key in ('correct', 'missed', 'extras')})
            n = 2 * count['correct']
            result[f'{metric}_{regime}'] = dict(count, f1=n / (n + count['missed'] + count['extras']))
    return result


def main():
    cases = [row for row in read(ARCHIVE) if not row['active'] and row['layout'] in LAYOUTS]
    assert len(cases) == 4 and {row['layout'] for row in cases} == LAYOUTS
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(run, cases))
    summary = {key: aggregate(rows, key) for key in ('noop', ARM)}
    save(REVIEW / 'results.json', dict(source_archive_sha256=digest(ARCHIVE), rows=rows, summary=summary))
    for row in rows:
        print(row['id'], len(row['offers']), row['insertion_options'], row['arms'][ARM]['paired'])
    print(summary)


if __name__ == '__main__':
    main()
