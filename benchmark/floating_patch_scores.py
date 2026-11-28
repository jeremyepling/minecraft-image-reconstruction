"""Audit the archived and new floating-build exports against their visible labels."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from benchmark import expanded_scores, floating_cloud_replay as cloud, floating_fresh_validation as fresh
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_score_validation import best_translation_fast
from benchmark.floating_patch_validation import (
    BRIDGE, BRIDGE_REVISION, FROZEN, HOLDOUT2_FROZEN, HOLDOUT2_OUT, OUT, REVIEW, bridge_runner,
)
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_scoring import _yaw
from benchmark.sky_crop_guard import pooled
from benchmark.typed_structures import geometry_score, score


PREFIX = Path('connected-atlas/sky')
METRICS = ('geometry_strict', 'geometry_facade', 'scores_strict', 'scores_facade')
PUBLISHED_SCORES_SHA256 = '4e76635d7f1ea263f1d76f88f52bcd2dd07990ee3f427014e46987f49b68ecab'


def case_reference(case):
    metadata = Path(case['frame']) / 'frame.json'

    def with_framing(path):
        data = read(path)
        if path != metadata or 'framing' in data:
            return data
        audit = read(ROOT / 'training/framing-audit-v2.json')
        proof = next(entry for entry in audit['frames'] if entry['id'] == case['id'])
        if proof['rgb_sha256'] != case['rgb_sha256']:
            raise ValueError(f'Framing audit covers a different source: {metadata}')
        return dict(data, framing=proof)

    with patch.object(expanded_scores, 'read', with_framing):
        return expanded_scores.load_reference(case)


def arm_exports(case, reference, name, assembly, charts, handoff):
    folder = assembly / PREFIX / case['id']
    chart = charts / 'sky' / case['id'] / 'charts.json'
    pack = folder / 'reconstruction.zip'
    receipt = read(folder / 'score.json')
    placement = read(folder / 'placements.json')
    packet = read(chart)
    image = case['variants']['sky']
    source_sha = digest(Path(image['image']))
    provenance = cloud.check_handoff(packet, handoff / 'sky' / case['id'] / 'source-handoff', source_sha)
    expected = dict(id=case['id'], source_sha256=source_sha, reference_sha256=reference.hashes,
                    chart_sha256=digest(chart), pack_sha256=digest(pack), **provenance)
    if source_sha != image['source_sha256'] or any(receipt[key] != value for key, value in expected.items()):
        raise ValueError(f'Export provenance differs from frozen handoffs: {folder}')

    local = decoded_pack(pack)
    archived = receipt['score']
    selected = archived['fixed'] or archived['relaxed_yaw']
    turned = {_yaw(cell, selected['yaw_quarters']): material for cell, material in local.items()}
    shift, _ = best_translation_fast(turned, reference.truth)
    aligned = {(x + shift[0], y + shift[1], z + shift[2]): material for (x, y, z), material in turned.items()}
    geometry = geometry_score(aligned, reference.truth, reference.optional)
    typed = score(aligned, reference.truth, reference.optional, set())
    count_fields = ('strict', 'facade', 'material')
    if (local != {tuple(cell): material for cell, material in placement['local']}
            or aligned != {tuple(cell): material for cell, material in placement['aligned']}
            or list(shift) != selected['translation'] or geometry != selected['geometry']
            or any(typed[key] != selected['scores'][key] for key in count_fields)):
        raise ValueError(f'Decoded blocks disagree with the saved alignment or score: {folder}')

    alignment = 'empty export' if not local else 'unresolved yaw' if archived['fixed'] is None else 'fixed yaw'
    return dict(arm=name, **expected, receipt_sha256=digest(folder / 'score.json'), alignment=alignment,
                yaw=receipt['yaw'], exported=len(local), score=selected), aligned, packet


def verify_case(job):
    case, paths = job
    reference = case_reference(case)
    arms, placements, charts = {}, {}, {}
    for label, inputs in paths.items():
        arms[label], placements[label], charts[label] = arm_exports(case, reference, label, *inputs)
    ordinary = arms['ordinary']
    if ordinary['alignment'] == 'fixed yaw':
        for label, arm in arms.items():
            if arm['alignment'] != 'unresolved yaw':
                continue
            compass = fresh.ordinary_compass(charts['ordinary'], charts[label], ordinary['score']['yaw_quarters'])
            if compass and compass['candidate_yaw_quarters'] == arm['score']['yaw_quarters']:
                arm.update(alignment='transferred fixed yaw', compass=compass)

    wall, horizontal = (placements[arm] for arm in ('whole-wall', 'horizontal-cells'))
    before, after = wall.keys(), horizontal.keys()
    truth = reference.truth.keys()
    gained, lost = after - before, before - after
    effects = dict(gained=len(gained & truth), lost=len(lost & truth),
                   extras_added=len(gained - truth), extras_removed=len(lost - truth),
                   changed_ids=sum(wall[p] != horizontal[p] for p in before & after))
    return dict(id=case['id'], dataset=case['dataset'], split=case['split'], layout=case['layout'],
                arms=arms, paired=effects)


def wall_roots(root, ordinary, tiles):
    material = root / 'source_material'
    return {
        'ordinary': (ordinary / 'assembly', ordinary / 'atlas/connected', ordinary),
        'whole-wall': (material / 'assembly', material / 'charts', ordinary),
        'horizontal-cells': (tiles, material / 'charts', ordinary),
    }


def run_panels():
    bridge = bridge_runner()
    run = BRIDGE / 'runs'
    for panel, cases, root, ordinary in (
            ('previous', bridge.cases(), bridge.OLD, bridge.ORDINARY),
            ('fresh', bridge.fresh_cases(), bridge.FRESH, bridge.FRESH / 'ordinary'),
            ('supplemental', bridge.fresh_cases(True), bridge.FRESH, bridge.FRESH / 'ordinary'),
            ('new', fresh.cases(FROZEN), OUT, OUT / 'ordinary')):
        tiles = (OUT if panel == 'new' else run) / 'horizontal-cells/assembly'
        yield panel, list(cases), wall_roots(root, ordinary, tiles)
    sky = run / 'six-sky'
    handoff = WORKTREES / 'floating-visible-builds/runs/floating-cloud-gate-01'
    yield 'sky', read(bridge.round_.SHARED / 'fixtures.json')['cases'], {
        'ordinary': (sky / 'ordinary', handoff / 'garden-anchor-extension/connected', handoff),
        'whole-wall': (sky / 'whole-wall', sky / 'charts', handoff),
        'horizontal-cells': (sky / 'horizontal-cells', sky / 'charts', handoff),
    }


def summarize(rows):
    groups = {'all': rows}
    for dataset in sorted({row['dataset'] for row in rows}):
        groups[dataset] = [row for row in rows if row['dataset'] == dataset]
    result = {}
    for group, members in groups.items():
        arms = {}
        for label in members[0]['arms']:
            exports = [row['arms'][label] for row in members]
            arms[label] = dict(pooled([arm['score'] for arm in exports]),
                               unresolved_yaw=[arm['id'] for arm in exports if arm['alignment'] == 'unresolved yaw'],
                               transferred_yaw=[arm['id'] for arm in exports if arm['alignment'] == 'transferred fixed yaw'],
                               empty_export=[arm['id'] for arm in exports if arm['alignment'] == 'empty export'])
        result[group] = dict(views=len(members), arms=arms)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--panel', choices=('previous', 'fresh', 'supplemental', 'sky', 'new', 'holdout2'))
    args = parser.parse_args()
    archive = BRIDGE / 'benchmark/reviews/floating-horizontal-bridges-01/results.json.gz'
    if digest(archive) != PUBLISHED_SCORES_SHA256:
        raise ValueError(f'Results differ from {BRIDGE_REVISION}: {archive}')
    published = read(archive)
    panels = {}
    frozen = FROZEN
    groups = run_panels()
    if args.panel == 'holdout2':
        frozen = HOLDOUT2_FROZEN
        extra = wall_roots(HOLDOUT2_OUT, HOLDOUT2_OUT / 'ordinary', HOLDOUT2_OUT / 'horizontal-cells/assembly')
        groups = [('holdout2', list(fresh.cases(frozen)), extra)]
    with ProcessPoolExecutor(args.workers) as pool:
        for panel, cases, roots in groups:
            if args.panel and args.panel != panel:
                continue
            rows = list(pool.map(verify_case, ((case, roots) for case in cases)))
            if panel in published:
                expected = {row['id']: row['score'] for row in published[panel]['rows']}
                if expected.keys() != {row['id'] for row in rows}:
                    raise ValueError(f'Archived frame set changed: {panel}')
                for row in rows:
                    for label in expected[row['id']].keys() & row['arms'].keys():
                        if row['arms'][label]['score'] != expected[row['id']][label]:
                            raise ValueError(f'Archived score changed: {panel}/{row["id"]}/{label}')
            panels[panel] = dict(rows=rows, cohorts=summarize(rows))
            for group, cohort in panels[panel]['cohorts'].items():
                print(panel, group, cohort['views'], {arm: [round(100 * totals[k]['f1'], 2) for k in METRICS]
                                                     for arm, totals in cohort['arms'].items()}, flush=True)
    output = OUT / f'{args.panel}-results.json.gz' if args.panel else REVIEW / 'results.json.gz'
    if args.panel == 'holdout2':
        output = REVIEW / 'holdout2-results.json.gz'
    save(output, dict(bridge_revision=BRIDGE_REVISION, published_scores_sha256=PUBLISHED_SCORES_SHA256,
                      frozen_manifest_sha256=fresh.fixture_hash(frozen), panels=panels))


if __name__ == '__main__':
    main()
