"""Verify published camera experiments against the independent decoded grades."""

import argparse
from pathlib import Path
from statistics import mean, median

from benchmark.camera_horizon_replay import camera_validation
from benchmark.camera_model_controls import REPEAT_BUILDS, REVIEW, check, data_hash, frozen_frames, indexes
from benchmark.expanded_baseline import WORKTREES, digest, read, save


MOGE = WORKTREES / 'camera-moge-surface-first'
GEO = WORKTREES / 'camera-geocalib'
MOGE_RUN = MOGE / 'runs/camera-moge-surface-first-01'
GEO_RUN = GEO / 'runs/camera-geocalib-01'
PORT_CHECK = WORKTREES / 'camera-model-defaults/runs/camera-repeat-expanded-port-check'


def camera_statistics(rows):
    if not rows:
        return dict(frames=0)
    diagnostics = [r['native_camera'] for r in rows]
    return dict(frames=len(rows), native_visible_corners=sum(r['vertices'] for r in diagnostics),
                corners_within_2px=sum(r['within_2px'] for r in diagnostics),
                median_per_frame_median_px=median(r['median_reprojection_px'] for r in diagnostics),
                worst_per_frame_p90_px=max(r['p90_reprojection_px'] for r in diagnostics),
                worst_fov_error_deg=max(r['fov_error_deg'] for r in diagnostics),
                worst_relative_height_error=max(abs(r['relative_scale_error']) for r in diagnostics),
                registered_correct=sum(r['registered_geometry']['strict']['correct'] for r in rows),
                translated_correct=sum(r['geometry']['strict']['correct'] for r in rows),
                extras=sum(r['geometry']['strict']['extras'] for r in rows))


def acceptances(arm, entries, scores, camera_and_folder):
    cases = []
    for scored in scores:
        if scored['model_coverage'] != 'candidate_export':
            continue
        key = scored['id']
        proposal, folder = camera_and_folder(scored)
        placement = read(folder / 'build-manifest.json')['placement']
        check(placement['P'], proposal['P'], f'{arm} {key} through-placement camera')
        check(digest(folder / 'reconstruction.zip'), scored['pack_sha256'], f'{arm} {key} pack')
        report = camera_validation(entries[key], placement)
        cases.append(dict(id=key, dataset=scored['dataset'], split=scored['split'],
                          source_sha256=scored['source_sha256'], pack_sha256=scored['pack_sha256'],
                          decoded_sha256=scored['decoded_sha256'], geometry=scored['geometry'], scores=scored['scores'],
                          registered_geometry=scored['registration']['geometry'], native_camera=report))
    return cases


def check_grade(name, graded, comparison):
    for field in ('geometry', 'scores', 'camera_usable', 'pack_sha256'):
        check(graded[field], comparison[field], f'{name} {graded["id"]} {field}')


def raw_errors(proposals, geo_rows, accepted):
    moge = {'accepted': [], 'abstained': []}
    for row in proposals.values():
        result = row['orientation']
        group = 'accepted' if row['result'] else 'abstained'
        moge[group].append(dict(worst_axis_deg=result['max_axis_deg'], focal_ratio_abs_error=abs(result['focal_ratio']-1),
                                orientation_within_2deg_and_5percent=result['max_axis_deg'] <= 2 and
                                abs(result['focal_ratio']-1) <= .05))
    geo = {'accepted': [], 'abstained': []}
    for row in geo_rows.values():
        if 'geocalib_camera' in row:
            group = 'accepted' if row['id'] in accepted else 'abstained'
            geo[group].append({k: v for k, v in row['proposal_error'].items() if k != 'true_horizon_px'})
    return dict(moge={name: dict(frames=len(rows), worst_axis_median_deg=median(r['worst_axis_deg'] for r in rows),
                                focal_ratio_median_abs_error=median(r['focal_ratio_abs_error'] for r in rows),
                                axis_and_focal_passes=sum(r['orientation_within_2deg_and_5percent'] for r in rows))
                      for name, rows in moge.items()},
                geocalib={name: dict(frames=len(rows), **{k: mean(r[k] for r in rows) for k in rows[0]})
                          for name, rows in geo.items()})


def budget_cases(baselines, graded, port_check):
    ordinary = {row['id']: row for row in baselines['ordinary']}
    rows = []
    for row in graded.values():
        key = row['id']
        previous = ordinary[key]
        evidence = dict(source_policy='ordinary', fallback_camera_sha256=None)
        if not previous['camera_usable']:
            proof = read(port_check / (key + '.json'))
            check(proof['source_sha256'], previous['source_sha256'], f'{key} production RGB')
            check(proof['dataset'], previous['dataset'], f'{key} production dataset')
            check(proof['split'], previous['split'], f'{key} production split')
            prior = read(REPEAT_BUILDS / key / 'repeat.json')['camera']
            if prior['P'] is None:
                prior = read(GEO_RUN / 'matched-control/metric' / (key + '.json'))['camera']
            camera = proof['camera']
            check(camera, prior, f'{key} serialized production fallback')
            supported = camera['P'] is not None
            check(row['camera_usable'], supported, f'{key} production fallback support')
            evidence.update(source_policy=(camera['camera_source'] if supported else 'abstained'),
                            fallback_camera_sha256=data_hash(camera),
                            prior_camera_reason=previous['camera_reason'],
                            atlas_camera_reason=previous['atlas_camera_reason'])
            if not supported:
                evidence.update(camera_status=previous['camera_status'], camera_reason=previous['camera_reason'],
                                repeat_camera_reason=camera['reason'])
        rows.append(dict(row, **evidence))
    return rows


def compare(output, grades, port_check):
    entries = {e['id']: e for e in frozen_frames()}
    moge_release = MOGE / 'benchmark/reviews/camera-moge-surface-first-01'
    geo_release = GEO / 'benchmark/reviews/camera-geocalib-01'
    moge_manifest = read(moge_release / 'manifest.json')
    geo_summary = read(geo_release / 'summary.json')
    check(digest(REVIEW / 'frames.json'), moge_manifest['control_frames_sha256'], 'MoGe scoring control frames')
    moge_proposals = indexes(moge_release / 'proposals.json.gz', moge_manifest['proposals_sha256'])
    geo_cases = indexes(geo_release / 'cases.json.gz', geo_summary['cases_sha256'])
    baselines = read(REVIEW / 'baseline-cases.json.gz')
    base_rows = {name: {r['id']: r for r in rows} for name, rows in baselines.items()}
    for key, proposal in (list(moge_proposals.items()) + list(geo_cases.items())):
        check(proposal['source_sha256'], entries[key]['rgb_sha256'], f'{key} published RGB')
        check(proposal['split'], entries[key]['split'], f'{key} published split')
        check(proposal['dataset'], entries[key]['dataset'], f'{key} published dataset')

    records, experimental_cases = {}, {}
    for name in ('moge-standalone', 'moge-repeat-first', 'geocalib', 'matched-boundary'):
        grade = grades / name
        stats = read(grade / 'summary.json')
        cases = indexes(grade / 'cases.json.gz', stats['cases_sha256'])
        check(cases.keys(), entries.keys(), f'{name} frame keys')
        check(stats['pending_attempts'], 0, f'{name} pending attempts')
        if name.startswith('moge'):
            for key, row in cases.items():
                proposal = moge_proposals.get(key)
                prior = base_rows[stats['inherited']][key]
                expected = proposal['result'] if proposal and row['model_coverage'] == 'candidate_export' else prior
                check_grade(name, row, expected)

            def proposal_and_build(row):
                return moge_proposals[row['id']]['projection'], MOGE_RUN / row['dataset'] / row['id'] / 'build'

        else:
            policy = 'geocalib' if name == 'geocalib' else 'matched'
            root = GEO_RUN if name == 'geocalib' else GEO_RUN / 'matched-control'
            for key, row in cases.items():
                check_grade(name, row, geo_cases[key][policy])

            def proposal_and_build(row):
                return geo_cases[row['id']][policy + '_camera'], root / 'builds' / row['dataset'] / row['id']

        accepted = acceptances(name, entries, cases.values(), proposal_and_build)
        records[name] = dict(grade=stats, accepted=accepted, accepted_summary=camera_statistics(accepted),
                             grade_summary_sha256=digest(grade / 'summary.json'))
        experimental_cases[name] = list(cases.values())
        if name == 'matched-boundary':
            matched_cases = budget_cases(baselines, cases, port_check)
            save(output / 'independent-matched-cases.json.gz', matched_cases)
            save(output / 'independent-matched-summary.json', dict(stats,
                 cases_sha256=digest(output / 'independent-matched-cases.json.gz'),
                 camera_port_summary_sha256=digest(port_check / 'summary.json'),
                 measurement='48 verified camera-handoff replays and 240 retained decoded packs'))
    duplicate = {row['id'] for row in records['geocalib']['accepted']} - {row['id'] for row in records['matched-boundary']['accepted']}
    native = raw_errors(moge_proposals, geo_cases, {row['id'] for row in records['geocalib']['accepted']})
    save(output / 'experimental-cases.json.gz', experimental_cases)
    save(output / 'experimental-comparison.json', dict(controls_sha256=digest(REVIEW / 'manifest.json'),
         moge_manifest_sha256=digest(moge_release / 'manifest.json'), geocalib_summary_sha256=digest(geo_release / 'summary.json'),
         cases_sha256=digest(output / 'experimental-cases.json.gz'),
         arms=records, evaluation_only_raw_proposal_errors=native,
         GeoCalib_only_camera_acceptances=sorted(duplicate)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grades', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=REVIEW)
    parser.add_argument('--port-check', type=Path, default=PORT_CHECK)
    args = parser.parse_args()
    compare(args.output, args.grades, args.port_check)


if __name__ == '__main__':
    main()
