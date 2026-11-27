"""Explain recorded camera acceptance on expanded captures with evaluation-only floor measurements."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_labels import screen_matrix
from benchmark.expanded_baseline import COHORTS, OUTPUT, ROOT, SOURCE, digest, frames, read, save
from engine.camera_atlas_floor import _atlas, register
from engine.camera_metric import predict
from engine.camera_texel_floor import _tracks, lattice


def measure(entry):
    frame = Path(entry['frame'])
    metadata = read(frame / 'frame.json')
    pose = read(frame / 'camera.json')
    files = metadata['files']
    if any(digest(frame / name) != files[name] for name in ('rgb.png', 'camera.json', 'labels.npz')):
        raise ValueError(f'Fixture changed for {entry["id"]}')
    if files['rgb.png'] != entry['rgb_sha256']:
        raise ValueError(f'Source changed for {entry["id"]}')
    with Image.open(frame / 'rgb.png') as image:
        rgb = np.asarray(image.convert('RGB'))
    with np.load(frame / 'labels.npz') as maps:
        owners = maps['face_id']
    h, w = owners.shape
    # The lower screen rectangle contains the staging floor and sometimes part of the build.
    # Use known world pose only for this analysis, after camera prediction.
    ground = metadata['ground_y'] + 1
    eye = np.array(pose['eye_world'])
    screen = screen_matrix(pose)
    ground_h = np.column_stack((screen[:, 0], screen[:, 2], screen[:, 1] * (ground - eye[1])))
    screen_to_ground = np.linalg.inv(ground_h)
    true_lattice = screen_to_ground.copy()
    true_lattice[:2] = 16 * (true_lattice[:2] + eye[[0, 2], None] * true_lattice[2])
    floor_horizon = np.cross(screen[:, 0], screen[:, 2])
    horizon_y = -(floor_horizon[0] * w/2 + floor_horizon[2]) / floor_horizon[1]
    offset, match = register(Image.fromarray(rgb), true_lattice, _atlas('26.3'))

    def floor_xz(x, y):
        pixels = np.stack((x, y, np.ones_like(x)))
        relative = screen_to_ground @ pixels
        return 16 * (relative[:2] / relative[2] + eye[[0, 2], None])

    # The two true X/Z floor-texel frequencies in cycles per *horizontal* screen pixel.
    # This matches how _tracks thresholds each row's spectrum, not isotropic texture scale.
    rows = np.arange(int(.86 * h), int(.983 * h)) + .5
    center = np.full_like(rows, w / 2)
    freq = np.abs(floor_xz(center + 1, rows) - floor_xz(center, rows))
    usable = np.all((freq >= .025) & (freq < .47), axis=0)
    # Test the follow-up bilateral atlas coverage criterion using the *true* texel lattice.
    yy, xx = np.mgrid[int(.74*h):int(.984*h):2, 0:w:2]
    xy = floor_xz(xx.ravel() + .5, yy.ravel() + .5)
    texel = np.floor(xy).astype(int)
    centers = np.max(abs(xy - texel - .5), axis=0) < .36
    pixels = rgb[yy, xx].reshape(-1, 3).astype(float)
    common_color = np.max(abs(pixels - np.median(pixels[::11], axis=0)), axis=1) < 14
    codes = texel[0] % 16 + 16 * (texel[1] % 16)
    empty = []
    for half in (xx.ravel() < w / 2, xx.ravel() >= w / 2):
        votes = np.bincount(codes[half & centers & common_color], minlength=256)
        empty.append(int((votes < 3).sum()))
    band = rgb[int(.86*h):int(.983*h)]
    data = band.astype(np.int16)
    median_color = np.median(data[::6, ::6], axis=(0, 1))
    similar = np.max(abs(data - median_color), axis=2) < 14
    edges = np.any(np.diff(data, axis=1) != 0, axis=2) & similar[:, 1:] & similar[:, :-1]
    peaks, _ = _tracks(edges[::8], rows[::8])
    found = [np.mean([bool(len(candidates) and np.min(abs(candidates - f)) < .0015)
                      for (_, candidates), f in zip(peaks, axis[::8], strict=True)]) for axis in freq]
    # For cases that reach registration, compare the final ordinary lower-band fit with true X/Z.
    # A pair like [1,0], [2,0] indicates one texel pitch and its second harmonic on the same axis.
    estimated, lattice_info = lattice(Image.fromarray(rgb), lower=.86)
    jacobian = None
    registration_reason = lattice_info.get('reason')
    if estimated is not None:
        transform = estimated @ np.linalg.inv(true_lattice)
        position = true_lattice @ np.array([w/2+.5, .90*h+.5, 1.])
        position /= position[2]
        site = transform @ position
        jacobian = ((transform[:2, :2] - (site[:2]/site[2])[:, None]*transform[2,:2]) / site[2]).tolist()
        estimated_offset, details = register(Image.fromarray(rgb), estimated, _atlas('26.3'))
        registration_reason = details.get('reason') if estimated_offset is None else 'atlas match'
    margins = metadata['framing']['visible_margins_px']
    return dict(id=entry['id'], dataset=entry['dataset'], split=entry['split'], layout=entry['layout'],
                palette=entry['palette'], angle=metadata['angle'], source_sha256=entry['rgb_sha256'],
                height_above_floor=float(eye[1]-ground), fov=pose['vertical_fov_deg'],
                pitch=pose['yaw_pitch_deg'][1], yaw=pose['yaw_pitch_deg'][0],
                bottom_margin_px=margins[3], minimum_margin_px=min(margins),
                labeled_fraction_in_registration_band=float((owners[yy, xx] != 0).mean()),
                true_track_band_fraction=float(usable.mean()),
                true_horizon_px=float(horizon_y), horizon_in_search=bool(-.4*h<horizon_y<.7*h),
                fraction_top_eight_matching_by_axis=list(map(float, found)),
                true_atlas_accepted=offset is not None,
                true_atlas_matches=match.get('atlas_matches'), true_atlas_reason=match.get('reason'),
                last_band_lattice_in_world_axes=jacobian,
                last_band_registration=registration_reason,
                true_pitch_px_at_90pct=sorted((1 / freq[:, abs(rows - .90*h).argmin()]).tolist()),
                true_atlas_bins_below_three=empty,
                lower_band_sha256=sha256(band.tobytes()).hexdigest())


def doubled_axis(row):
    jacobian = row['last_band_lattice_in_world_axes']
    if jacobian is None:
        return False
    values = np.abs(jacobian)
    axes = np.argmax(values, axis=1)
    if axes[0] != axes[1]:
        return False
    main = values[:, axes[0]]
    other = values[:, 1-axes[0]]
    return bool(max(other) < .1 and np.allclose(sorted(main), [1.,2.], atol=.1))


def parallel_tracks(row):
    if row['last_band_lattice_in_world_axes'] is None:
        return False
    jacobian = np.asarray(row['last_band_lattice_in_world_axes'])
    lengths = np.linalg.norm(jacobian, axis=1)
    return bool(min(lengths) > 0 and abs(np.linalg.det(jacobian))/np.prod(lengths) < np.sin(np.deg2rad(1)))


def summarize(rows):
    group_by_angle = defaultdict(Counter)
    groups = defaultdict(list)
    reasons = Counter()
    for row in rows:
        status = 'supported' if row['supported'] else 'failed'
        group_by_angle[row['angle']][status] += 1
        groups[status].append(row)
        if not row['supported']:
            reasons[(row['seam_reason'], row['atlas_reason'])] += 1
    same_view = defaultdict(list)
    for row in rows:
        same_view[(row['layout'], row['angle'])].append(row)
    pairs = [v for v in same_view.values() if len(v) == 2]
    by_failure = {}
    for reason in sorted({r['atlas_reason'] for r in rows if not r['supported']}):
        cases = [r for r in rows if not r['supported'] and r['atlas_reason'] == reason]
        by_failure[reason] = dict(views=len(cases),
                floor_passes_with_reference_lattice=sum(r['true_atlas_accepted'] for r in cases),
                last_band_matching_recorded_reason=sum(r['last_band_registration']==reason for r in cases),
                insufficient_true_tracks_in_top_eight=sum(min(r['fraction_top_eight_matching_by_axis']) < .55 for r in cases),
                true_frequencies_outside_range=sum(r['true_track_band_fraction'] < .55 for r in cases),
                true_horizon_outside_range=sum(not r['horizon_in_search'] for r in cases),
                parallel_floor_directions=sum(parallel_tracks(r) for r in cases),
                doubled_same_axis=sum(doubled_axis(r) for r in cases))
    by_status = {}
    for status, members in groups.items():
        def stats(key):
            values = [r[key] for r in members if r.get(key) is not None]
            return dict(n=len(values), min=min(values), median=float(np.median(values)), max=max(values)) if values else None
        by_status[status] = dict(views=len(members), surface_count=stats('surface_count'),
                longest_build_edge=stats('longest_build_edge'), camera_target_distance=stats('camera_target_distance'),
                height_above_floor=stats('height_above_floor'),
                floor_obscured_fraction=stats('labeled_fraction_in_registration_band'),
                bottom_margin_px=stats('bottom_margin_px'), pitch=stats('pitch'),
                median_fine_pitch_px=float(np.median([r['true_pitch_px_at_90pct'][0] for r in members])),
                median_coarse_pitch_px=float(np.median([r['true_pitch_px_at_90pct'][1] for r in members])),
                poor_track_range=sum(r['true_track_band_fraction'] < .55 for r in members),
                horizon_out_of_range=sum(not r['horizon_in_search'] for r in members),
                insufficient_true_tracks_in_top_eight=sum(min(r['fraction_top_eight_matching_by_axis']) < .55 for r in members),
                true_lattice_registers=sum(r['true_atlas_accepted'] for r in members),
                failed_true_fold_coverage=sum(max(r['true_atlas_bins_below_three']) > 0 for r in members))
    return dict(completed=len(rows), by_angle=dict(group_by_angle), by_status=by_status,
                atlas_failure_diagnostics=by_failure,
                vertical_fov_degrees=sorted({r['fov'] for r in rows}),
                reasons=[dict(seam=a, atlas=b, views=v) for (a,b),v in reasons.most_common()],
                same_layout_angle_palette_pairs=len(pairs),
                palette_pairs_disagree=sum(a['supported'] != b['supported'] for a,b in pairs),
                palette_pairs_identical_lower_strip=sum(a['lower_band_sha256'] == b['lower_band_sha256'] for a,b in pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--builds', type=Path, default=OUTPUT)
    parser.add_argument('--out', type=Path, default=ROOT/'benchmark/reviews/expanded-baseline-01/camera-diagnosis.json')
    parser.add_argument('--camera-replay', action='store_true', help='Verify production camera output on every RGB independently')
    args = parser.parse_args()
    cache = args.builds / 'scores' / 'camera-floor-reference-diagnostics.json'
    entries = list(frames(args.source, COHORTS))
    feature_rows = read(cache) if cache.is_file() else [measure(e) for e in entries]
    if not cache.is_file():
        save(cache, feature_rows)
    if [(r['id'],r['source_sha256']) for r in feature_rows] != [(e['id'],e['rgb_sha256']) for e in entries]:
        raise ValueError('Cached camera diagnostics differ from frozen source set')
    for entry, features in zip(entries, feature_rows, strict=True):
        frame = Path(entry['frame'])
        metadata = read(frame/'frame.json')
        camera = read(frame/'camera.json')
        features['longest_build_edge'] = max(metadata['build_size_xyz'])
        features['camera_target_distance'] = float(np.linalg.norm(np.subtract(camera['eye_world'], metadata['target'])))
    camera_replays = {}
    for entry in entries:
        path = args.builds/'scores'/'camera-stage-replay'/(entry['id']+'.json')
        if args.camera_replay and not path.exists():
            image_path = Path(entry['frame'])/'rgb.png'
            if digest(image_path) != entry['rgb_sha256']:
                raise ValueError(f'Source changed for {image_path}')
            with Image.open(image_path) as image:
                camera = predict(image, '26.3')
            save(path, dict(id=entry['id'], source_sha256=entry['rgb_sha256'], camera=camera))
        if path.exists():
            cached = read(path)
            if cached['source_sha256'] != entry['rgb_sha256']:
                raise ValueError(f'Source fingerprint mismatch in {path}')
            camera_replays[entry['id']] = cached['camera']
    measured = []
    for features in feature_rows:
        folder = args.builds/features['dataset']/features['id']
        if not (folder/'receipt.json').exists():
            continue
        manifest = read(folder/'build-manifest.json')
        if manifest['image_sha256'] != features['source_sha256']:
            raise ValueError(f'Wrong build for {folder}')
        placement = manifest['placement']
        replay = camera_replays.get(features['id'])
        if replay is not None and (placement['P'] != replay['P'] or placement['reason'] != replay['reason']
                or placement.get('atlas_camera_reason') != replay.get('atlas_reason')):
            raise ValueError(f'Standalone camera replay differs from full build for {folder}')
        supported = placement['P'] is not None and placement['camera_status'] == 'supported'
        surfaces = read(folder/'surfaces.json')['planes']
        measured.append(dict(features, supported=supported, placed=placement['placed_blocks'],
                seam_reason=placement['reason'] if not supported else None,
                atlas_reason=placement.get('atlas_camera_reason'), surface_count=len(surfaces)))
    summaries = {k:summarize([r for r in measured if r['dataset']==k]) for k in COHORTS}
    camera_cases = []
    completed = {r['id']:r for r in measured}
    for features in feature_rows:
        key = features['id']
        if key in completed:
            camera_cases.append(dict(completed[key],full_build=True))
        elif key in camera_replays:
            camera = camera_replays[key]
            supported = camera['P'] is not None
            camera_cases.append(dict(features, full_build=False, supported=supported,
                    seam_reason=camera['reason'] if not supported else None, atlas_reason=camera.get('atlas_reason')))
    camera_summaries = {k:summarize([r for r in camera_cases if r['dataset']==k]) for k in COHORTS}
    save(args.out, dict(revision=read(args.builds/'source_revision.json')['revision'],
                        available=COHORTS, verified_camera_replays=len({r['id'] for r in measured} & camera_replays.keys()),
                        completed_build_summaries=summaries, camera_summaries=camera_summaries, cases=camera_cases))
    print({k:v['completed'] for k,v in summaries.items()})


if __name__ == '__main__':
    main()
