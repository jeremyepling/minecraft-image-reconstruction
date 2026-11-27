"""Scoring-only replay on recorded native visible corners and source-proposed metric cameras."""
from concurrent.futures import ProcessPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from benchmark.camera_native_floor_run import CONTROL, REVIEW, RUN, SOURCE, WORKTREES, read, save, digest
from benchmark.camera_horizon_replay import camera_validation

SCORER = Path(__file__).with_name('camera_relative_calibration.py')
if not SCORER.is_file():
    SCORER = WORKTREES / 'camera-recovery-next/benchmark/camera_relative_calibration.py'


def load_scorer():
    spec = spec_from_file_location('native_corner_eval', SCORER)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.native_corner_score


def score(row):
    native = load_scorer()
    name = row['id'] + '.json'
    frame = SOURCE / row['dataset'] / 'data' / row['split'] / 'frames' / row['id']
    pose, faces = read(frame / 'camera.json'), read(frame / 'labels.json')['faces']
    selected = read(RUN / 'metric' / name)
    details = {}
    for arm in ('boundary', 'geocalib'):
        receipt = read(RUN / arm / 'metric' / name)
        camera = receipt['camera']
        records = []
        for proposal in camera['evidence'].get('refined_proposals', []):
            try:
                assessment = native(proposal['P'], pose, faces)
            except (ValueError, ArithmeticError):
                assessment = dict(rejected_geometry=True)
            records.append(dict(proposal=proposal, accepted=proposal['accepted'], native=assessment))
        details[arm] = dict(accepted=camera['P'] is not None, tested=camera['evidence']['checked'],
                            refined=records, file_sha256=digest(RUN / arm / 'metric' / name))
    result = dict(id=row['id'], dataset=row['dataset'], split=row['split'], rgb_sha256=selected['rgb_sha256'],
                  choice=selected['arm'], camera=selected['camera'], arms=details)
    if selected['camera']['P'] is not None:
        result['registered'] = camera_validation(dict(frame=frame), selected['camera'])
        result['relative'] = native(selected['camera']['P'], pose, faces)
    return result


if __name__ == '__main__':
    with ProcessPoolExecutor(3) as pool:
        cases = list(pool.map(score, (r for r in read(CONTROL / 'independent-matched-cases.json.gz')
                                      if not r['camera_usable'])))
    save(REVIEW / 'native-cameras.json.gz', cases)
    print(len(cases), 'attempts', sum('relative' in c for c in cases), 'calibrated cameras')
