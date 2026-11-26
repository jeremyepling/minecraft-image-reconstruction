"""Reproduce the Stage 1 comparison on the frozen Java 26.3 captures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
from PIL import Image, __version__ as pillow_version

from benchmark.camera_metrics import score, summarize
from engine.camera_grid import METHODS, predict
from runtime.mode import image_only, mark_image_only


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'benchmark/data/camera-round-01'
RULE = ('Highest layout-averaged fraction of visible vertices within 2 source pixels, '
        'counting missing projections as misses. Break ties by metric frame count, '
        'then lower conditional visible-span MAE; retain registry order on exact ties.')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf8')


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def fixture_digest(source, index):
    paths = {'index.json', index['layout'], index['buildings'], index['views']}
    for frame in index['frames']:
        paths.update(frame[k] for k in ('image', 'pose', 'states', 'labels'))
    hashes = ''.join(f'{name} {digest(source / name)}\n' for name in sorted(paths))
    return hashlib.sha256(hashes.encode()).hexdigest()


def infer(job):
    image_path, version, method, output = job
    with Image.open(image_path) as src:
        # Only pixel data crosses the interface; strip filename and PNG metadata.
        rgb = Image.fromarray(np.asarray(src.convert('RGB')))
    save(output, predict(rgb, version, method))


def revision():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT).decode().strip()
    paths = [ROOT / 'reconstruct.py', ROOT / 'requirements.txt']
    for name in ('benchmark', 'engine', 'calibration', 'runtime'):
        paths.extend(p for p in (ROOT / name).rglob('*') if p.suffix in ('.py', '.java'))
    fingerprint = hashlib.sha256()
    for path in sorted(paths):
        fingerprint.update(f'{path.relative_to(ROOT).as_posix()} {digest(path)}\n'.encode())
    return dict(commit=git('rev-parse', 'HEAD'), predictor_and_runner_sha256=fingerprint.hexdigest(),
                source_dirty=bool(git('status', '--porcelain', '--', '*.py', '*.java', 'requirements.txt')))


def evaluate(fixture, out, frames, version, split, workers):
    jobs, rows = [], []
    for method in METHODS:
        dest = out / 'predictions' / method
        dest.mkdir(exist_ok=True, parents=True)
        jobs.extend((fixture / f['image'], version, method, dest / (f['id'] + '.json')) for f in frames)
    with ProcessPoolExecutor(max_workers=workers, initializer=mark_image_only) as pool:
        list(pool.map(infer, jobs))
    for f in frames:
        labels = read(fixture / f['labels'])['faces']
        pose = read(fixture / f['pose'])
        for m in METHODS:
            prediction = read(out / 'predictions' / m / (f['id'] + '.json'))
            rows.append(dict(id=f['id'], layout=f['layout'], condition=f['condition'], method=m,
                             metric=prediction['scale_status'] == 'supported', **score(prediction, pose, labels)))
    summary = {m: summarize([r for r in rows if r['method'] == m]) for m in METHODS}
    save(out / (split + '-scores.json'), dict(summary=summary, rows=rows))
    return summary, rows


def validate(fixture):
    index = read(fixture / 'index.json')
    frames = index['frames']
    if index['version'] != '26.3' or not frames or len({f['id'] for f in frames}) != len(frames):
        raise ValueError('Invalid Stage 1 fixture index')
    layouts = {s: sorted({f['layout'] for f in frames if f['split'] == s}) for s in ('dev', 'holdout')}
    if (set(f['split'] for f in frames) != set(layouts) or not all(layouts.values())
        or set(layouts['dev']).intersection(layouts['holdout'])):
        raise ValueError('The two splits require disjoint layouts.')
    for f in frames:
        for k, sha in [('image','image_sha256'), ('labels','label_sha256')]:
            if digest(fixture / f[k]) != f[sha]:
                raise ValueError(f'Fixture hash mismatch in {f[k]}')
    return index, layouts


@image_only()
def run(fixture, out, workers=3, dev_only=False):
    out.mkdir(parents=True, exist_ok=False)
    fixture = (fixture or FROZEN).resolve()
    index, layouts = validate(fixture)
    run_id = out.resolve().name
    dev, _ = evaluate(fixture, out, [f for f in index['frames'] if f['split'] == 'dev'],
                      index['version'], 'dev', workers)
    def rank(method):
        s = dev[method]
        err = s['layout_span_mae_blocks']
        return (s['layout_hit_rate'], s['metric_images'], -err if err is not None else -float('inf'))
    winner = max(METHODS, key=rank)
    print(f'Development selection: {winner}', flush=True)
    settings = {}
    for method in METHODS:
        path = out / 'predictions' / method / (next(f['id'] for f in index['frames'] if f['split'] == 'dev') + '.json')
        settings[method] = read(path).get('settings', {})
    code = revision()
    lock = dict(run_id=run_id, stage='camera', revision=code, methods=list(METHODS), settings=settings,
                selection_rule=RULE, selected=winner, dev=dev, fixture_sha256=fixture_digest(fixture, index))
    save(out / 'selection.json', lock)
    if dev_only:
        return
    holdout, rows = evaluate(fixture, out, [f for f in index['frames'] if f['split'] == 'holdout'],
                             index['version'], 'holdout', workers)
    from benchmark.camera_sheet import sheet
    regions = sheet(fixture, out, run_id, index['frames'], rows, holdout, winner)
    save(out / 'regions.json', regions)
    (out / 'feedback.jsonl').touch()
    manifest = dict(schema=1, run_id=run_id, revision=code, version=index['version'],
                    environment=dict(python=platform.python_version(), numpy=np.__version__, pillow=pillow_version),
                    fixture=dict(index=os.path.relpath(fixture / 'index.json', out).replace('\\','/'),
                                 sha256=lock['fixture_sha256'], index_sha256=digest(fixture / 'index.json'),
                                 layouts=layouts, **index['verified']),
                    stage='camera', baseline='pavement', methods=list(METHODS), settings=settings,
                    common=dict(camera_input='one unannotated RGB framebuffer and game version',
                                charts='engine.camera_grid.contact_planes',
                                unknown_scale_axes='engine.cube_axes.fit_axes',
                                registration_reference_uv=[.5,.72], corner_tolerance_px=2,
                                span_minimum_blocks=3, span_face_minimum_visible_fraction=.98,
                                span_endpoints_and_plane='supplied during scoring'),
                    recommended=winner, selection_rule=RULE, scores=dict(dev=dev, holdout=holdout),
                    artifacts=dict(lock='selection.json', dev='dev-scores.json', holdout='holdout-scores.json',
                                   predictions='predictions/', sheet='comparison.png', regions='regions.json',
                                   visual_feedback='feedback.jsonl'))
    save(out / 'manifest.json', manifest)
    print(out.resolve() / 'manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, help='Use a captured fixture directory instead of the bundled frames')
    parser.add_argument('--out', type=Path, help='New run directory; its name is the run ID')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--dev-only', action='store_true')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run(args.fixture, args.out or ROOT / 'runs/camera' / ('camera-01-' + stamp), args.workers, args.dev_only)


if __name__ == '__main__':
    main()
