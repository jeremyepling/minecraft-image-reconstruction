"""Verify the model-free production fallback against the frozen research arm."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

from benchmark.camera_model_controls import REPEAT_BUILDS, inventory_hash
from benchmark.camera_model_entry_check import CLI
from benchmark.camera_native_floor_run import BASE, COHORTS, CONTROL, REVIEW, ROOT, RUN, SOURCE, digest, frames, read, save
from benchmark.mask_blocks import decoded_pack
from engine import camera_metric


def check_camera(entry):
    import cv2
    cv2.setNumThreads(1)
    filename = entry['id'] + '.json'
    path = Path(entry['frame']) / 'rgb.png'
    expected = read(RUN / 'boundary/metric' / filename)
    assert digest(path) == entry['rgb_sha256'] == expected['rgb_sha256']
    with Image.open(path) as image:
        result = camera_metric.predict(image, '26.3')
    if result['P'] != expected['camera']['P']:
        raise ValueError(f"Production camera differs on {entry['id']}")
    return dict(id=entry['id'], split=entry['split'], rgb_sha256=entry['rgb_sha256'],
                camera_source=result['camera_source'], accepted=result['P'] is not None,
                exact_matrix_or_abstention=True, research_receipt_sha256=digest(RUN / 'boundary/metric' / filename))


def check_cli(job):
    entry, incumbent = job
    key = entry['id']
    baseline = BASE / entry['dataset'] / key
    out = RUN / 'normal-cli' / key
    out.mkdir(parents=True, exist_ok=True)
    rgb = Path(entry['frame']) / 'rgb.png'
    expected = baseline
    if incumbent['source_policy'] == 'horizon-repeat-floor':
        expected = REPEAT_BUILDS / key / 'repeat'
    if incumbent['source_policy'] == 'boundary-budget-repeat':
        from benchmark.camera_model_entry_check import MATCHED
        expected = MATCHED / entry['dataset'] / key
    new_camera = False
    if incumbent['source_policy'] == 'abstained' and read(RUN / 'boundary/metric' / (key + '.json'))['camera']['P'] is not None:
        expected = RUN / 'builds' / entry['dataset'] / key
        new_camera = True
    if not new_camera:
        assert digest(expected / 'reconstruction.zip') == incumbent['pack_sha256']
    with (out / 'execution.log').open('w') as log:
        subprocess.run([sys.executable, '-c', CLI, str(rgb), str(baseline), str(out), 'cached-fields'],
                       cwd=ROOT, env={**os.environ, 'MCIR_IMAGE_ONLY':'1'}, stdout=log, stderr=subprocess.STDOUT, check=True)
    actual = read(out / 'build-manifest.json')
    reference = read(expected / 'build-manifest.json')
    assert actual['image_sha256'] == entry['rgb_sha256'] == reference['image_sha256']
    assert actual['placement']['P'] == reference['placement']['P']
    if new_camera:
        verified = read(RUN / 'boundary/metric' / (key+'.json'))['camera']['evidence']
        assert actual['placement']['native_floor_evidence']['native_correlation'] == verified['native_correlation']
    observed, archived = (decoded_pack(path / 'reconstruction.zip') for path in (out, expected))
    assert observed == archived
    return dict(id=key, rgb_sha256=entry['rgb_sha256'], source_policy=incumbent['source_policy'],
                camera=actual['placement'].get('camera_source'), exact_projection=True, exact_decoded_export=True,
                placed=len(observed), decoded_sha256=inventory_hash(observed),
                expected_pack_sha256=digest(expected/'reconstruction.zip'), cli_pack_sha256=digest(out/'reconstruction.zip'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['camera','cli'])
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    control={r['id']:r for r in read(CONTROL/'independent-matched-cases.json.gz')}
    entries=list(frames(SOURCE,COHORTS))
    with ProcessPoolExecutor(args.workers) as pool:
        if args.stage=='camera':
            cases=list(pool.map(check_camera,(e for e in entries if not control[e['id']]['camera_usable'])))
        else:
            keys={'train-viaduct-02-riverstone-high-right','train-pillars-01-marble-high-right','test-castle-01-limestone-low-left',
                  'validation-harbor-02-seafoam-rear-oblique','test-castle-02-limestone-high-right'}
            cases=list(pool.map(check_cli,((e,control[e['id']]) for e in entries if e['id'] in keys)))
    save(REVIEW/('production-'+args.stage+'.json'),dict(cases=cases,files={name:digest(ROOT/'engine'/name) for name in
         ('camera_native_floor.py','camera_metric.py','camera_repeat_floor.py','camera_atlas_floor.py')}))
    print(args.stage, len(cases), 'matched', flush=True)
