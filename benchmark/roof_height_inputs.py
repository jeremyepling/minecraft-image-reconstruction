"""Prepare four sealed RGB inputs and replay ordinary build entry points in isolated outputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

import numpy as np

from benchmark import blind_shape_inputs
from benchmark.camera_metrics import score as camera_score
from benchmark.roof_height_fixture import ANGLES, PALETTES
from benchmark.texture_camera_eval import ROOT, STABLE, checked, digest, read, save
from training.framing import measure_framing

RECIPE = '5bef42645b9871f30f495d42473c4c59514f1f68'
ATLAS = STABLE / 'runs/26.3/atlas'
FILES = ('build-manifest.json', 'surfaces.json', 'counts.json', 'reconstruction.zip')


def build(root, frame):
    key = frame['id']
    source = checked(root/'scoring/captures'/(key+'.png'), frame['files']['rgb.png'])
    dest = root/'upstream'/key
    dest.mkdir(exist_ok=True, parents=True)
    with (dest/'build.log').open('w', encoding='utf8') as log:
        subprocess.run(['pwsh', '-NoProfile', '-File', str(ROOT/'Invoke-WallRecovery.ps1'), str(source),
                        '-Through', 'placement', '-Atlas', str(ATLAS), '-Out', str(dest)],
                       stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=True)
    am = read(dest/'build-manifest.json')
    assert am['through'] == 'placement' and am['image_sha256'] == frame['files']['rgb.png']
    return dict(id=key, files={name:digest(dest/name) for name in FILES})


def prepare(root, evidence):
    blind_shape_inputs.prepare(root, PALETTES, ANGLES, RECIPE, ATLAS)
    screen = read(root/'screen.json')
    margins = []
    for f in screen['selected']:
        key = f['id']
        base = root/'scoring/captures'
        pose = read(checked(base/(key+'.json'), f['files']['camera.json']))
        raw = read(checked(base/(key+'-blocks.json'), f['files']['world-readback.json']))
        with np.load(checked(root/'truth'/key/'labels.npz', f['files']['labels.npz'])) as labels:
            framing = measure_framing(pose, raw, f['ground_y'], labels['face_id'])
        faces = read(root/'native/labels'/(key+'.json'))['faces']
        cam = camera_score(dict(f['camera'], planes=[]), pose, faces)
        margins.append(dict(id=key, framing=framing, camera=cam))
        f['target_block_y'] = {'lowawning':3, 'highwalkroof':8}[f['family']]
    save(evidence/'framing.json', margins)
    save(evidence/'selection.json', screen)
    print([(r['id'],r['framing']['accepted']) for r in margins],flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare','build','package'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root, evidence = args.root.resolve(), args.out.resolve()
    evidence.mkdir(exist_ok=True, parents=True)
    if args.stage == 'prepare':
        prepare(root,evidence)
    elif args.stage == 'build':
        frames=read(evidence/'selection.json')['selected']
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows=list(pool.map(lambda frame: build(root,frame),frames))
        save(root/'builds.json',dict(rows=rows,revision='54c9fee',
                                     selection_sha256=digest(evidence/'selection.json')))
    else:
        from zipfile import ZipFile, ZIP_DEFLATED
        archive=root.parent/'roof-height-confirmation-v1.zip'
        blind_shape_inputs.package(root,archive)
        with ZipFile(archive,'a',ZIP_DEFLATED) as zipf:
            zipf.write(root/'builds.json','builds.json')
            for f in read(root/'builds.json')['rows']:
                path=root/'upstream'/f['id']/'reconstruction.zip'
                zipf.write(path,path.relative_to(root).as_posix())
        with ZipFile(archive) as zipf:
            assert zipf.testzip() is None and len(zipf.namelist())==58
        print(digest(archive),archive.stat().st_size)
