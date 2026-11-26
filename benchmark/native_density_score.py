"""Replay density-only postprocessing on hash-verified ordinary placement packs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark import visible_depth_native as native
from benchmark.camera_metrics import registration
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.native_sample_density import residual
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import DATA, ROOT, STABLE, checked, digest, frames, read
from benchmark.visible_depth_transfer import structure
from training.visible_labels import visible_labels


MODES = {'four64':(4,64), 'two64':(2,64), 'two16':(2,16),
         'visible16':(2,16), 'interior16':(2,16), 'visible64':(2,64),
         'interior4pitch':(4,16), 'interior_verified':(2,16), 'interior_verified4':(4,16)}
REPLAY = ROOT/'benchmark/reviews/relative-build-01/scores.zip'
FIELDS = ('target','placed','correct','extras','hidden_extras','gained','lost','extra_added','extra_removed')


def evaluate(job):
    split,f,score,modes,runs,out = job
    hashes, key = f['files'], f['id']
    if split in ('old','fresh'):
        fixture = runs / ('visible-depth-rounds/transfer' if split=='old' else 'canopy-placement/confirmation')
        parts = read(checked(fixture/'upstream'/key/'surfaces.json', score['surface_sha256']))['planes']
        camera = f['camera']
    else:
        source = read(checked(runs/'visible-depth-rounds/inputs'/(key+'.json'), score['input_sha256']))
        parts, camera = source['parts'],source['camera']
    if split in ('old','fresh','native'):
        stem = f.get('source_id',key)
        folder = native.BUILDS/f['directory']/'captures'
        rgb = folder/(stem+'.png')
        pose_file = folder/(stem+'.json')
        raw_file = folder/(stem+'-blocks.json')
    else:
        folder = DATA/'data'/split/f['directory']
        rgb,pose_file,raw_file = (folder/p for p in ('rgb.png','camera.json','world-readback.json'))
    frozen = score['methods']['current']
    baseline = set(decoded_pack(checked(runs/'canopy-placement/release'/split/key/'reconstruction.zip',
                                       frozen['pack_sha256'])))
    with Image.open(checked(rgb,hashes['rgb.png'])) as image:
        photo = image.convert('RGB')
    predictions = {}
    for mode in modes:
        start = perf_counter()
        cells = residual(photo,parts,camera,baseline,*MODES[mode],
                         depth_aware=mode.startswith(('visible','interior')), interior=mode.startswith('interior'),
                         confirm_axes=mode.startswith('interior_verified'))
        seconds = perf_counter()-start
        path = out/'packs'/split/key/(mode+'.zip')
        path.parent.mkdir(exist_ok=True,parents=True)
        predictions[mode] = (_write_read(path,cells),seconds,digest(path))
    # Read scoring truth only after every prediction is serialized.
    pose = read(checked(pose_file,hashes['camera.json']))
    raw = read(checked(raw_file,hashes['world-readback.json']))
    if split in ('old','fresh','native'):
        states = {tuple(b[k] for k in 'xyz'):b for b in raw}
        labels,ids,_ = visible_labels(pose,states,(0,0,0))
        occupied = set(structure(raw,64))
    else:
        labels = read(checked(folder/'labels.json', hashes['labels.json']))
        with np.load(checked(folder/'labels.npz',hashes['labels.npz'])) as raster:
            ids = raster['face_id']
        occupied = set(structure(raw,f['ground_y']))
    truth = {tuple(b['xyz_world']):b['block_id'] for b in labels['blocks']}
    sampled = set(np.unique(ids[2::4,2::4]))
    invisible_to_lattice = truth.keys()-{tuple(face['xyz_world']) for face in labels['faces'] if face['id'] in sampled}
    transform = registration(np.asarray(camera['P']),pose) if camera['P'] is not None else None
    before = registered(baseline,transform)
    row = dict(id=key, split=split, shape=f.get('layout', f.get('shape')),
               rgb_sha256=hashes['rgb.png'], baseline_pack=frozen['pack_sha256'],
               baseline={k:frozen['aligned'][k] for k in FIELDS[:5]}, methods={})
    for mode,(cells,seconds,pack_hash) in predictions.items():
        world = registered(cells, transform)
        scores = placement_scores(world,before,truth,occupied)
        assert list(scores['baseline_translation']) == frozen['offset']
        aligned = scores['relative_geometry']
        assert aligned['correct']-aligned['gained']+aligned['lost'] == frozen['aligned']['correct']
        result = {k:aligned[k] for k in FIELDS}
        dx,dy,dz = scores['translation']
        shifted = {(x+dx,y+dy,z+dz) for x,y,z in world}
        assert frozen['offset'] == [0,0,0]
        result.update(unsampled_gained=len((shifted-before)&invisible_to_lattice))
        row['methods'][mode] = dict(metrics=result,seconds=seconds,pack_sha256=pack_hash)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sets', nargs='+', required=True)
    p.add_argument('--modes', nargs='+', choices=MODES, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int,default=3)
    p.add_argument('--diagnostic-train',action='store_true',help='Two preregistered views of layout 1 in each training family.')
    args = p.parse_args()
    with ZipFile(REPLAY) as z:
        scores = {(r['split'],r['id']):r for r in json.loads(z.read('scores.json'))['cases']}
    runs = STABLE/'runs'
    rows = []
    for split in args.sets:
        if split in ('old','fresh'):
            fixture = runs/('visible-depth-rounds/transfer' if split=='old' else 'canopy-placement/confirmation')
            sample = read(fixture/'screen.json')['selected']
        else:
            sample = native.rows() if split=='native' else frames(DATA,split)
        if args.diagnostic_train:
            assert split=='train'
            sample = [f for f in sample if f['layout'].endswith('1') and
                      f['id'].endswith(('cool-high-right','earth-low-left'))]
        jobs = [(split, f, scores[split,f['id']], args.modes, runs, args.out) for f in sample]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            result = list(pool.map(evaluate,jobs))
        rows.extend(result)
        for mode in args.modes:
            total = sum((Counter(r['methods'][mode]['metrics']) for r in result),Counter())
            print(split,mode,len(result),dict(total),flush=True)
    report = dict(baseline_replay_sha256=digest(REPLAY),source_sha256=digest(Path(__file__).with_name('native_sample_density.py')),
                  modes={m:MODES[m] for m in args.modes}, cases=rows)
    (args.out/'scores.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
