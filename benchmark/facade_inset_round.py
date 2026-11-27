"""Grade inset candidates against the ordinary c589f1a Surface Finder and placer."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
import importlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark import framed_depth_eval as framed
from benchmark.coplanar_fragment_report import index
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import ROOT, STABLE, checked, digest, read, save
from benchmark.wall_texture_evaluate import metrics
from engine import atlas_plane_depth, surface_model, texture_guided_count
from training.visible_labels import visible_labels


RELEASE = ROOT/'benchmark/reviews/strip-orientation-01/combined-entry.json'
ARCHIVE = ROOT/'benchmark/reviews/visible-error-budget-01/comparisons.zip'
WORKTREES = framed.EXTRA_INPUTS.parents[2]
CONTROL = WORKTREES/'visible-error-budget/runs/primary-production'
SHELTER = WORKTREES/'framed-depth-transfer/runs/transfer-input'
ORDINARY = ROOT/'benchmark/reviews/facade-inset-round-01/baseline-stages.zip'
ATLAS = STABLE/'runs/26.3/atlas'
FOCUS = 'heldout-facade-oblique-right'


def ordinary_inputs(split, frame, prior, control):
    if split == 'shelter':
        prefix = SHELTER/frame['directory']/'captures'/frame['source_id']
        with Image.open(checked(prefix.with_suffix('.png'), frame['files']['rgb.png'])) as src:
            image = src.convert('RGB')
        stage = SHELTER/'upstream'/frame['id']
        parts = read(checked(stage/'surfaces.json', control['surface_sha256']))['planes']
        counts = read(checked(stage/'counts.json', control['counts_sha256']))
        camera = frame['camera']
    else:
        image, parts, counts, camera = framed.photo(split, frame, prior)
    # Only two inputs changed since the frozen upstream releases. The normal
    # standalone and through-placement stage artifacts are identical on both.
    for entry in read(RELEASE)['examples']:
        if frame['files']['rgb.png'] == entry['rgb']:
            with ZipFile(ORDINARY) as z:
                def artifact(filename):
                    data = z.read(entry['example']+'/'+filename)
                    assert sha256(data).hexdigest() == entry['hashes'][filename]
                    return json.loads(data)
                parts, counts = artifact('surfaces.json')['planes'], artifact('counts.json')
    return image, parts, counts, camera


def jobs(splits):
    stable = index(CONTROL)
    for split, frame, prior in framed.cases([s for s in splits if s != 'shelter']):
        yield split, frame, prior, stable[frame['id']]
    if 'shelter' in splits:
        with ZipFile(checked(ARCHIVE, read(ARCHIVE.with_name('results.json'))['archive_sha256'])) as archive:
            controls = {r['id']:r for r in json.loads(archive.read('production/transfer/cases.json'))}
        for frame in read(SHELTER/'screen.json')['selected']:
            yield 'shelter', frame, {}, controls[frame['id']]


def visible_map(split, frame, *, metadata=False):
    files = frame['files']
    if split not in ('shelter', 'new', 'extra', 'old', 'fresh', 'native'):
        base = framed.TRAINING_DATA/'data'/split/frame['directory']
        labels = read(checked(base/'labels.json', files['labels.json']))
        with np.load(checked(base/'labels.npz', files['labels.npz'])) as arr:
            rays = arr['face_id'].copy()
    else:
        if split == 'native':
            rgb = framed.native.rgb_path(frame, framed.native.BUILDS)
        else:
            source = SHELTER if split == 'shelter' else framed.NEW_INPUTS if split == 'new' else framed.transfer.BUILDS
            rgb = source/frame['directory']/'captures'/(frame['source_id'] + '.png')
        pose = read(checked(rgb.with_suffix('.json'), files['camera.json']))
        states = read(checked(rgb.with_name(rgb.stem+'-blocks.json'), files['world-readback.json']))
        labels, rays, _ = visible_labels(pose, {tuple(r[k] for k in 'xyz'):r for r in states}, (0,0,0))
    lut = np.zeros(len(labels['faces'])+1, dtype='int32')
    for face in labels['faces']:
        lut[face['id']] = face['surface_id']
    return (labels, lut[rays]) if metadata else lut[rays]


def surface_stats(parts, truth, size):
    data = metrics(parts, truth, size)
    sizes = np.bincount(truth.ravel())
    best = {}
    for sid in range(1, len(sizes)):
        candidates = []
        for part in data['parts']:
            overlap = part['intersections'].get(sid, 0)
            iou = overlap / max(1, part['pixels'] + int(sizes[sid]) - overlap)
            candidates.append((iou, overlap, part['id']))
        iou, overlap, key = max(candidates, default=(0., 0, None))
        best[sid] = dict(iou=round(iou, 6), overlap=overlap, reference_px=int(sizes[sid]), mask=key)
    summary = {key:data[key] for key in ('matched','predicted','target','paired_pixels','unmatched_predictions','background_pixels')}
    return dict(**summary, best=best)


def build(image, parts, counts, camera):
    atlas = None
    if 'atlas_phase' in counts:
        from engine.atlas_count import load_atlas
        atlas = load_atlas(ATLAS, '26.3')
    with TemporaryDirectory() as work:
        measured = surface_model._count_regions(image, parts, '26.3', texture_guided_count.METHOD, atlas, Path(work))
    cells, _ = atlas_plane_depth.place_ranked(image, '26.3', parts, measured, camera, atlas_path=ATLAS)
    dimensions = {r['id']:r['dimensions']['blocks'] for r in measured['surfaces']}
    return set(map(tuple,cells)), dimensions


def geometry(split, frame, control, image, parts, revised, counts, camera):
    if split == 'shelter':
        with ZipFile(ARCHIVE) as archive:
            packed = archive.read('production/transfer/'+frame['id']+'/reconstruction.zip')
        assert sha256(packed).hexdigest() == control['pack_sha256']
        original = set(decoded_pack(BytesIO(packed)))
        root, inputs, ref = SHELTER, SHELTER, 'old'
    else:
        original = set(decoded_pack(checked(CONTROL/split/frame['id']/'reconstruction.zip', control['pack_sha256'])))
        root = framed.NEW_INPUTS if split == 'new' else framed.transfer.BUILDS
        inputs = framed.NATIVE_INPUTS.get(split)
        ref = 'old' if split in ('extra','new') else split
    baseline, old_dims = build(image, parts, counts, camera)
    changed, new_dims = build(image, revised, counts, camera)
    truth, occupied, transform, _ = framed.references(ref, frame, inputs, framed.TRAINING_DATA, framed.DEPTH,
        dict(native=framed.native.BUILDS, old=root, fresh=root))
    result = placement_scores(registered(changed,transform), registered(baseline,transform), dict.fromkeys(truth), occupied)
    return dict(result['relative_geometry'], baseline_reproduced=baseline==original,
                baseline_placed=len(baseline), original_dims=old_dims, new_dims=new_dims)


def initialize(candidate_root, qualified):
    global refine
    if not qualified:
        refine = None
        return
    import sys
    module, function = qualified.split(':')
    sys.path.insert(0, str(candidate_root))
    for name in ('engine','benchmark'):
        package = importlib.import_module(name)
        package.__path__ = [str(candidate_root/name), *package.__path__]
    refine = getattr(importlib.import_module(module), function)


def inspect(job):
    split, frame, prior, control, out, with_geometry = job
    im, parts, counts, camera = ordinary_inputs(split, frame, prior, control)
    after, events = (parts, []) if refine is None else refine(im, deepcopy(parts), camera)
    if len({p['id'] for p in after}) != len(after):
        raise ValueError('Duplicate surface IDs on ' + frame['id'])
    changed = parts != after
    result = dict(id=frame['id'], split=split, rgb_sha256=frame['files']['rgb.png'], changed=changed, events=events)
    if not changed and frame['id'] != FOCUS:
        return result
    truth = visible_map(split, frame)
    result.update(before=surface_stats(parts, truth, im.size), after=surface_stats(after, truth, im.size))
    if changed:
        directory = out/split/frame['id']
        directory.mkdir(parents=True, exist_ok=True)
        save(directory/'surfaces.json', dict(planes=after, image_size=list(im.size)))
        result['surfaces_sha256'] = digest(directory/'surfaces.json')
        if with_geometry:
            result['geometry'] = geometry(split, frame, control, im, parts, after, counts, camera)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path, default=ROOT)
    parser.add_argument('--candidate', help='module:refine')
    parser.add_argument('--splits', nargs='+', default=[*framed.SIZES,'shelter'])
    parser.add_argument('--key')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--geometry', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    work = [(*j, args.out, args.geometry) for j in jobs(args.splits) if not args.key or j[1]['id']==args.key]
    if not work:
        parser.error('no images selected')
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(args.candidate_root.resolve(),args.candidate)) as pool:
        rows = list(pool.map(inspect, work))
    save(args.out/'cases.json',rows)
    changed = [r for r in rows if r['changed']]
    summary = dict(method=args.candidate, baseline='c589f1a', frames=len(rows), changed=len(changed),
                   changed_ids=[r['id'] for r in changed], cases_sha256=digest(args.out/'cases.json'))
    save(args.out/'summary.json',summary)
    print(summary)


if __name__ == '__main__':
    main()
