"""Choose between two exported wall proposals using their first visible pixels in the photo."""
import argparse
from pathlib import Path

import numpy as np

import benchmark
from benchmark import framed_depth_eval as frozen
from benchmark.mask_blocks import decoded_pack
from benchmark.texture_camera_eval import checked, read, save


def choose(image, parts, camera, original, alternative):
    from benchmark.wall_source_projection import FaceRenderer, Scene

    if original == alternative:
        return False, {'reason': 'identical'}
    old = Scene(image, parts, camera, original)
    new = Scene(image, parts, camera, alternative)
    # The per-render owner IDs differ when a cell is inserted. Compare actual cells.
    ids = {xyz: i for i, xyz in enumerate(sorted(original | alternative))}
    def owners(scene):
        table = np.array([ids[xyz] for xyz in scene.cells] + [-1], dtype=int)
        return table[scene.owner]

    changed = np.flatnonzero((owners(old) != owners(new)) | (old.axes != new.axes))
    record = {'changed_pixels': len(changed), 'added_cells': len(alternative - original)}
    if len(changed) < 48:
        return False, dict(record, reason='too_few_changed_first_hits')
    def errors(scene):
        foreground = scene.owner[changed] >= 0
        inside = scene.union[changed]
        axes = scene.masks[:, changed]
        oriented = axes.any(0)
        orientation = foreground & inside & oriented & ~scene.masks[scene.axes[changed], changed]
        return int((foreground != inside).sum()), int(orientation.sum())
    old_sil, old_orientation = errors(old)
    new_sil, new_orientation = errors(new)
    silhouette = old_sil - new_sil
    orientation = old_orientation - new_orientation
    # Held-out patches are identical for the two renderings. Use pixels for which both
    # first-hit faces have enough native texels to fit an atlas texture on the other patch.
    a, b = FaceRenderer(old), FaceRenderer(new)
    tested = changed[(old.owner[changed] >= 0) & (new.owner[changed] >= 0) & old.union[changed]
                     & a.test[changed] & a.fitted[changed] & b.fitted[changed]]
    before = float(abs(a.source[tested] - a.image[tested]).mean()) if len(tested) else None
    after = float(abs(b.source[tested] - b.image[tested]).mean()) if len(tested) else None
    supported = []
    for i, cell in enumerate(new.cells):
        if cell in original:
            continue
        visible = new.owned.get(i, np.zeros(0, int))
        supported.append(int((new.union[visible] & new.masks[new.axes[visible], visible]).sum()))
    gain = (silhouette + .65 * orientation) / len(changed)
    accept = (silhouette >= 0 and orientation >= 0 and gain >= .01 and len(tested) >= 24
              and before - after >= .20 and min(supported, default=4) >= 4)
    return bool(accept), dict(record, silhouette_gain=silhouette, orientation_gain=orientation,
                              geometry_gain_per_pixel=gain, heldout_texture_pixels=len(tested),
                              old_texture_mae=before, new_texture_mae=after,
                              added_cells_supported_pixels=supported,
                              reason='agrees' if accept else 'abstain')


def main(args):
    if args.projection_root:
        benchmark.__path__.append(str(args.projection_root.resolve() / 'benchmark'))
    source = args.original
    candidate = args.alternative
    def load_manifest(path):
        meta = read(path / 'summary.json')
        cases = read(checked(path / 'cases.json', meta['cases_sha256']))
        return {r['id']:r for r in cases},meta['cases_sha256']
    control, control_hash = load_manifest(source)
    changes, changes_hash = load_manifest(candidate)
    assert set(changes) == set(control) and len(control) == sum(frozen.SIZES.values())

    decisions = []
    for split, frame, inputs in frozen.cases(args.splits):
        key = frame['id']
        before, after = control[key], changes[key]
        assert before['pack_sha256'] == after['control_pack_sha256']
        assert before['rgb_sha256'] == after['rgb_sha256'] == inputs['rgb_sha256']
        assert before['input_hashes'] == after['input_hashes']
        # Identify identical candidates only from pack hashes. Benchmarks and
        # reference data never enter the selector.
        accept, status = False, {'reason':'identical'}
        if before['pack_sha256'] != after['pack_sha256']:
            image,parts,_,cam = frozen.photo(split,frame,inputs)
            def cells(root, row):
                path = checked(root/split/key/'reconstruction.zip', row['pack_sha256'])
                return set(decoded_pack(path))
            accept,status = choose(image, parts, cam, cells(source,before), cells(candidate,after))
        decisions.append(dict(split=split,id=key,accept=accept,**status))
    # Only now open the recorded score fields.
    totals = {}
    for group, splits in dict(development=('train',),fixed=frozen.EVALUATION,alternate=('extra',),new=('new',)).items():
        subset = [d for d in decisions if d['split'] in splits]
        if not subset:
            continue
        def counts(mode):
            fields = ('target','placed','correct','extras','hidden_extras')
            values = {k:0 for k in fields}
            for d in subset:
                use_change = mode == 'ungated' or (mode == 'gate' and d['accept'])
                row = (changes if use_change else control)[d['id']]['relative_geometry']
                for k in fields:
                    values[k] += row[k]
            values['f1'] = 2 * values['correct'] / (values['placed'] + values['target'])
            return values
        totals[group] = dict(frames=len(subset),changed=sum(d['reason']!='identical' for d in subset),
                             accepted=sum(d['accept'] for d in subset),
                             control=counts('control'),ungated=counts('ungated'),gate=counts('gate'))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    save(args.out,dict(original=control_hash,alternative=changes_hash,totals=totals,
                       decisions=[d for d in decisions if d['reason']!='identical']))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,default=Path('../wall-section-round/runs/control'))
    parser.add_argument('--alternative',type=Path,default=Path('runs/placement-selective'))
    parser.add_argument('--projection-root',type=Path)
    parser.add_argument('--splits',nargs='+',choices=frozen.SIZES,default=list(frozen.SIZES))
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
