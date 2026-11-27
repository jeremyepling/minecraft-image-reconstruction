"""Pilot the frozen common-wall-corner placement intervention on actual exports."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from io import BytesIO
import os
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

os.environ.update(MCIR_IMAGE_ONLY='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

import numpy as np
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.camera_horizon_replay import replay
from benchmark.camera_model_controls import DEFAULT_BUILDS, REPEAT_BUILDS
from benchmark.camera_model_entry_check import MATCHED
from benchmark.camera_native_recovery_grade import NATIVE_RUN, policy_pack
from benchmark.expanded_baseline import COHORTS, ROOT, SOURCE, frames, read, save, digest
from benchmark.expanded_scores import evaluate_pack, load_reference
from benchmark.mask_blocks import decoded_pack
from benchmark.visible_wall_corner import recommend
from engine import foreground_conflicts as fg, visible_placement as vp

DEV = ['train-amphitheater-02-ashlar-high-right', 'train-viaduct-02-riverstone-high-right',
       'train-shrine-01-ochre-low-left', 'train-observatory-02-seafoam-low-left']
HOLD = ['validation-harbor-02-kiln-high-right','validation-harbor-02-seafoam-high-right',
        'validation-harbor-03-seafoam-high-right','test-castle-01-ashlar-high-right',
        'test-castle-02-ashlar-rear-oblique','heldout-facade-oblique-left',
        'inverse-voxels-26.3--confirm--test-colonnade-01-oblique-right',
        'train-market-court-1-cool-high-right','test-walled-garden-1-warm-high-right']
OUT = ROOT / 'runs/visible-wall-corner-01'


class NoAlternative(Exception):
    pass


def intervention(stack, report, view):
    context = {}
    estimate = vp._estimate_plane_cells

    def capture(image, parts, masks, projection, *args, **kwargs):
        context.update(projection=projection, union=np.logical_or.reduce(masks))
        return estimate(image, parts, masks, projection, *args, **kwargs)

    class Align(fg.ForegroundConflicts):
        def __init__(self):
            super().__init__()
            self.source_rows = {}

        def record(self, key, mask, eye, rays, axis, baseline, rows, chosen, weights):
            super().record(key, mask, eye, rays, axis, baseline, rows, chosen, weights)
            self.source_rows[key] = {r['level']: r for r in rows}

        def resolve(self):
            levels = super().resolve()
            if 'audit' not in report:
                changes, audit = recommend(self, levels, **context)
                report.update(audit=audit, prior={k:levels[k] for k in changes})
                save(OUT/view/'selection.json', report)
                if not changes:
                    raise NoAlternative()
            for key, chosen in report['audit']['changes'].items():
                if levels.get(key) == report['prior'][key]:
                    levels[key] = chosen
            self.levels.update(levels)
            return levels

    stack.enter_context(patch.object(fg, 'ForegroundConflicts', Align))
    stack.enter_context(patch.object(vp, '_estimate_plane_cells', capture))


def run(entry):
    key, dataset = entry['id'], entry['dataset']
    out = OUT / key
    base = next(r for r in read(ROOT/'benchmark/reviews/camera-native-recovery-controls-01/independent-default-cases.json.gz') if r['id']==key)
    note = dict(id=key, rgb_sha256=entry['rgb_sha256'], incumbent_sha256=base['pack_sha256'],
                method_sha256=digest(ROOT/'benchmark/visible_wall_corner.py'))
    print('START', key, flush=True)
    if dataset != 'fixed-104':
        field = DEFAULT_BUILDS / dataset / key
        if base['native_floor_coverage']=='candidate_export':
            pack = NATIVE_RUN/'builds'/dataset/key
            camera = read(NATIVE_RUN/'boundary/metric'/f'{key}.json')['camera']
        else:
            pack = policy_pack(entry, base)
            policy = base['source_policy']
            if policy == 'horizon-repeat-floor':
                camera = read(REPEAT_BUILDS/key/'repeat.json')['camera']
            elif policy == 'boundary-budget-repeat':
                camera = read(MATCHED.parent/'metric'/f'{key}.json')['camera']
            else:
                camera = read(field/'handoffs/placement-input.json.gz')['camera']
        status = read(pack/'build-manifest.json')['placement']
        assert camera['P'] == status['P'] and digest(pack/'reconstruction.zip') == base['pack_sha256']
    try:
        with ExitStack() as stack:
            intervention(stack, note, key)
            if dataset == 'fixed-104':
                inputs = fixed.inference_inputs(entry)
                result = fixed.replay_placement(inputs)
                out.mkdir(parents=True, exist_ok=True)
                (out/'reconstruction.zip').write_bytes(result.pack)
                case, pred = fixed.score_result(entry, result)
            else:
                case = replay(entry, camera, field, out/'build')
                _, pred = evaluate_pack(entry, out/'build/reconstruction.zip', status)
    except NoAlternative:
        note.update(kept=True)
    else:
        if dataset == 'fixed-104':
            with ZipFile(fixed._release()[2]) as archive:
                payload = archive.read(key+'/reconstruction.zip')
            local = decoded_pack(BytesIO(payload))
            _, prior = fixed.score_result(entry, fixed.ReplayResult(payload,dict.fromkeys(local,'fallback'),inputs.camera))
            truth = {tuple(b['xyz_world']) for b in fixed.fixed_reference(entry)[0]['blocks']}
        else:
            _, prior = evaluate_pack(entry, pack/'reconstruction.zip', status)
            truth = load_reference(entry).truth.keys()
        assert len(prior.keys() & truth) == base['geometry']['strict']['correct']
        note.update(kept=False, case=case, incumbent_geometry=base['geometry'], incumbent_scores=base['scores'],
                    recovered=len((pred.keys()-prior.keys()) & truth),lost=len((prior.keys()-pred.keys()) & truth),
                    extra_added=len((pred.keys()-prior.keys())-truth),extra_removed=len((prior.keys()-pred.keys())-truth))
    save(out/'result.json', note)
    print('DONE', key, note.get('kept'), flush=True)
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('group', choices=['dev','hold'])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--ids', nargs='+')
    args = ap.parse_args()
    keys = args.ids or (DEV if args.group=='dev' else HOLD)
    entries = {e['id']:e for e in frames(SOURCE, COHORTS)}
    if any(k not in entries for k in keys):
        entries.update((e['id'],e) for e in fixed.fixed_frames())
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run,[entries[k] for k in keys]))


if __name__=='__main__':
    main()

