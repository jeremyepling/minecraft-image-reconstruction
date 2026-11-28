"""Replay the frozen local-grid predictor on the paired screenshot cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from unittest.mock import patch

from benchmark.floating_build_local_grid import predict
from benchmark.full_block_lattice import atlas
from benchmark.expanded_baseline import read, save, digest
from benchmark.expanded_scores import evaluate_pack, pooled
from engine import surface_fields, surface_model, visible_placement
from engine.export import write_pack, JAVA_26_3_FORMAT
from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/floating-cloud-gate-01"


def build_handoff(source, cache, handoff):
    from engine.floating_sky import METHOD
    receipt = handoff / 'build-manifest.json'
    if receipt.exists():
        saved = read(receipt)
        if saved['image_sha256'] == digest(source) and saved['stages'].get('background_filter') == METHOD:
            return

    def field(image, kind):
        with np.load(cache / (kind + '.npz')) as data:
            return data['value']

    with patch.object(surface_fields, 'infer', field), patch.object(
            visible_placement, 'seam_camera', side_effect=AssertionError("No floor-camera calls permitted")):
        surface_model.build(source, '26.3', handoff, through='counts', boundary='wall-depth-anything')


def arm(e, variant, cache, out, handoff, native_fallback):
    spec = e['variants'][variant]
    source = Path(spec['image'])
    out.mkdir(parents=True, exist_ok=True)
    if digest(source) != spec['source_sha256']:
        raise ValueError("Source hash mismatch")
    build_handoff(source, cache, handoff)
    from benchmark.mobile_surface_evidence import build_local
    if (out/'case.json').is_file():
        with Image.open(source) as image:
            build_local(handoff, out, read(out/'case.json'), image)
        return
    parts = read(handoff/'surfaces.json')['planes']
    counts = read(handoff/'counts.json')
    image = Image.open(source).convert('RGB')
    attempts = []
    anchors,p,evidence = predict(image,parts,counts,native_fallback=native_fallback,attempts=attempts)
    cells = {tuple(a['cell']):a['block'] for a in anchors}
    pack = out/'reconstruction.zip'
    write_pack(pack,cells,JAVA_26_3_FORMAT)
    evidence.update(anchors=anchors, sheet_or_global_P=p.tolist() if p is not None else None,
                    source_sha256=spec['source_sha256'], cells=len(cells),
                    field_sha256={kind:digest(cache/(kind+'.npz')) for kind in ('normal','depth')})
    # Local sheets expose only a compass basis to the grader. Their free origin is deliberately arbitrary.
    camera = dict(P=evidence['sheet_or_global_P'],scale_status='supported' if p is not None else 'uncertain',
                  camera_source='build-local' if evidence['global_projection'] else 'sheet-direction-only')
    score,world = evaluate_pack(e,pack,camera,assignment={cell:'native atlas' for cell in cells})
    score.update(source_sha256=spec['source_sha256'],original_source_sha256=e['rgb_sha256'],
                 variant=variant,global_projection=evidence['global_projection'])
    if not evidence['global_projection']:
        score['camera_usable']=False
        score['camera_status']='local-sheet-only' if anchors else 'uncertain'
        score['registration'].update(absolute_applicable=False)
    draw=ImageDraw.Draw(image)
    for a in anchors:
        corners=[tuple(point) for point in a['quad']]
        draw.line(corners+corners[:1],fill='lime',width=3)
    image.save(out/'source.jpg',quality=88)
    save(out/'aligned-placements.json.gz',[[*cell,block] for cell,block in sorted(world.items())])
    record = dict(variant=variant, id=e['id'], evidence=evidence, score=score)
    build_local(handoff, out, record, image, attempts)
    save(out/'case.json', record)
    print(variant,e['id'],evidence['status'],len(cells),flush=True)


def summarize(cohort, cases, manifest, native_fallback):
    rows = [read(cohort/v/e['id']/'case.json') for e in cases for v in e['variants']]
    def group(records):
        report = pooled([r['score'] for r in records], len(records))
        report.pop('registration',None)
        report['local_only_nonempty'] = sum(r['evidence']['status']=='local_only_nonempty' for r in records)
        return report
    save(cohort/'summary.json',dict(
        fixture_manifest_sha256=digest(manifest), native_sheet_fallback=native_fallback,
        predictor_sha256=digest(Path(__file__).with_name('floating_build_local_grid.py')),
        atlas_sha256=atlas().provenance['atlas_sha256'],
        by_variant={v:group([r for r in rows if r['variant']==v]) for v in cases[0]['variants']},
        all_arms=group(rows), direction_registration='Source sheet tangent axes matched to compass by evaluator; single integer translation by whole-build overlap. Absolute location unscored.'))


def private_fields(source, source_sha, archived, private, frame_id=None):
    """Reuse a matching pinned inference receipt, or infer into this job's directory."""
    fingerprints = {kind: spec[2] for kind, spec in surface_fields.MODELS.items()}
    if digest(source) != source_sha:
        raise ValueError(f'Source hash mismatch: {source}')
    directory = archived/source_sha
    receipt = directory/'receipt.json'
    proof = read(receipt) if receipt.is_file() else None
    if proof is None and frame_id:
        donor = archived/frame_id/'receipt.json'
        if donor.is_file():
            candidate = read(donor)
            if candidate.get('source_sha256') == source_sha:
                receipt, directory, proof = donor, donor.parent, candidate
    if proof is not None:
        if (proof.get('source_sha256') != source_sha or proof.get('model_sha256') != fingerprints
                or proof.get('preprocessing') != surface_fields.SETTINGS
                or any(digest(directory/(kind+'.npz')) != proof['fields_sha256'][kind] for kind in fingerprints)):
            raise ValueError(f'Invalid frozen model field receipt: {receipt}')
        return directory
    private.mkdir(parents=True)
    hashes = {}
    with Image.open(source) as image:
        for kind in fingerprints:
            field = surface_fields.infer(image, kind)
            path = private/(kind+'.npz')
            np.savez_compressed(path, value=field)
            hashes[kind] = digest(path)
    save(private/'receipt.json', dict(source_sha256=source_sha, model_sha256=fingerprints,
                                      fields_sha256=hashes, preprocessing=surface_fields.SETTINGS))
    return private


def private_arm(e, variant, out, archived, native_fallback, handoffs_only):
    spec = e['variants'][variant]
    source = Path(spec['image'])
    cache = private_fields(source, spec['source_sha256'], archived, out/'fields', e['id'])
    handoff = out/'source-handoff'
    if handoffs_only:
        build_handoff(source, cache, handoff)
    else:
        arm(e, variant, cache, out, handoff, native_fallback)


def run_private(args, cases):
    if not cases:
        raise ValueError('No Floating cases matched the requested cohort and frame selection')
    out = args.out.resolve()
    shared = args.shared_root.resolve()
    archived = (args.baseline or shared/'fields').resolve()
    atlas_dir = Path(os.environ.get('MCIR_PREPARED_ATLAS_26_3', shared/'atlas')).resolve()
    inputs = [shared, archived, atlas_dir]
    if any(out.is_relative_to(path) or path.is_relative_to(out) for path in inputs):
        raise ValueError(f'Private output must be separate from inputs: {out}')
    # A fresh directory gives every invocation exclusive ownership, including across processes.
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Private output directory must be new or empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'fixtures.json'
    with manifest.open('x', encoding='utf-8') as stream:
        json.dump(dict(cases=cases), stream)
    os.environ.setdefault('MCIR_PREPARED_ATLAS_26_3', str(shared/'atlas'))
    jobs = [(e, variant) for e in cases for variant in e['variants']]
    # Active cohort IDs are unique. Preserve the case layout consumed by the review publisher.
    if len({(e['id'], variant) for e, variant in jobs}) != len(jobs):
        raise ValueError('Duplicate Floating case and variant in the selected cohort')
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(private_arm, e, variant, out/variant/e['id'], archived,
                               args.native_fallback, args.handoffs_only) for e, variant in jobs]
        for future in as_completed(futures):
            future.result()
    if not args.handoffs_only:
        summarize(out, cases, manifest, args.native_fallback)


def main():
    from training.floating_cohort import RETIRED_PALETTES, frame_index
    from training.floating_combined import BACKGROUND, PINNED, select_active_cases

    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('manifest', type=Path, nargs='?')
    source.add_argument('--shared-root', type=Path, help='Versioned Floating input bundle')
    parser.add_argument('--out', type=Path, help='New private run directory (required with --shared-root)')
    parser.add_argument('--baseline', type=Path, help='Read-only verified field directory for a private run')
    parser.add_argument('--cohort', choices=['visible', 'decorative'], help='Select an active shared cohort (default: visible)')
    parser.add_argument('--case', action='append', help='Select a shared frame ID (repeatable)')
    parser.add_argument('--handoffs-only', action='store_true', help='Build only the selected private source handoffs')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--native-fallback',action='store_true')
    parser.add_argument('--sky-handoffs-only', action='store_true', help='Build the floating Surface Finder and counter inputs for the six-view benchmark.')
    parser.add_argument('--include-decorative', action='store_true',
                        help='Include Decorative Builds v1 from the supplied fixture manifest.')
    parser.add_argument('--historical-cohort', action='store_true',
                        help='Replay archived layouts while skipping retired palettes.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.shared_root:
        if not args.out:
            parser.error('--shared-root requires --out')
        if args.historical_cohort or args.include_decorative or args.sky_handoffs_only:
            parser.error('Use --cohort and --handoffs-only with shared inputs')
        from training.floating_shared_inputs import cases as shared_cases
        cases = shared_cases(args.shared_root, cohort=args.cohort or 'visible', ids=args.case)
        run_private(args, list(cases))
        return
    if args.cohort or args.case or args.out or args.baseline or args.handoffs_only:
        parser.error('--cohort, --case, --out, --baseline and --handoffs-only require --shared-root')
    cohort = OUTPUT.parent/'floating-cloud-gate-native-01' if args.native_fallback else OUTPUT
    import time
    baseline = args.manifest.parent/'baseline'
    cases = [case for case in read(args.manifest)['cases'] if case.get('palette') not in RETIRED_PALETTES]
    if not args.include_decorative:
        cases = [case for case in cases if case.get('dataset') != 'decorative-builds-v1']
    if args.historical_cohort:
        indexed = frame_index()
        cases = [case for case in cases
                 if indexed.get((case.get('dataset'), case['id']), case).get('palette') not in RETIRED_PALETTES]
    else:
        known_datasets = PINNED.keys() | {BACKGROUND}
        indexed = [case for case in cases if case.get('dataset') in known_datasets]
        active = {(case['dataset'], case['id']) for case in select_active_cases(indexed)}
        cases = [case for case in cases if case.get('dataset') not in known_datasets
                 or (case['dataset'], case['id']) in active]
    if not cases:
        parser.exit(message="No active floating cases in this fixture manifest.\n")
    if args.sky_handoffs_only:
        with ProcessPoolExecutor(args.workers) as pool:
            jobs = []
            for e in cases:
                source = Path(e['variants']['sky']['image'])
                if digest(source) != e['variants']['sky']['source_sha256']:
                    raise ValueError(f'Source hash mismatch: {source}')
                jobs.append(pool.submit(build_handoff, source,
                                        baseline/'sky'/e['dataset']/e['id']/'handoffs',
                                        OUTPUT/'sky'/e['id']/'source-handoff'))
            for job in as_completed(jobs):
                job.result()
        return
    tasks = [(e,v,baseline,args.native_fallback) for e in cases for v in e['variants']
             if not (cohort/v/e['id']/'case.json').exists() or not (cohort/v/e['id']/'surfaces.json').exists()]
    pending = {}
    with ProcessPoolExecutor(args.workers) as pool:
        while tasks or pending:
            for task in tasks[:]:
                e,v,_,_ = task
                # receipt or planarity publication proves the per-image normal/depth writes finished.
                cache=baseline/v/e['dataset']/e['id']/'handoffs'
                if args.native_fallback and not (OUTPUT/v/e['id']/'case.json').exists():
                    continue
                if (cohort/v/e['id']/'case.json').exists() or ((cache/'mlsd.json.gz').exists() and (cache/'depth.npz').exists()):
                    pending[pool.submit(arm, e, v, cache, cohort/v/e['id'], OUTPUT/v/e['id']/'source-handoff', args.native_fallback)]=task
                    tasks.remove(task)
            for future in list(pending):
                if future.done():
                    future.result()
                    del pending[future]
            if tasks or pending:
                time.sleep(2)
    summarize(cohort,cases,args.manifest,args.native_fallback)

if __name__=='__main__':
    main()
