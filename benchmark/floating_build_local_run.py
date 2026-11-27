"""Replay the frozen local-grid predictor on the paired screenshot cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor
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
OUTPUT = ROOT / "runs/floating-build-grid-01"


def arm(task):
    e, variant, baseline, native_fallback = task
    spec = e['variants'][variant]
    source = Path(spec['image'])
    cohort = OUTPUT.parent/'floating-build-grid-native-01' if native_fallback else OUTPUT
    out = cohort / variant / e['id']
    out.mkdir(parents=True, exist_ok=True)
    cache = baseline / variant / e['dataset'] / e['id'] / 'handoffs'
    if digest(source) != spec['source_sha256']:
        raise ValueError("Source hash mismatch")
    def field(image,kind):
        with np.load(cache/(kind+'.npz')) as data:
            return data['value']
    handoff = OUTPUT / variant / e['id'] / 'source-handoff'
    if not (handoff / 'build-manifest.json').exists():
        with patch.object(surface_fields, 'infer', field), patch.object(
                visible_placement, 'seam_camera', side_effect=AssertionError("No floor-camera calls permitted")):
            surface_model.build(source,'26.3',handoff,through='counts',boundary='wall-depth-anything')
    parts = read(handoff/'surfaces.json')['planes']
    counts = read(handoff/'counts.json')
    image = Image.open(source).convert('RGB')
    anchors,p,evidence = predict(image,parts,counts,native_fallback=native_fallback)
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
    save(out/'case.json',dict(variant=variant,id=e['id'],evidence=evidence,score=score))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--native-fallback',action='store_true')
    args = parser.parse_args()
    cohort = OUTPUT.parent/'floating-build-grid-native-01' if args.native_fallback else OUTPUT
    import time
    baseline = args.manifest.parent/'baseline'
    cases = read(args.manifest)['cases']
    tasks = [(e,v,baseline,args.native_fallback) for e in cases for v in e['variants']
             if not (cohort/v/e['id']/'case.json').exists()]
    pending = {}
    with ProcessPoolExecutor(args.workers) as pool:
        while tasks or pending:
            for task in tasks[:]:
                e,v,_,_ = task
                # receipt or planarity publication proves the per-image normal/depth writes finished.
                cache=baseline/v/e['dataset']/e['id']/'handoffs'
                if args.native_fallback and not (OUTPUT/v/e['id']/'case.json').exists():
                    continue
                if (cache/'mlsd.json.gz').exists() and (cache/'depth.npz').exists():
                    pending[pool.submit(arm,task)]=task
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
