"""Run and score the ordinary Expanded Split-Border build on pinned Floating captures."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import subprocess

import cv2

from benchmark.expanded_baseline import digest, read, save
from benchmark.expanded_scores import load_reference, pooled_typed, target_breakdown
from benchmark.mask_blocks import decoded_pack
from benchmark.mobile_surface_evidence import build_expanded
from benchmark.relative_face_assembly import axis_vector
from benchmark.relative_face_scoring import evaluate_relative_voxels, yaw_from_anchor_normal
from benchmark.typed_structures import overlap, pooled_scores
from engine.floating_fields import private_fields
from training.floating_front_release import LIGHTING
from training.floating_shared_inputs import COHORTS, DEFAULT_COHORT, cases


def score_export(entry, pack, assembly, *, reference=None):
    """Register from the visible anchor and charge unresolved exports as in Original V1."""
    reference = reference if reference is not None else load_reference(entry)
    if reference.hashes != entry['reference_sha256']:
        raise ValueError(f"Reference hash mismatch: {entry['id']}")
    blocks = decoded_pack(pack)
    if len(blocks) != assembly['size']:
        raise ValueError(f"Decoded pack count differs from assembly: {entry['id']}")
    assigned = {tuple(cell) for cell in assembly['assigned']}
    anchor = assembly['orientation']
    orientation = (yaw_from_anchor_normal(anchor['normal_camera'], axis_vector(anchor['local_normal_axis']),
                                          reference.pose) if blocks and anchor else None)
    quarter = orientation['yaw_quarters'] if orientation else None
    status = 'empty' if not blocks else 'unresolved' if quarter is None else 'resolved'
    result, aligned = evaluate_relative_voxels(
        blocks, reference, yaw_quarters=0 if not blocks else quarter, assigned=assigned)
    graded = result['fixed']
    if graded is None:
        geometry = {mode: overlap(0, visible, len(blocks)) for mode, visible in
                    (('strict', result['visible_truth']), ('facade', result['required_truth']))}
        scores = dict(geometry, material=dict(correct=0, positions=0, accuracy=None),
                      assigned=len(assigned), fallback=len(blocks)-len(assigned),
                      assigned_correct=0, assigned_positions=0)
        graded = dict(geometry=geometry, scores=scores, translation=None, targets=target_breakdown({}, reference))
    return dict(graded, registration_status=status, orientation=orientation,
                yaw_quarters=result['fixed']['yaw_quarters'] if result['fixed'] else None, pack_sha256=digest(pack),
                source_sha256=entry['variants']['original']['source_sha256'],
                reference_sha256=reference.hashes, exported=result['exported'],
                visible_truth=result['visible_truth'], required_truth=result['required_truth']), aligned


def run_case(job):
    from engine import floating

    cv2.setNumThreads(1)
    entry, root, archive = job
    marker = entry['id']
    source = Path(entry['variants']['original']['image'])
    source_sha = entry['variants']['original']['source_sha256']
    output = root / 'original' / marker
    output.mkdir(parents=True)
    cache = private_fields(source, source_sha, archive, output / 'fields', marker)
    assembly = floating.build(source, output, field_cache=cache)
    score, aligned = score_export(entry, output / 'reconstruction.zip', assembly)
    save(output / 'aligned-placements.json.gz', [[*cell, block] for cell, block in sorted(aligned.items())])
    handoff = output / 'source-handoff'
    manifest = read(handoff / 'build-manifest.json')
    counts = read(handoff / 'counts.json')['surfaces']
    sky = manifest['surface_finder']['boundary_settings']['background_filter']
    if manifest['image_sha256'] != source_sha:
        raise ValueError(f"Source handoff hash mismatch: {marker}")
    chart_file = output / 'charts.json'
    oriented_file = output / 'oriented-charts.json'
    stages = dict(masks_proposed=sky['input_count'], masks_kept=sky['kept_count'],
                  counted_surfaces=len(counts),
                  surfaces_with_both_dimensions=sum(
                      len(row['dimensions']['blocks']) == 2 and all(v is not None for v in row['dimensions']['blocks'])
                      for row in counts),
                  split_border_charts=len(read(chart_file)['charts']),
                  oriented_charts=len(read(oriented_file)['charts']),
                  selected_charts=len(assembly['selected']), recorded_joins=len(assembly['joins']))
    record = dict(id=marker, dataset=entry['dataset'], split=entry['split'], family=entry['family'],
                  lighting=entry.get('lighting'), dataset_manifest_sha256=entry['dataset_manifest_sha256'],
                  source_sha256=source_sha, field_sha256=read(cache / 'receipt.json')['fields_sha256'],
                  assembly_sha256=digest(output / 'assembly.json'), charts_sha256=digest(chart_file),
                  oriented_charts_sha256=digest(oriented_file),
                  score=score, stages=stages)
    build_expanded(output, source, record, assembly)
    save(output / 'case.json', record)
    print(marker, score['registration_status'], score['exported'], flush=True)
    return record


def summarize(rows):
    scores = [row['score'] for row in rows]
    statuses = Counter(score['registration_status'] for score in scores)
    stages = Counter()
    for row in rows:
        stages.update(row['stages'])
    return dict(tested=len(rows), empty_exports=statuses['empty'], unresolved_nonempty=statuses['unresolved'],
                oriented_nonempty=statuses['resolved'], exported=sum(score['exported'] for score in scores),
                visible_truth=sum(score['visible_truth'] for score in scores),
                required_truth=sum(score['required_truth'] for score in scores),
                geometry=pooled_scores([score['geometry'] for score in scores]),
                scores=pooled_typed([score['scores'] for score in scores]), stages=stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cohort', choices=COHORTS, default=DEFAULT_COHORT)
    parser.add_argument('--lighting', choices=LIGHTING)
    parser.add_argument('--release-sha256')
    parser.add_argument('--case', action='append')
    parser.add_argument('--baseline', type=Path, help='Verified field archive or earlier run directory')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    shared = args.shared_root.resolve()
    root = args.out.resolve()
    archive = (args.baseline or shared / 'fields').resolve()
    atlas_path = Path(os.environ.get('MCIR_PREPARED_ATLAS_26_3', shared / 'atlas')).resolve()
    if any(root.is_relative_to(path) or path.is_relative_to(root) for path in (shared, archive, atlas_path)):
        parser.error('The run directory must be separate from the fixture, field archive and atlas')
    if root.exists() and any(root.iterdir()):
        parser.error('Choose a new or empty run directory')
    entries = cases(shared, cohort=args.cohort, ids=args.case, lighting=args.lighting,
                    release_sha256=args.release_sha256)
    if not entries:
        parser.error('No matches for the selected cohort and captures')
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'fixtures.json'
    save(manifest, dict(cases=entries))
    os.environ['MCIR_PREPARED_ATLAS_26_3'] = str(atlas_path)
    jobs = [(entry, root, archive) for entry in entries]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_case, jobs))
    from benchmark.full_block_lattice import atlas
    from engine import floating

    checkout = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
    save(root / 'summary.json', dict(method='Expanded Split-Border', revision=revision, cohort=args.cohort,
                                    fixture_manifest_sha256=digest(manifest), predictor_sha256=digest(Path(floating.__file__)),
                                    atlas_sha256=atlas().provenance['atlas_sha256'], all_arms=summarize(rows),
                                    by_split={key: summarize([row for row in rows if row['split'] == key])
                                              for key in dict.fromkeys(row['split'] for row in rows)},
                                    by_lighting={key: summarize([row for row in rows if row['lighting'] == key])
                                                 for key in dict.fromkeys(row['lighting'] for row in rows) if key}))


if __name__ == '__main__':
    main()
