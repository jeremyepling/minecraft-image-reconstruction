"""Replay final corner reprojection on the 204 verified placement exports."""
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

from benchmark import current_placement_viewer as viewer, facade_inset_round as suite
from benchmark import pavement_opening_eval as replay, smooth_joint_placement as previous, smooth_local_release as release
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement_run import ref_inputs
from engine.corner_reprojection import proposals, recheck

OUT = viewer.ROOT/'courtyard-corner-01'
BASE = viewer.ROOT/'window-wall-01'


def main():
    summary, history, payloads = viewer.read_releases([viewer.RELEASE, viewer.ROOT/'smooth-local-placement-01',
                                                       viewer.ROOT/'smooth-joint-placement-01', BASE])
    controls = {row['id']:row for row in read(checked(BASE/'cases.json', summary['cases_sha256']))}
    stages = {r['id']:r for r in read(viewer.ROOT/'facade-inset-round-01/combined.json')['rows']}
    rows = []
    OUT.mkdir(exist_ok=True)
    archive = OUT/'default-exports.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as zipped:
        for job in suite.jobs([*suite.framed.SIZES, 'shelter']):
            split, frame, _, _ = job
            key = frame['id']
            control = controls[key]
            assert control['source_hash'] == frame['files']['rgb.png']
            old_hash = control.get('pack_sha256', control['previous_pack_sha256'])
            if key in payloads:
                assert history[key][-1]['pack_sha256'] == old_hash
                cells = set(decoded_pack(BytesIO(payloads[key])))
            else:
                actual_hash, cells = previous.old_pack(job)
                assert actual_hash == old_hash
            candidates = proposals(cells)
            row = dict(id=key, split=split, source_hash=control['source_hash'], previous_pack_sha256=old_hash,
                       baseline=control['result'], result=control['result'], candidates=len(candidates))
            rows.append(row)
            if len(candidates) != 1:
                continue
            image, parts, _, camera = viewer.latest_stages(job, stages[key])
            now, decision = recheck(image, parts, camera, cells)
            if decision is None:
                continue
            # Compare visible positions only after fixing the new block set.
            path = suite.ROOT/'runs/courtyard-face-reprojection'/key/'reconstruction.zip'
            path.parent.mkdir(parents=True, exist_ok=True)
            packed = _write_read(path, now)
            assert set(packed) == now
            truth, occupied, transform, _ = ref_inputs(job)
            labels, ids = replay.visible_faces(split, frame)
            optional = horizontal_slivers(labels, ids)
            raw, prior = registered(now, transform), registered(cells, transform)
            scores = placement_scores(raw, prior, dict.fromkeys(truth), occupied)
            shifted = suite.framed.spacing.shift(raw, scores['translation'])
            original = suite.framed.spacing.shift(prior, scores['baseline_translation'])
            assert replay.priority(original, truth, optional) == row['baseline']['facade']
            relative = scores['relative_geometry']
            row.update(decision, pack_sha256=digest(path), result=dict(
                strict={k:relative[k] for k in replay.FIELDS},
                facade=replay.priority(shifted, truth, optional),
                registered={k:scores['metrics'][k] for k in replay.FIELDS}, translation=scores['translation']))
            row['block_deltas'] = {k:relative[k] for k in ('gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')}
            zipped.write(path, key+'/reconstruction.zip')
    assert len(rows) == len(controls) == 204
    before, after = (release.summarize(rows, member) for member in ('baseline', 'result'))
    assert before == summary['cohorts']
    fixed = after['fixed']['strict']
    historical = 2*fixed['correct']/(2*fixed['correct']+fixed['missed']+fixed['extras']+384)
    save(OUT/'cases.json', rows)
    save(OUT/'summary.json', dict(baseline='deba62a', frames=len(rows), changed=[r for r in rows if 'pack_sha256' in r],
                                  cohorts=after, previous_cohorts=before, archive_sha256=digest(archive),
                                  cases_sha256=digest(OUT/'cases.json'), historical_112_strict_f1=historical))
    print(after['fixed'])


if __name__ == '__main__':
    main()
