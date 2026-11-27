"""Grade the exterior-connected foreground refinement on the frozen 204 source images."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

from benchmark import current_placement_viewer as viewer, pavement_opening_eval as replay
from benchmark import smooth_joint_placement as previous, smooth_local_release as release
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement_run import ref_inputs
from engine import exterior_foreground
from engine.pillar_seams import native_mask

suite = viewer.suite
OUT = viewer.ROOT/'workshop-foreground-01'
BASE = viewer.ROOT/'counted-ground-corner-01'
RUN = suite.ROOT/'runs/workshop-foreground-support/packs'


@lru_cache(maxsize=1)
def controls():
    summary, history, payloads = viewer.read_releases([viewer.RELEASE, viewer.ROOT/'smooth-local-placement-01',
                     viewer.ROOT/'smooth-joint-placement-01', viewer.ROOT/'window-wall-01', BASE])
    rows = {r['id']:r for r in read(checked(BASE/'cases.json', summary['cases_sha256']))}
    stages = {r['id']:r for r in read(viewer.ROOT/'facade-inset-round-01/combined.json')['rows']}
    return history,payloads,rows,stages


def grade(job):
    split,frame,_,_=job
    key=frame['id']
    history,payloads,rows,stages=controls()
    control=rows[key]
    old_hash=control.get('pack_sha256',control['previous_pack_sha256'])
    if key in payloads:
        assert history[key][-1]['pack_sha256']==old_hash
        cells=set(decoded_pack(BytesIO(payloads[key])))
    else:
        actual_hash,cells=previous.old_pack(job)
        assert actual_hash==old_hash
    assert frame['files']['rgb.png']==control['source_hash']
    image,parts,_,camera=viewer.latest_stages(job,stages[key])
    row=dict(id=key,split=split,source_hash=control['source_hash'],previous_pack_sha256=old_hash,
             baseline=control['result'],result=control['result'],surface=[])
    if not exterior_foreground.supported_camera(camera):
        return row
    bg=exterior_foreground.exterior(image)
    revised,events=exterior_foreground.refine(image,parts,camera,bg)
    status=job[3]['status']
    add,remove=set(),set()
    if status.get('foreground_depth',{}).get('method')=='joint':
        remove=exterior_foreground.misplaced(image,revised,status,cells,bg)
        row['exterior_only_removed']=len(remove)
        add,shelf=exterior_foreground.roof_edits(image,parts,revised,events,status,cells,bg)
        remove |= shelf
    if events:
        truth=suite.visible_map(split,frame)
        byid={p['id']:p for p in parts}
        for event in events:
            pixels=native_mask(byid[event['id']],image.size)&bg
            assert int(pixels.sum())==event['trimmed']
            row['surface'].append(dict(event,structure=int((pixels&(truth>0)).sum())))
    if not (add or remove):
        return row
    path=RUN/key/'reconstruction.zip'
    path.parent.mkdir(parents=True,exist_ok=True)
    packed=_write_read(path,(cells-remove)|add)
    assert set(packed)==(cells-remove)|add
    truth,occupied,t,_=ref_inputs(job)
    now,prior=registered(packed,t),registered(cells,t)
    score=placement_scores(now,prior,dict.fromkeys(truth),occupied)
    optional=horizontal_slivers(*replay.visible_faces(split,frame))
    aligned=suite.framed.spacing.shift(now,score['translation'])
    assert replay.priority(suite.framed.spacing.shift(prior,score['baseline_translation']),truth,optional)==row['baseline']['facade']
    rel=score['relative_geometry']
    row.update(removed=sorted(remove),added=sorted(add),pack_sha256=digest(path),
               block_deltas={k:rel[k] for k in ('gained','lost','extra_removed','extra_added','hidden_added','hidden_removed')},
               result=dict(strict={k:rel[k] for k in replay.FIELDS},registered={k:score['metrics'][k] for k in replay.FIELDS},
                           facade=replay.priority(aligned,truth,optional),translation=score['translation']))
    return row


def main():
    jobs=list(suite.jobs([*suite.framed.SIZES,'shelter']))
    with ProcessPoolExecutor(5) as pool:
        rows=list(pool.map(grade,jobs))
    assert len(rows)==204
    before,after=[release.summarize(rows,key) for key in ('baseline','result')]
    assert before==read(BASE/'summary.json')['cohorts']
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'cases.json',rows)
    changed=[r for r in rows if 'pack_sha256' in r]
    archive=OUT/'default-exports.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for row in changed:
            key=row['id'];z.write(RUN/key/'reconstruction.zip',key+'/reconstruction.zip')
    fixed=after['fixed']['strict']
    historical=2*fixed['correct']/(2*fixed['correct']+fixed['missed']+fixed['extras']+384)
    save(OUT/'summary.json',dict(baseline='53252df',frames=len(rows),changed=changed,
        cohorts=after,previous_cohorts=before,archive_sha256=digest(archive),
        cases_sha256=digest(OUT/'cases.json'),historical_112_strict_f1=historical))
    print(after['fixed'])
if __name__=='__main__':
    main()
