"""Compare localized counted-corner reconciliation on the frozen 204 ordinary source images."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from io import BytesIO
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from benchmark import current_placement_viewer as viewer, smooth_joint_placement as joint
from benchmark import pavement_opening_eval as replay, smooth_local_release as release
from benchmark.edge_vote_outline_eval import recount
from benchmark.mask_blocks import _write_read, decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement_run import ref_inputs
from engine import counted_ground_corner as corner, eave_pinholes, inset_returns, jagged_texture_join, visible_placement
from engine.surface_perimeter import with_outline
from engine.visible_raster import SAMPLE_STRIDE, observed_support

suite = viewer.suite
OUT = viewer.ROOT/'counted-ground-corner-01'
PREVIOUS = viewer.ROOT/'window-wall-01'
RUN = suite.ROOT/'runs/local-counted-corner'


@lru_cache(maxsize=1)
def controls():
    _,_,payloads = viewer.read_releases([viewer.RELEASE,viewer.ROOT/'smooth-local-placement-01',
                                         viewer.ROOT/'smooth-joint-placement-01',PREVIOUS])
    release = read(PREVIOUS/'summary.json')
    rows = read(checked(PREVIOUS/'cases.json',release['cases_sha256']))
    return {r['id']:r for r in rows},payloads


def grade(job):
    split, frame, _, _ = job
    key = frame['id']
    old, payloads = controls()
    base = old[key]
    assert base['source_hash']==frame['files']['rgb.png']
    row=dict(id=key,split=split,source_hash=base['source_hash'],previous_pack_sha256=base.get('pack_sha256',base['previous_pack_sha256']),
             baseline=base['result'],result=base['result'],added=[],removed=[])
    im,parts,counts,camera=suite.ordinary_inputs(*job)
    parts,rim,phase,outlines,_=inset_returns.detect(im,parts,'26.3',camera=camera)
    if rim or phase or outlines or base.get('openings') or camera.get('P') is None:
        return row
    parts,eaves=eave_pinholes.refine(im,parts)
    parts,joins,_=jagged_texture_join.refine(im,parts,'26.3',camera=camera)
    if eaves:
        parts=eave_pinholes.measurement_parts(parts)
    if joins:
        counts=recount(im,[with_outline(p) for p in parts],viewer.calibrated_atlas() if 'atlas_phase' in counts else None)
    before=perf_counter()
    charts=visible_placement._charts(im,'26.3',parts,counts)
    shape=tuple((n + (SAMPLE_STRIDE-1)//2)//SAMPLE_STRIDE for n in im.size[::-1])
    masks=[observed_support(p['support'],shape,im.size) for p in parts]
    anchors=corner.grounded_pairs(im,charts,parts,masks,np.asarray(camera['P']))
    row['screen_seconds']=round(perf_counter()-before,5)
    if not anchors: return row
    row['corners']={parts[i]['id']:plane for i,plane in anchors.items()}
    cells=set(decoded_pack(BytesIO(payloads[key]))) if key in payloads else joint.old_pack(job)[1]
    _,status=visible_placement.place(im,'26.3',parts,counts=counts,camera=camera,atlas_path=suite.ATLAS)
    before=perf_counter()
    added,removed,anchors=corner.edits(im,parts,counts,status,cells)
    row.update(added=sorted(added),removed=sorted(removed),edit_seconds=round(perf_counter()-before,5))
    if not (added or removed): return row
    path=RUN/key/'reconstruction.zip'
    path.parent.mkdir(parents=True,exist_ok=True)
    packed=_write_read(path,(cells-removed)|added)
    truth,occupied,t,_=ref_inputs(job)
    now,prior=registered(packed,t),registered(cells,t)
    scores=placement_scores(now,prior,dict.fromkeys(truth),occupied)
    opts=horizontal_slivers(*replay.visible_faces(split,frame))
    assert replay.priority(suite.framed.spacing.shift(prior,scores['baseline_translation']),truth,opts)==row['baseline']['facade']
    rel=scores['relative_geometry']
    row.update(pack_sha256=digest(path),block_deltas={k:rel[k] for k in ('gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')},
               result=dict(strict={k:rel[k] for k in replay.FIELDS},registered={k:scores['metrics'][k] for k in replay.FIELDS},
               facade=replay.priority(suite.framed.spacing.shift(now,scores['translation']),truth,opts),translation=scores['translation']))
    return row


def main():
    jobs=list(suite.jobs([*suite.framed.SIZES,'shelter']))
    with ProcessPoolExecutor(6) as pool:
        rows=list(pool.map(grade,jobs))
    assert len(rows)==204
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'cases.json',rows)
    before,after=[release.summarize(rows,k) for k in ('baseline','result')]
    assert before==read(PREVIOUS/'summary.json')['cohorts']
    changed=[r for r in rows if r['added'] or r['removed']]
    archive=OUT/'default-exports.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for r in changed:
            key=r['id']
            z.write(RUN/key/'reconstruction.zip',key+'/reconstruction.zip')
    fixed=after['fixed']['strict']
    historical=2*fixed['correct']/(2*fixed['correct']+fixed['missed']+fixed['extras']+384)
    save(OUT/'summary.json',dict(baseline='deba62a',frames=len(rows),changed=changed,cohorts=after,
            previous_cohorts=before,archive_sha256=digest(archive),cases_sha256=digest(OUT/'cases.json'),
            historical_112_strict_f1=historical))
    print([(r['id'],r['block_deltas']) for r in changed],after['fixed'])


if __name__=='__main__':
    main()
