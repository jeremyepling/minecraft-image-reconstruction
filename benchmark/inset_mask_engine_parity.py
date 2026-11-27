"""Check the engine's visible support against the frozen chooser and replay the RGB camera."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from time import perf_counter

import numpy as np

from benchmark.inset_mask_chooser_eval import OUT, suite, shipped
from benchmark.inset_mask_supplement import supplemental as reference
from benchmark.texture_camera_eval import read,save
from engine import camera_metric, inset_mask_chooser, inset_returns
from engine.facade_inset_phase import supplemental as texture
from engine.pillar_seams import native_mask


@lru_cache(maxsize=1)
def expected():
    return {r['id']:r for r in read(OUT/'collar.json')['frames']}


def timed(job):
    _,frame,_,_=job
    image,parts,counts,camera=suite.ordinary_inputs(*job)
    key=frame['id']
    base,events=inset_returns.refine(image,parts,camera)
    previously,phase=texture(image,base,camera,events) if events else (base,[])
    assert (len(events),len(phase))==(shipped()[key]['returns'],shipped()[key]['texture'])
    start=perf_counter()
    viable=inset_mask_chooser.possible(image.size,parts)
    viability=perf_counter()-start
    camera_time=None
    same_camera=None
    if viable:
        start=perf_counter()
        estimated=camera_metric.predict(image,'26.3',atlas_path=suite.ATLAS)
        camera_time=perf_counter()-start
        p,q=estimated.get('P'),camera.get('P')
        same_camera=(p is None and q is None) or (p is not None and q is not None and np.allclose(p,q,atol=1.e-8,rtol=1.e-8))
    else:
        estimated=camera
    atlas=None
    if 'atlas_phase' in counts:
        from engine.atlas_count import load_atlas
        atlas=load_atlas(suite.ATLAS,'26.3')
    start=perf_counter()
    after,actual=inset_mask_chooser.supplemental(image,parts,camera,previously,events,atlas=atlas)
    runtime=perf_counter()-start
    target=expected()[key].get('events',[])
    assert [(r['output'],r['host']) for r in actual]==[(r['output'],r['host']) for r in target],key
    if actual:
        ref,_=reference(image,parts,camera,previously,events,atlas=atlas)
        assert [p['id'] for p in after]==[p['id'] for p in ref]
        assert all(np.array_equal(native_mask(a,image.size),native_mask(b,image.size)) for a,b in zip(after,ref)),key
        by_id={p['id']:p for p in after}
        for event in actual:
            child=by_id[event['output']]
            assert child['parent_id']==event['host'] and child['relative_inverse_depth'] is None
            assert child['outline_evidence']['depth']=='unresolved' and 'inset_back_plane' not in child
    if same_camera is False:
        _,raw=inset_mask_chooser.supplemental(image,parts,estimated,previously,events,atlas=atlas)
        same_events=raw==actual
    else:
        same_events=True
    return dict(id=key,split=job[0],rgb_sha256=frame['files']['rgb.png'],possible=viable,viability_seconds=viability,
                camera_seconds=camera_time,camera_matches_frozen=same_camera, chooser_seconds=runtime,
                own_camera_same_events=same_events,events=actual)


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=5) as pool:
        rows=list(pool.map(timed,suite.jobs([*suite.framed.SIZES,'shelter'])))
    save(OUT/'default-parity.json',dict(views=len(rows),viable=sum(r['possible'] for r in rows),
        events=sum(len(r['events']) for r in rows),estimated_camera_changes=sum(not r['own_camera_same_events'] for r in rows),frames=rows))
