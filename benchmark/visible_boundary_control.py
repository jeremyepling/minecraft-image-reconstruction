"""Verify expanded replay's frozen fields, camera, atlas and decoded no-op build."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.visible_boundary_capture import BASE,RUN,configure,read,save
from benchmark.visible_boundary_screen import entries
from benchmark.visible_boundary_report import decoded_digest
from benchmark.expanded_baseline import replay_source_fields
from engine import surface_model


def verify(entry):
    source=Path(entry['frame'])/'rgb.png'
    baseline=BASE/entry['dataset']/entry['id']
    out=RUN/entry['dataset']/entry['id']/'control'
    with replay_source_fields(source,baseline):
        surface_model.build(source,'26.3',out,through='placement',atlas_path=fixed.suite.ATLAS)
    before=decoded_digest((baseline/'reconstruction.zip').read_bytes())
    after=decoded_digest((out/'reconstruction.zip').read_bytes())
    camera_a=read(baseline/'build-manifest.json')['placement']['P']
    camera_b=read(out/'build-manifest.json')['placement']['P']
    assert before==after and np.array_equal(camera_a,camera_b)
    save(out/'match.json',dict(decoded_sha256=after,projection_match=True))
    return entry['id']


if __name__=='__main__':
    configure()
    selected=[e for e in entries() if e['dataset']!=fixed.FIXED_DATASET and
              any(read(RUN/e['dataset']/e['id']/f) for f in ('proposal.json','parallel-proposal.json'))]
    with ProcessPoolExecutor(2,initializer=configure) as pool:
        for result in pool.map(verify,selected):
            print(result,flush=True)
