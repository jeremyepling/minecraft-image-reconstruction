"""Intervene on visible geometry and camera independently, then run the real material exporter."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark import typed_structures as typed
from benchmark.material_surface_feedback import sites
from benchmark.pavement_opening_eval import visible_faces
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.thin_face_placement_run import ref_inputs
from engine.placed_materials import assign

OUT = Path(__file__).resolve().parent/'reviews/material-feedback-surfaces-01'


@lru_cache(maxsize=1)
def baseline():
    summary = typed.read(typed.REVIEW/'summary.json')
    return {r['id']:r for r in typed.read(typed.checked(typed.REVIEW/'cases.json',summary['cases_sha256']))}


def export_control(job):
    """Round-trip the exact visible world coordinates and IDs through the real exporter and scorer."""
    split, frame, _, _ = job
    labels, pixels = visible_faces(split, frame)
    truth = {tuple(block["xyz_world"]): block["block_id"] for block in labels["blocks"]}
    optional = horizontal_slivers(labels, pixels)
    _, _, transform, _ = ref_inputs(job)
    transform = transform.copy()
    transform[:3, 3] += baseline()[frame["id"]]["translation"]
    shift = np.zeros(3, int)
    local = typed.aligned_sites(list(truth), np.linalg.inv(transform), shift)
    ids = dict(zip(local, (v.removeprefix("minecraft:") for v in truth.values()), strict=True))
    data, decoded = typed.typed_pack(ids)
    coords = typed.aligned_sites(list(decoded), transform, shift)
    world = dict(zip(coords, decoded.values(), strict=True))
    if world != truth:
        raise ValueError(f"Exact visible block or ID lost on export: {frame['id']}")
    result = typed.score(world, truth, optional, set(world))
    for key in ("strict", "facade"):
        if result[key]["missed"] or result[key]["extras"] or result[key]["f1"] != 1:
            raise ValueError(f"Exact export failed {key} scoring: {frame['id']}")
    return dict(id=frame["id"], split=split, pack_sha256=sha256(data).hexdigest(), **result)


def case(job):
    split, frame, _, _ = job
    key = frame['id']
    source = typed.checked(typed.source_path(split,frame), frame['files']['rgb.png'])
    orig = baseline()[key]
    snapshot = sites()[key]
    assert snapshot['pack_sha256'] == orig['pack_sha256'] and orig['source_hash'] == frame['files']['rgb.png']
    pred = [tuple(cell[:3]) for cell in snapshot['rows']]
    labels, pixels = visible_faces(split,frame)
    truth = {tuple(b['xyz_world']): b['block_id'] for b in labels['blocks']}
    optional = horizontal_slivers(labels, pixels)
    _, _, transform, _ = ref_inputs(job)
    transform = transform.copy()
    transform[:3,3] += orig['translation']
    shift = np.zeros(3, int)
    inverse = np.linalg.inv(transform)
    true_cells = typed.aligned_sites(list(truth), inverse, shift)
    assert set(typed.aligned_sites(true_cells, transform, shift)) == truth.keys()
    projected = np.array(labels['camera']['world_to_pixel_homogeneous']) @ transform
    recorded = {'P':projected.tolist()}
    _,_,_,estimated = typed.SUITE.ordinary_inputs(*job)
    archive = dict(zip(typed.aligned_sites(pred,transform,shift),[r[3] for r in snapshot['rows']],strict=True))
    assert typed.geometry_score(archive,truth,optional) == orig['geometry']
    row = dict(id=key,split=split,source_hash=orig['source_hash'],geometry_hash=orig['geometry_pack_sha256'],
               released_pack_sha256=orig['pack_sha256'],default=orig['result'])
    modes = [('camera', pred, recorded), ('positions', true_cells, estimated), ('both', true_cells, recorded)]
    with Image.open(source) as im:
        for mode, cells, camera in modes:
            ids, evidence = assign(im, camera, cells, typed.viewer.calibrated_atlas())
            data, decoded = typed.typed_pack(ids)
            world = typed.aligned_sites(list(decoded),transform,shift)
            choices = dict(zip(world,decoded.values(),strict=True))
            assigned = {w for w,c in zip(world,decoded,strict=True) if evidence[c]!='fallback'}
            score = typed.score(choices,truth,optional,assigned)
            row[mode] = dict(score, pack_sha256=sha256(data).hexdigest(),
                             geometry=typed.geometry_score(choices,truth,optional))
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--export-control-only', action='store_true',
                        help='Verify exact visible XYZ and IDs through the writer and scorer on all 204 views.')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.export_control_only:
        jobs = list(typed.SUITE.jobs([*typed.SUITE.framed.SIZES, 'shelter']))
        with ProcessPoolExecutor(args.workers) as pool:
            cases = list(pool.map(export_control, jobs))
        typed.compact(args.out / 'perfect-export-control.json',
                      dict(views=len(cases), pooled=typed.pooled_scores(cases), cases=cases))
        return
    jobs=list(typed.SUITE.jobs([*typed.SUITE.framed.EVALUATION,'extra','new','shelter']))
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(case,jobs,chunksize=1))
    groups={'fixed':[r for r in rows if r['split'] in typed.SUITE.framed.EVALUATION],
            'additional':[r for r in rows if r['split'] not in typed.SUITE.framed.EVALUATION]}
    assert [len(g) for g in groups.values()] == [104,28]
    summary={k:{mode:typed.pooled_scores([r[mode] for r in g]) for mode in ('default','camera','positions','both')}
             for k,g in groups.items()}
    typed.compact(args.out/'material-oracle-cases.json',rows)
    typed.compact(args.out/'material-oracle-summary.json',summary)


if __name__=='__main__':
    main()
