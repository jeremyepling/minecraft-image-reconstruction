"""Inject visible source-reference surfaces at the counter/placement handoff, for diagnosis only."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import numpy as np
from PIL import ImageFilter

from benchmark import current_placement_viewer as viewer, typed_structures as typed
from benchmark.pavement_opening_eval import visible_faces
from benchmark.relative_geometry import best_translation
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.thin_face_placement_run import ref_inputs
from engine import counted_ground_corner, eave_pinholes, exterior_foreground, placed_materials
from engine import pavement_opening, smooth_atlas_outline, smooth_extent_placement, window_wall_recheck
from engine.pavement_obstructions import window_obstructions
from engine.pillar_seams import native_mask
from engine.plane_consensus import row_runs, with_mask
from engine.surface_model import canonical
from engine.visible_placement import place


ROOT = Path(__file__).resolve().parent / 'reviews/reference-surface-feedback-01'
MODES = ('baseline', 'mask', 'directions', 'chart', 'grids', 'camera', 'joint')
SUITE = viewer.suite


@lru_cache(maxsize=1)
def releases():
    checks = {r['id']:r for r in typed.read(typed.REVIEW/'cases.json')}
    expected = {r['id']:r for r in typed.read(viewer.ROOT/'facade-inset-round-01/combined.json')['rows']}
    return checks, expected


def chart(faces, normal, camera):
    axis = int(np.argmax(abs(normal)))
    xyz = np.asarray([p for f in faces for p in f['corners_world']])
    low, hi = xyz.min(0), xyz.max(0)
    axes = [i for i in range(3) if i != axis]
    corners = np.tile(low, (4,1))
    corners[1:3, axes[0]] = hi[axes[0]]
    corners[2:, axes[1]] = hi[axes[1]]
    P = np.asarray(camera['world_to_pixel_homogeneous'])
    uvw = corners @ P[:,:3].T + P[:,3]
    if min(uvw[:,2]) <= 0:
        raise ValueError('Reference metering chart crosses camera')
    xy = uvw[:,:2] / uvw[:,2,None]
    q = canonical(xy)
    order = [np.argmin(np.linalg.norm(xy-point,axis=1)) for point in q]
    dims = np.linalg.norm(np.diff(corners[order[:3]],axis=0),axis=1).round().astype(int).tolist()
    return q, dims


def oracle_parts(im, proposals, labels, ids, mode):
    """Only supplied fields change. Every proposal keeps native frontmost pixel support."""
    mapping = np.zeros(len(labels['faces'])+1,np.int32)
    groups = defaultdict(list)
    for f in labels['faces']:
        s = f['surface_id']
        groups[s].append(f)
        mapping[f['id']] = s
    surface = mapping[ids]
    overlaps = np.array([np.bincount(surface[native_mask(p,im.size)],minlength=max(groups)+1)
                         for p in proposals])
    rotation = np.asarray(labels['camera']['world_to_camera_rotation'])
    gray = np.asarray(im.convert('L').filter(ImageFilter.GaussianBlur(.9)),dtype=np.float32)[...,None]
    result, dims, donorless = [], {}, 0
    for sid, faces in groups.items():
        mask = surface==sid
        key = 'surface-'+str(sid)
        donor = None
        if overlaps.size and overlaps[:,sid].max()>0:
            # Greatest covered area is independent of the reference normal, grid and depth.
            donor = proposals[np.argmax(overlaps[:,sid])]
        else:
            donorless += 1
        normal = np.asarray(faces[0]['normal_world'])
        n = donor.get('normal_camera') if donor else None
        if mode in ('directions','chart','grids','joint'):
            n = (rotation @ normal).tolist()
        part = dict(id=key, normal_camera=n, relative_inverse_depth=donor.get('relative_inverse_depth') if donor else None)
        if n is None:
            part.update(support=dict(size=list(im.size), runs=row_runs(mask)),support_area=int(mask.sum()))
        else:
            part = with_mask(part,mask,im.size,gray,np.ones(2))
        q, counts = chart(faces,normal,labels['camera'])
        dims[key] = counts
        if mode in ('chart','grids','joint'):
            part['quad'] = q
        result.append(part)
    return result,dims,donorless


def downstream(im, parts, counts, camera):
    cells, status = place(im,'26.3',parts,counts=counts,camera=camera,atlas_path=SUITE.ATLAS)
    cells = set(cells)
    # These ordinary terminal checks use the same public inputs in all arms. The
    # replay lacks the private pre-inset/pre-exterior context used by some builds.
    if cells and camera.get('P') is not None and status['foreground_depth']['method']=='joint':
        matched = set(pavement_opening.detect(im,parts,status))
        if matched:
            regions = [p for p in parts if p['id'] in matched]
            cells -= window_obstructions(im.size,status,regions,cells)
            cells |= window_wall_recheck.recover(im,parts,status,matched,cells)
        else:
            more,_ = smooth_extent_placement.recover(im,'26.3',parts,cells,camera,SUITE.ATLAS)
            cells |= more
            add, remove,_ = counted_ground_corner.edits(im,parts,counts,status,cells)
            cells = (cells-set(remove)) | set(add)
        cells -= exterior_foreground.misplaced(im,parts,status,cells,exterior_foreground.exterior(im))
    return cells


@lru_cache(maxsize=1)
def prepared(key):
    job = all_jobs()[key]
    _, expected = releases()
    image, parts, _, camera = viewer.structure_stages(job, expected[key])
    labels, ids = visible_faces(*job[:2])
    _, _, transform, _ = ref_inputs(job)
    return image, parts, camera, labels, ids, transform


@lru_cache(maxsize=1)
def all_jobs():
    return {j[1]['id']:j for j in SUITE.jobs([*SUITE.framed.SIZES, 'shelter'])}


def evaluate(task):
    mode,job,out = task
    split, frame, _, _ = job
    key = frame['id']
    release, _ = releases()
    old = release[key]
    im, parts, camera, labels, ids, transform = prepared(key)
    given = {}
    donorless = 0
    if mode in ('camera','joint'):
        matrix = np.asarray(labels['camera']['world_to_pixel_homogeneous']) @ transform
        camera = dict(P=matrix.tolist(), scale_status='supported',reason='oracle capture pose in registered local units')
    if mode in ('baseline','camera'):
        measured = eave_pinholes.measurement_parts(smooth_atlas_outline.measurement_parts(parts))
    else:
        parts,given,donorless = oracle_parts(im,parts,labels,ids,mode)
        measured = parts
    counts = viewer.recount(im,measured,None)
    natural = 0
    for row in counts['surfaces']:
        if row['id'] in given:
            natural += row['dimensions']['blocks'] == given[row['id']]
        if mode in ('grids','joint'):
            part = next(p for p in parts if p['id']==row['id'])
            row.update(quad=part['quad'],counts=given[row['id']])
    cells = downstream(im,parts,counts,camera)
    blocks,evidence = placed_materials.assign(im,camera,cells,viewer.calibrated_atlas())
    pack,decoded = typed.typed_pack(blocks)
    truth = {tuple(b['xyz_world']):b['block_id'] for b in labels['blocks']}
    world = typed.aligned_sites(list(decoded),transform,(0,0,0))
    translation = best_translation(world,truth)[0] if world else (0,0,0)
    points = typed.aligned_sites(list(decoded),transform,translation)
    placed = dict(zip(points,decoded.values(),strict=True))
    assigned = {p for p,cell in zip(points,decoded,strict=True) if evidence[cell]!='fallback'}
    optional = horizontal_slivers(labels,ids)
    geo = typed.geometry_score(placed,truth,optional)
    scores = typed.score(placed,truth,optional,assigned)
    row = dict(id=key,split=split,source_sha256=frame['files']['rgb.png'],mode=mode,regions=len(parts),
                no_overlap_donor=donorless, natural_span_exact=natural if given else None,archive_geometry=old['geometry'],archive_scores=old['result'],
                geometry=geo,scores=scores,translation=translation,pack_sha256=sha256(pack).hexdigest(),
                points=[[list(p),v,truth.get(p),p in optional] for p,v in sorted(placed.items())])
    (out/(key+'--'+mode+'.json')).write_text(json.dumps(row,separators=(',',':')),encoding='utf-8')
    return key, mode, geo['strict']['f1']


def evaluate_view(tasks):
    return [evaluate(task) for task in tasks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--out',type=Path,default=Path('runs/reference-surface-feedback'))
    parser.add_argument('--per-split',type=int,default=1)
    parser.add_argument('--fixed',action='store_true',help='Run every view in the 104-view fixed suite')
    parser.add_argument('--first',type=int,default=0,help='Start index within the source-hash order per split')
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    args=parser.parse_args()
    pool=defaultdict(list)
    for job in all_jobs().values():
        pool[job[0]].append(job)
    jobs=[j for group in pool.values() for j in sorted(group,key=lambda j:j[1]['files']['rgb.png'])[args.first:args.first+args.per_split]]
    if args.fixed:
        jobs = [j for j in all_jobs().values() if j[0] in SUITE.framed.EVALUATION]
        assert len(jobs) == 104
    tasks = [[(mode,j,args.out) for mode in args.modes if not (args.out/(j[1]['id']+'--'+mode+'.json')).exists()] for j in jobs]
    tasks = [t for t in tasks if t]
    args.out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(args.workers) as workers:
        for row in workers.map(evaluate_view,tasks):
            print(*row,flush=True)


if __name__=='__main__':
    main()

