"""Paired exact-pose diagnostics at the ranked placer's frozen source handoff."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmark.camera_metrics import truth_camera
from benchmark.expanded_baseline import STABLE, ROOT, read, digest
from benchmark.expanded_scores import evaluate_pack, load_reference
from engine.camera_atlas_floor import _atlas
from engine.export import write_pack, JAVA_26_3_FORMAT
from engine import placed_materials, visible_placement

METHODS = ('exact-floating-base','exact-floor-base','exact-floating-roof','exact-floor-roof')
SHARED = STABLE / 'runs/floating-build-grid-01'
OUTPUT = ROOT / 'runs/floating-build-grid-01-verified'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def exact_camera(entry, roof, floating):
    frame = Path(entry['frame'])
    labels, pose = read(frame/'labels.json'), read(frame/'camera.json')
    xyz = np.array([b['xyz_world'] for b in labels['blocks']], dtype=int)
    origin = np.array(labels['origin_world'], dtype=int)
    origin[1] = xyz[:,1].max()+1 if roof else xyz[:,1].min()
    p = truth_camera(pose)
    p[:,3] += p[:,:3] @ origin
    errors = []
    for f in labels['faces']:
        points = np.array(f['corners_world']) - origin
        pix = points @ p[:,:3].T + p[:,3]
        errors.extend(np.linalg.norm(pix[:,:2]/pix[:,2,None]-f['corners_px'],axis=1))
    worst = float(max(errors))
    if worst > .01:
        raise ValueError(f'native corner error {worst} px')
    width, height = pose['image_size']
    # The existing floorless fallback depends only on the lens and frame height.
    unit = p / np.linalg.norm(p[2,:3])
    distance = float(np.linalg.norm(unit[0,:3] - width/2*unit[2,:3])/(height/24))
    camera = dict(P=p.tolist(),scale_status='supported',reason='recorded capture projection',
                  camera_source='reference-exact-grid',floating_grid=floating,build_grid_distance=distance)
    return camera, dict(origin_world=origin.tolist(),corner_count=len(errors),maximum_native_error_px=worst,
                        median_native_error_px=float(np.median(errors)),source_default_distance=distance)


def replay(task):
    entry, variant, method, handoff = task
    source = Path(entry['variants'][variant]['image'])
    source_sha = entry['variants'][variant]['source_sha256']
    if digest(source) != source_sha:
        raise ValueError(source)
    camera, checks = exact_camera(entry,method.endswith('roof'),'floating' in method)
    if method == 'exact-distance-sky':
        labels = read(Path(entry['frame'])/'labels.json')
        origin = np.array(checks['origin_world'])
        local_centers = np.array([b['xyz_world'] for b in labels['blocks']]) + .5 - origin
        plane = np.asarray(camera['P'])[2]
        camera['build_grid_distance'] = float(np.median(local_centers @ plane[:3] + plane[3])/np.linalg.norm(plane[:3]))
        camera['floating_grid'] = True
        checks['reference_median_visible_block_axial_distance'] = camera['build_grid_distance']
    reference = load_reference(entry)
    if reference.hashes != entry['reference_sha256']:
        raise ValueError('Reference changed since fixture creation')
    data = read(handoff)
    folder = OUTPUT/method/variant/entry['id']
    folder.mkdir(parents=True,exist_ok=True)
    image = Image.open(source).convert('RGB')
    atlas = _atlas('26.3')
    if atlas is None:
        raise RuntimeError('Java 26.3 atlas missing')
    cells, status = visible_placement.place(image,'26.3',data['parts'],counts=data['counts'],
                                             joint_foreground=data['joint_foreground'],camera=camera)
    blocks,evidence = placed_materials.assign(image,status,cells,atlas)
    pack = folder/'reconstruction.zip'
    write_pack(pack,blocks,JAVA_26_3_FORMAT)
    result, aligned = evaluate_pack(entry,pack,status,assignment=evidence,reference=reference)
    origin = np.array(checks['origin_world'])
    truth = reference.truth
    absolute = {tuple(np.array(cell)+origin) for cell in cells}
    checks.update(exact_registered_correct=len(absolute & truth.keys()),exact_registered_extra=len(absolute-truth.keys()))
    write(folder/'prediction.json', dict(status=status, aligned=[[list(k),v] for k,v in sorted(aligned.items())],
          local_blocks=[[list(k),v] for k,v in sorted(blocks.items())]))
    result.update(method=method,variant=variant,projection_checks=checks,source_sha256=source_sha,
                  handoff_sha256=digest(handoff),placement_input=str(handoff),atlas=atlas.provenance)
    write(folder/'score.json', result)
    return entry['id'],variant,method,len(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    cases=read(SHARED/'fixtures.json')['cases']
    jobs=[(e,v,m,SHARED/'baseline'/v/e['dataset']/e['id']/'handoffs/placement-input.json.gz')
          for m in METHODS for v in ('pavement','plain','sky') for e in cases]
    jobs.extend((e,"sky","exact-distance-sky",SHARED/"baseline/sky"/e["dataset"]/e["id"]/"handoffs/placement-input.json.gz") for e in cases)
    pending=[]
    for e,v,m,path in jobs:
        if not (OUTPUT/m/v/e['id']/'score.json').exists():
            pending.append((e,v,m,path))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        running={}
        while pending or running:
            for task in pending[:]:
                if len(running)>=args.workers: break
                e,variant,method,_ = task
                if (OUTPUT/method/variant/e['id']/'score.json').exists():
                    pending.remove(task)
                    continue
                if task[-1].exists() and time.time()-task[-1].stat().st_mtime>2:
                    running[pool.submit(replay,task)]=task
                    pending.remove(task)
            if running:
                done,_=wait(running,timeout=3,return_when=FIRST_COMPLETED)
                for f in done:
                    print(f.result(),flush=True)
                    del running[f]
            elif args.wait and pending:
                time.sleep(5)
            else:
                break

if __name__=='__main__':
    main()
