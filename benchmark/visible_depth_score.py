"""Score visible-only coordinates from frozen RGB, camera, count and surface inputs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile

from benchmark.texture_camera_eval import DATA, ROOT, STABLE, SPLITS, frames, checked, digest, read, save
from benchmark.relative_geometry import PRIMARY_METRIC, placement_scores
from benchmark.mask_blocks import registered, _write_read
from benchmark.camera_metrics import registration
import numpy as np
from PIL import Image


CACHE = STABLE/'runs/visible-depth-rounds'
ARCHIVE = ROOT/'benchmark/reviews/texture-camera-01/scoring.zip'


def init_candidate(root, method):
    global predict
    predict = None
    if method:
        sys.path.insert(0,str(root))
        for name in ('engine','benchmark'):
            pkg=import_module(name)
            pkg.__path__ = [str(root/name),*[v for v in pkg.__path__ if v!=str(root/name)]]
        module, _, entry = method.partition(':')
        predict=getattr(import_module(module),entry or 'predict')


def stock(image, version, parts, counts, camera):
    from engine import visible_placement
    return visible_placement.place_with_camera(image,version,parts,counts,camera)


def integrated(image, version, parts, counts, camera):
    """Replay the public placement entry point with the frozen image-derived camera."""
    from unittest.mock import patch
    from engine import visible_placement
    with patch.object(visible_placement, 'seam_camera', return_value=camera):
        return visible_placement.place(image, version, parts, counts=counts)


def strict_only(image,version,parts,counts,camera):
    """Plain control retaining cells supported by the original grounded chart logic."""
    from engine.visible_placement import _charts, _blocks, _carriers, observed_support, SAMPLE_STRIDE
    if camera['P'] is None or not parts:
        return set()
    shape=tuple((n+SAMPLE_STRIDE-1)//SAMPLE_STRIDE for n in image.size[::-1])
    masks=[observed_support(p['support'],shape,image.size) if p.get('support') else np.zeros(shape,bool) for p in parts]
    projection=np.asarray(camera['P'])
    carriers=_carriers(image,_charts(image,version,parts,counts),masks,projection)
    cells,_,_=_blocks(masks,carriers,projection,{})
    return cells


def evaluate(job):
    f,shared,base_dir,dest,data,fixtures,frozen_hash=job
    dest.mkdir(exist_ok=True,parents=True)
    inputs=read(checked(shared,frozen_hash) if frozen_hash else shared)
    folder=data/'data'/f['split']/f['directory']
    sha=f['files']
    assert inputs['id']==f['id'] and inputs['rgb_sha256']==sha['rgb.png']
    native = f['split']=='native'
    if native:
        from benchmark import visible_depth_native as extra
    rgb_path = extra.rgb_path(f,fixtures) if native else folder/'rgb.png'
    with Image.open(checked(rgb_path,sha['rgb.png'])) as source:
        image=Image.fromarray(np.asarray(source.convert('RGB')))
    camera=inputs['camera']
    start=perf_counter()
    answer=(predict or stock)(image,'26.3',inputs['parts'],inputs['counts'],camera)
    cells,status=answer if isinstance(answer,tuple) else (answer,{})
    seconds=perf_counter()-start
    assert camera['P'] is not None or not cells
    pack=dest/'reconstruction.zip'
    local=_write_read(pack,cells)
    if native:
        pose,truth,structure=extra.truth(f,fixtures)
    else:
        pose=read(checked(folder/'camera.json',sha['camera.json']))
        labels=read(checked(folder/'labels.json',sha['labels.json']))['blocks']
        truth={tuple(b['xyz_world']):b['block_id'] for b in labels}
        states=read(checked(folder/'world-readback.json',sha['world-readback.json']))
        structure={(b['x'],b['y'],b['z']) for b in states if b['y']>f['ground_y']
                   and b['block_id'].removeprefix('minecraft:')!='air'}
    world=registered(local,registration(np.asarray(camera['P']),pose)) if local else set()
    assert truth.keys()<=structure
    if base_dir:
        base=read(base_dir/f['split']/f['id']/'cells.json')
        assert base['input_sha256']==digest(shared)
        before=set(map(tuple,base['world']))
    else:
        before=world
    if f['split']!='train' and not base_dir:
        member=f"placement/floor-r3/{f['split']}/{f['id']}/registered.json"
        if native:
            member=f"geometry/frames/{f['id']}/registered.json"
        with ZipFile(extra.GEOMETRY if native else ARCHIVE) as archive:
            archived=json.loads(archive.read(member))
        assert world==set(map(tuple,archived['registered'])),f['id']
    scores=placement_scores(world,before,truth,structure)
    input_hash=digest(shared)
    save(dest/'cells.json',dict(id=f['id'],input_sha256=input_hash,local=sorted(local),world=sorted(world)))
    return dict(id=f['id'],split=f['split'],layout=f['layout'],family=f.get('family',f['layout']),
                input_sha256=input_hash,rgb_sha256=sha['rgb.png'],pack_sha256=digest(pack),
                camera_supported=camera['P'] is not None,seconds=seconds,**scores)


def _aggregate_metrics(rows, field):
    counts=Counter()
    for row in rows:
        counts.update(row[field])
    total=dict(counts)
    total.update(frames=len(rows),supported=sum(r['camera_supported'] for r in rows),
                 precision=total['correct']/max(1,total['placed']),recall=total['correct']/max(1,total['target']),
                 f1=2*total['correct']/max(1,total['placed']+total['target']))
    return total


def aggregate(rows):
    summary = _aggregate_metrics(rows, 'metrics')
    scored = sum('relative_geometry' in row for row in rows)
    if scored:
        if scored != len(rows):
            raise ValueError('Rescore all frames before pooling relative geometry')
        summary['relative_geometry'] = _aggregate_metrics(rows, 'relative_geometry')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=(*SPLITS,'native','all'),required=True)
    p.add_argument('--candidate',help='MODULE[:FUNCTION], taking image,version,parts,counts,camera')
    p.add_argument('--candidate-root',type=Path,default=ROOT)
    p.add_argument('--inputs',type=Path,default=CACHE/'inputs')
    p.add_argument('--data',type=Path,default=DATA,help='Visible Builds v1 release and extracted data directory.')
    p.add_argument('--native-fixtures',type=Path,default=STABLE.parent/'workspace/builds')
    p.add_argument('--baseline',type=Path,default=CACHE/'baseline')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ids',nargs='+',help='Explicit frame IDs within the selected splits')
    p.add_argument('--families',nargs='+')
    p.add_argument('--ready',action='store_true',help='Select only currently available frozen input files')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.candidate and args.baseline.resolve()==args.out.resolve():
        p.error('Candidate runs need a private output directory')
    root=args.candidate_root.resolve()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain','--','engine','benchmark'],cwd=root,text=True).strip())
    source=root.joinpath(*(args.candidate.split(':')[0].split('.'))).with_suffix('.py') if args.candidate else ROOT/'engine/visible_placement.py'
    for split in SPLITS if args.split=='all' else [args.split]:
        jobs=[]
        frozen_manifest=args.inputs.parent/('inputs-'+split+'.json')
        frozen=read(frozen_manifest)['input_sha256'] if frozen_manifest.exists() else {}
        if split=='native':
            from benchmark.visible_depth_native import rows as native_rows
            samples=native_rows()
        else:
            samples=frames(args.data,split)
        for f in samples:
            inp=args.inputs/(f['id']+'.json')
            if args.ready and not inp.exists(): continue
            if args.ids and f['id'] not in args.ids: continue
            if args.families and f.get('family',f['layout']) not in args.families: continue
            dest=args.out/split/f['id']
            jobs.append((f,inp,args.baseline if args.candidate else None,dest,args.data,args.native_fixtures,
                         frozen[f['id']] if frozen else None))
        if not jobs: continue
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_candidate,initargs=(root,args.candidate)) as pool:
            rows=list(pool.map(evaluate,jobs))
        dest=args.out/split
        save(dest/'cases.json',rows)
        result=dict(split=split,primary_metric=PRIMARY_METRIC,candidate=args.candidate,revision=revision,source_dirty=dirty,candidate_sha256=digest(source),
                    scoring_sha256=digest(Path(__file__)),cases_sha256=digest(dest/'cases.json'),
                    summary=aggregate(rows),families={family:aggregate([r for r in rows if r['family']==family])
                                                    for family in sorted({r['family'] for r in rows})})
        save(dest/'manifest.json',result)
        print(split,json.dumps(result['summary']),flush=True)


if __name__=='__main__':
    main()
