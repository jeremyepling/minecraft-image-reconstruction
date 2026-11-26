"""Screen historical native exteriors with novel geometry for a sealed placement test."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile

from benchmark.relative_geometry import PRIMARY_METRIC, placement_scores
from benchmark.texture_camera_eval import DATA,ROOT,STABLE,checked,digest,frames,read,save
import numpy as np
from PIL import Image

BUILDS=Path(os.environ.get('MCIR_NATIVE_TRANSFER_FIXTURES',STABLE.parent/'workspace/builds'))
OUT=STABLE/'runs/visible-depth-rounds/transfer'
ARCHIVE=ROOT/'benchmark/reviews/texture-camera-01/scoring.zip'
# Fixed before screening: genuine vanilla 26.3, native structure-only readback,
# camera/appearance benchmarks with authored exterior full-block builds.
SUITES=('inverse-voxels-26.3/confirm','signed-relief-26.3/dev','signed-relief-26.3/test',
        'entrance-confirm-26.3/positive','entrance-confirm-26.3/controls',
        'recess-positive-control-26.3','recess-control-26.3','corner-mixed-26.3',
        'corner-views-26.3/original','corner-views-26.3/positive','corner-views-26.3/controls',
        'rim-views-26.3/original','rim-views-26.3/positive','rim-views-26.3/controls',
        'shape-benchmark-26.3','shape-benchmark-open-26.3')
METHODS=('prior','plane','art','art_filtered','family_rule','confirmed','conjunctions','invisible')
PRODUCTION={'production':'engine.atlas_plane_depth:place_ranked',
            'production_without_ray':'engine.atlas_plane_depth:place_ranked_without_ray'}

def structure(rows,ground):
    return {(r['x'],r['y'],r['z']):r['block_id'].removeprefix('minecraft:')
            for r in rows if r['block_id'].removeprefix('minecraft:')!='air' and r['y']>ground}

def shape_key(states):
    xyz=np.array(list(states),dtype=np.int32)
    variants=[]
    for swapped in [False,True]:
        p=xyz[:,[2,1,0]] if swapped else xyz
        for xsign in (-1,1):
            for zsign in (-1,1):
                points=p*[xsign,1,zsign]
                points-=points.min(0)
                variants.append(np.asarray(sorted(map(tuple,points)),dtype='<i4').tobytes())
    return hashlib.sha256(min(variants)).hexdigest()

def eligible():
    used_rgb=set()
    used_shape=set()
    for split in ('train','validation','test','regression'):
        for f in frames(DATA,split):
            used_rgb.add(f['files']['rgb.png'])
            states=read(checked(DATA/'data'/split/f['directory']/'world-readback.json',f['files']['world-readback.json']))
            used_shape.add(shape_key(structure(states,f['ground_y'])))
    with ZipFile(ARCHIVE) as z:
        for f in json.loads(z.read('native-inputs/index.json')):
            used_rgb.add(f['sha256'])
            prefix='native-inputs/'+f['fixture_directory']+'/captures/'+f['id']
            used_shape.add(shape_key(structure(json.loads(z.read(prefix+'-blocks.json')),64)))
    index=read(BUILDS/'mcir-port-validation/final-cli-26.3/atlas/placement-index.json')
    full={block for ids in index['choices']['south'].values() for block in ids}
    output=[]
    for suite in SUITES:
        folder=BUILDS/suite
        if not (folder/'layout.json').exists():continue
        layout=read(folder/'layout.json')
        ground=layout['ground_y']
        if layout['version']!='26.3' or ground!=64:continue
        pages=defaultdict(list)
        for b in layout['blocks']:
            pages[b['page']].append(b)
        authored=[structure(blocks,ground) for blocks in pages.values()]
        for f in sorted((folder/'captures').glob('*-blocks.json')):
            name=f.name.removesuffix('-blocks.json')
            rgb=f.with_name(name+'.png')
            pose=f.with_name(name+'.json')
            if not rgb.is_file() or not pose.is_file():continue
            sha=digest(rgb)
            if sha in used_rgb:continue
            meta=read(pose)
            if meta['minecraft_version']!='26.3' or meta.get('resource_packs',['vanilla'])!=['vanilla']:continue
            cells=structure(read(f),ground)
            if not cells or not any(cells==page for page in authored):continue
            names=set(cells.values())
            if not names<=full or any(any(label in b for label in ('glass','leaves','ice','honey_block','slime'))
                                      for b in names):continue
            shape=shape_key(cells)
            if shape in used_shape:continue
            used_rgb.add(sha)
            key=suite.replace('/','--')+'--'+name
            output.append(dict(id=key,source_id=name,directory=suite,shape=shape,blocks=len(cells),
                    files={'rgb.png':sha,'camera.json':digest(pose),'world-readback.json':digest(f)}))
    return output

def rgb_path(f):
    return BUILDS/f['directory']/'captures'/(f['source_id']+'.png')

def screen(f):
    from engine.camera_metric import predict
    with Image.open(checked(rgb_path(f),f['files']['rgb.png'])) as im:
        camera=predict(im.convert('RGB'),'26.3')
    return dict(f,camera=camera)

def select(destination,workers):
    destination.mkdir(exist_ok=True,parents=True)
    candidates=eligible()
    save(destination/'eligible.json',dict(methods=METHODS,suites=SUITES,frames=candidates))
    print('Eligible',len(candidates),'unique geometry',len({r['shape'] for r in candidates}),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        screened=list(pool.map(screen,candidates))
    # No placement scores or reference-pose accuracy influence this decision.
    groups=defaultdict(list)
    for f in sorted(screened,key=lambda r:r['files']['rgb.png']):
        if f['camera']['P'] is not None:groups[f['shape']].append(f)
    ordered=sorted(groups.values(),key=lambda group:group[0]['files']['rgb.png'])
    chosen=[group[i] for i in (0,1) for group in ordered if len(group)>i][:12]
    save(destination/'screen.json',dict(eligible=len(candidates),supported=sum(len(g) for g in groups.values()),
         supported_shapes=len(groups),methods=METHODS,selected=chosen,screened=screened))
    print('Selected',len(chosen),'of',sum(len(g) for g in groups.values()),'supported, groups',len(groups))

def build(job):
    f,out=job
    target=out/'upstream'/f['id']
    if not (target/'build-manifest.json').exists():
        target.mkdir(exist_ok=True,parents=True)
        with (target/'build.log').open('w',encoding='utf8') as log:
            subprocess.run(['pwsh','-NoProfile','-File',str(ROOT/'Invoke-WallRecovery.ps1'),
                            str(rgb_path(f).resolve()),'-Through','counts','-Out',str(target.resolve())],
                            check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)

def score_case(job, *, baseline_local=None):
    from benchmark import visible_depth_score as scorer
    from benchmark.camera_metrics import registration
    from benchmark.mask_blocks import _write_read,registered
    from training.visible_labels import visible_labels
    f,out,method,*paths=job
    key=f['id']
    results,baseline=paths if paths else (out/'results'/method,out/'results'/'prior')
    dest=results/key
    dest.mkdir(parents=True,exist_ok=True)
    upstream=out/'upstream'/key
    manifest=read(upstream/'build-manifest.json')
    assert manifest['image_sha256']==f['files']['rgb.png'] and manifest['through']=='counts'
    parts=read(upstream/'surfaces.json')['planes']
    counts=read(upstream/'counts.json')
    png=checked(rgb_path(f),f['files']['rgb.png'])
    with Image.open(png) as src:
        photo=Image.fromarray(np.asarray(src.convert('RGB')))
    cells,status=scorer.predict(photo,'26.3',parts,counts,f['camera'])
    pack=dest/'reconstruction.zip'
    local=_write_read(pack,cells)
    pose=read(checked(png.with_suffix('.json'),f['files']['camera.json']))
    raw=read(checked(png.with_name(f['source_id']+'-blocks.json'),f['files']['world-readback.json']))
    states={tuple(b[k] for k in 'xyz'):b for b in raw}
    visible,_,_=visible_labels(pose,states,(0,0,0))
    truth={tuple(b['xyz_world']):b['block_id'] for b in visible['blocks']}
    occupied=structure(raw,64).keys()
    assert truth.keys()<=occupied
    transform=registration(np.asarray(f['camera']['P']),pose)
    world=registered(local,transform)
    if baseline_local is None:
        before=world if baseline is None or method=='prior' else set(map(tuple,read(baseline/key/'cells.json')['world']))
    else:
        before=registered(baseline_local,transform)
    scores=placement_scores(world,before,truth,occupied)
    save(dest/'cells.json',dict(id=key,local=sorted(local),world=sorted(world)))
    return dict(id=key,camera_supported=True,shape=f['shape'],rgb_sha256=f['files']['rgb.png'],
                surface_sha256=digest(upstream/'surfaces.json'),counts_sha256=digest(upstream/'counts.json'),
                pack_sha256=digest(pack),**scores)

def score_all(out,root,workers,method):
    from benchmark.visible_depth_score import aggregate,init_candidate
    frames=read(out/'screen.json')['selected']
    for choice in METHODS if method=='all' else [method]:
        if choice=='prior':
            adapter='engine.visible_placement:place_with_camera'
        elif choice in PRODUCTION:
            adapter=PRODUCTION[choice]
        else:
            adapter='benchmark.face_reprojection_round:'+choice
        with ProcessPoolExecutor(max_workers=workers,initializer=init_candidate,initargs=(root,adapter)) as pool:
            results=list(pool.map(score_case,[(f,out,choice) for f in frames]))
        dest=out/'results'/choice
        save(dest/'cases.json',results)
        evidence=dict(primary_metric=PRIMARY_METRIC,method=choice,adapter=adapter,selection_sha256=digest(out/'screen.json'),cases_sha256=digest(dest/'cases.json'),
             candidate_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
             scorer_sha256=digest(Path(__file__)),summary=aggregate(results))
        if choice in PRODUCTION:
            module=root/'engine/atlas_plane_depth.py'
            evidence['module_sha256']=digest(module)
        elif choice!='prior':
            evidence.update(fit_sha256=digest(Path(os.environ['MCIR_OFFSET_FIT'])),
                gate_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=os.environ['MCIR_CELL_GATE_ROOT'],text=True).strip())
        save(dest/'manifest.json',evidence)
        print(choice,json.dumps(aggregate(results)),flush=True)

def camera_audit(out):
    """Compare frozen estimates against recorded visible corners after placement scoring."""
    from benchmark.camera_labels import face_labels
    from benchmark.camera_metrics import score
    results=[]
    for frame in read(out/'screen.json')['selected']:
        png=rgb_path(frame)
        pose=read(checked(png.with_suffix('.json'),frame['files']['camera.json']))
        source=png.with_name(frame['source_id']+'-blocks.json')
        blocks=read(checked(source,frame['files']['world-readback.json']))
        states={tuple(b[k] for k in 'xyz'):dict(b,block_id=b['block_id'].removeprefix('minecraft:'))
                for b in blocks}
        faces=face_labels(pose,states,dict.fromkeys(states,'structure'))
        metrics=score(dict(frame['camera'],planes=[]),pose,faces)
        fields=('vertices','within_2px','projected','mean_px','p90_px','span_mae','fov_error_deg')
        results.append(dict(id=frame['id'],**{name:metrics[name] for name in fields}))
    save(out/'camera-audit.json',results)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--stage',choices=('screen','upstream','score','camera-audit'),default='screen')
    p.add_argument('--candidate-root',type=Path)
    p.add_argument('--method',choices=(*METHODS,*PRODUCTION,'all'),default='all')
    args=p.parse_args()
    if args.stage=='screen':
        select(args.out,args.workers)
    elif args.stage=='camera-audit':
        camera_audit(args.out)
    elif args.stage=='upstream':
        selected=read(args.out/'screen.json')['selected']
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(build,[(f,args.out) for f in selected]))
    else:
        if args.candidate_root is None:
            p.error('Set the candidate root before placement scoring.')
        if args.method not in (*PRODUCTION,'prior') and not all(os.environ.get(k) for k in ('MCIR_OFFSET_FIT','MCIR_CELL_GATE_ROOT')):
            p.error('Set MCIR_OFFSET_FIT and MCIR_CELL_GATE_ROOT for the experimental adapters.')
        score_all(args.out,args.candidate_root.resolve(),args.workers,args.method)

if __name__=='__main__':
    main()
