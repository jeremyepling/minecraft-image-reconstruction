"""Freeze balanced pairs of previously unscored roof layouts before testing image-only models."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

from benchmark import framed_depth_eval as framed
from benchmark.texture_camera_eval import checked, digest, read, save
from training.framing import measure_framing
from training.visible_labels import visible_labels


SUITES=('shape-benchmark-26.3','shape-benchmark-open-26.3')
HEIGHTS=(5,6,7)
VIEWS=('left','right')


def known():
    shapes,images=set(),set()
    for split,f,_ in framed.cases(framed.SIZES):
        src='old' if split in ('extra','new') else split
        builds=framed.NEW_INPUTS if split=='new' else framed.transfer.BUILDS
        _,occupied,_,_=framed.references(src,f,framed.NATIVE_INPUTS.get(split),framed.TRAINING_DATA,framed.DEPTH,
                dict(native=framed.native.BUILDS,old=builds,fresh=framed.transfer.BUILDS))
        shapes.add(framed.transfer.shape_key(occupied))
        images.add(f['files']['rgb.png'])
    for f in read(framed.REVIEW/'framed-depth-transfer-01/screen.json')['selected']:
        shapes.add(f['shape'])
        images.add(f['files']['rgb.png'])
    return shapes,images


def frame(suite,name):
    key=suite+'--'+name
    folder=framed.transfer.BUILDS/suite/'captures'
    rgb=folder/(name+'.png')
    raw_path=folder/(name+'-blocks.json')
    pose_path=folder/(name+'.json')
    raw,pose=read(raw_path),read(pose_path)
    assert pose['minecraft_version']=='26.3' and pose['hud_hidden']
    occupied=framed.transfer.structure(raw,64)
    y=max(p[1] for p in occupied)
    states={tuple(b[k] for k in 'xyz'):b for b in raw}
    _,ids,_=visible_labels(pose,states,(0,0,0))
    framing=measure_framing(pose,raw,64,ids)
    return dict(id=key,source_id=name,directory=suite,shape=framed.transfer.shape_key(occupied),
                target_block_y=y-65,files={'rgb.png':digest(rgb),'camera.json':digest(pose_path),
                'world-readback.json':digest(raw_path)},framing=framing)


def select(out):
    shapes,images=known()
    groups=defaultdict(list)
    seen=set()
    for suite in SUITES:
        root=framed.transfer.BUILDS/suite/'captures'
        for source in sorted(root.glob('*-canopy-*-left-blocks.json')):
            prefix=source.name.removesuffix('-left-blocks.json')
            occupied=framed.transfer.structure(read(source),64)
            shape=framed.transfer.shape_key(occupied)
            if shape in seen or shape in shapes:
                continue
            pair=[frame(suite,prefix+'-'+view) for view in VIEWS]
            h=max(p[1] for p in occupied)-65
            assert all(f['shape']==shape and f['target_block_y']==h for f in pair)
            if h in HEIGHTS and all(f['framing']['accepted'] and f['files']['rgb.png'] not in images for f in pair):
                groups[h].append(pair)
                seen.add(shape)
    selected=[]
    for h in HEIGHTS:
        options=sorted(groups[h],key=lambda pair:pair[0]['shape'])
        # Balance open and rear-walled builds when both types contain a novel eligible shape.
        chosen=[]
        for suite in SUITES:
            matching=[pair for pair in options if pair[0]['directory']==suite]
            if matching:
                chosen.append(matching[0])
        chosen.extend(pair for pair in options if pair not in chosen and len(chosen)<2)
        selected.extend(f for pair in chosen[:2] for f in pair)
    assert {r['target_block_y'] for r in selected}==set(HEIGHTS)
    assert len({f['files']['rgb.png'] for f in selected})==len(selected)
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,dict(selection='Two shapes per height where available, balancing suite then ascending occupancy SHA; left and right complete photos. No camera-success or score filtering.',
        checked_control_images=len(images),checked_control_shapes=len(shapes),
        eligible_novel_pairs_by_height={h:len(groups[h]) for h in HEIGHTS},selected=selected))
    print([(f['source_id'],f['target_block_y']) for f in selected])


def build(f,root,out):
    key=f['id']
    path=checked(framed.transfer.rgb_path(f),f['files']['rgb.png'])
    dest=out/'upstream'/key
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'cli.log').open('w',encoding='utf8') as log:
        subprocess.run(['pwsh','-NoProfile','-File',str(root/'Invoke-WallRecovery.ps1'),str(path),'-Out',str(dest.resolve())],
                       check=True,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    manifest=read(dest/'build-manifest.json')
    assert manifest['image_sha256']==f['files']['rgb.png'] and manifest['through']=='placement'
    print('built',key,flush=True)
    return dict(id=key,files={name:digest(dest/name) for name in ('surfaces.json','counts.json','build-manifest.json','reconstruction.zip')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('select','build'))
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--previous-root',type=Path)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.stage=='select':
        select(a.selection)
    else:
        root=a.previous_root.resolve()
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        assert revision.startswith('54c9fee')
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            rows=list(pool.map(lambda f:build(f,root,a.out),read(a.selection)['selected']))
        save(a.out/'builds.json',dict(revision=revision,selection_sha256=digest(a.selection),rows=rows))
