"""Prepare portable RGB, image-only inputs and separately labeled scoring truth."""
import argparse
from pathlib import Path
from shutil import copyfile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from benchmark import camera_labels
from benchmark.blind_shape_fixture import ANGLES, PALETTES
from benchmark.texture_camera_eval import digest, read, save
from benchmark.visible_depth_transfer import shape_key, structure
from engine.camera_metric import predict
from training.visible_labels import write_labels


def prepare(root, palettes=PALETTES, angles=ANGLES, revision='434f44b', atlas_path=None):
    native = root/'native'
    camera_labels.label_capture(native)
    index = read(native/'index.json')
    expected = [f'holdout-{layout}-{palette}-{angle}'
                for layout,(palette,_) in palettes.items() for angle,_,_ in angles]
    if [f['id'] for f in index['frames']] != expected:
        raise ValueError('Capture IDs do not match the sealed fixture')
    records = root/'scoring/captures'
    records.mkdir(exist_ok=True, parents=True)
    frames = []
    for frame in index['frames']:
        key = frame['id']
        png = records/f'{key}.png'
        pose_file = records/f'{key}.json'
        state_file = records/f'{key}-blocks.json'
        copyfile(native/frame['image'],png)
        copyfile(native/frame['pose'],pose_file)
        raw = read(native/frame['states'])
        ground = index['ground_y']
        building = [b for b in raw if b['y'] > ground]
        save(state_file,building)
        occupied = structure(building,ground)
        origin = np.array(list(occupied)).min(axis=0).tolist()
        expected_hash = shape_key(occupied)
        states = {tuple(row[axis] for axis in 'xyz'):row for row in building}
        labels = root/'truth'/key
        labels.mkdir(exist_ok=True,parents=True)
        summary = write_labels(read(pose_file),states,origin,labels)
        with Image.open(png) as image:
            camera = predict(image.convert('RGB'),'26.3',atlas_path=atlas_path)
        frames.append(dict(id=key,source_id=key,directory='scoring',shape=expected_hash,
                           family=frame['layout'],angle=frame['angle'],palette=frame['palette'],
                           origin_world=origin,blocks=len(occupied),ground_y=ground,camera=camera,
                           target=summary,files={'rgb.png':digest(png),'camera.json':digest(pose_file),
                           'world-readback.json':digest(state_file),'raw-readback.json':digest(native/frame['states']),
                           'labels.json':digest(labels/'labels.json'),'labels.npz':digest(labels/'labels.npz')}))
    if len({f['shape'] for f in frames})!=2 or len({f['files']['rgb.png'] for f in frames})!=4:
        raise ValueError('Expected two distinct geometries and four distinct screenshots')
    save(root/'screen.json',dict(recipe_revision=revision,geometry_split='unseen structures, confirmation only',
                                 selected=frames))
    print([(f['id'],f['camera']['scale_status'],f['target']['visible_blocks']) for f in frames])


def package(root, path):
    frames=read(root/'screen.json')['selected']
    with ZipFile(path,'w',ZIP_DEFLATED) as archive:
        files=[root/'screen.json', *(root/'native'/f for f in ('layout.json','buildings.json','views.json','index.json'))]
        for f in frames:
            key = f['id']
            files.extend([*(root/'scoring/captures'/(key+suffix) for suffix in ('.png','.json','-blocks.json')),
                          *(root/'native/captures'/(key+suffix) for suffix in ('.png','.json','-blocks.json')),
                          root/'native/labels'/(key+'.json'),
                          *(root/'truth'/key/n for n in ('labels.json','labels.npz')),
                          *(root/'upstream'/key/n for n in ('build-manifest.json','surfaces.json','counts.json'))])
        for file in files:
            archive.write(file,file.relative_to(root).as_posix())
    print(path,digest(path),path.stat().st_size)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args()
    if args.archive:
        package(args.root,args.archive)
    else:
        prepare(args.root)
