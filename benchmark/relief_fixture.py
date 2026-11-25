"""Stage signed-relief walls for two oblique Minecraft 26.3 framebuffer captures."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark.fixtures import Building, UP
from calibration.world import create_world, GROUND, PACK
from engine.atlas import discover


# Local left, sill, width, height, signed displacement towards the camera, face block.
# Vary layout, sign order, wall texture and angle independently of the detector.
LAYOUTS = {
    'dev': ('tuff', (-28, 30), [
        (2, 2, 4, 4, -2, 'tuff'), (8, 1, 3, 4, 1, 'smooth_stone'),
        (14, 2, 4, 3, 0, 'polished_deepslate'), (21, 1, 4, 4, -1, 'andesite'),
        (28, 2, 3, 3, 2, 'tuff'), (3, 8, 3, 3, 0, 'smooth_stone'),
        (10, 8, 4, 3, -1, 'polished_deepslate'), (18, 7, 4, 4, 2, 'polished_deepslate'),
        (26, 8, 4, 3, 0, 'andesite')]),
    'test': ('andesite', (-32, 26), [
        (1, 1, 4, 4, 0, 'cobbled_deepslate'), (7, 2, 4, 4, -2, 'andesite'),
        (14, 2, 3, 3, 1, 'andesite'), (20, 1, 3, 4, 0, 'smooth_stone'),
        (26, 2, 5, 3, -1, 'cobbled_deepslate'), (2, 8, 4, 3, 2, 'smooth_stone'),
        (10, 8, 3, 3, 0, 'tuff'), (17, 7, 4, 4, -1, 'smooth_stone'),
        (25, 8, 4, 3, 2, 'cobbled_deepslate')])
}


def quad(x, y, w, h, z):
    return [[x, UP+y+h, z], [x+w, UP+y+h, z], [x+w, UP+y, z], [x, UP+y, z]]


def generate(out, split, jar):
    with ZipFile(jar) as archive:
        meta = json.loads(archive.read('version.json'))
    if meta['id'] != '26.3':
        raise ValueError('Signed-relief fixture requires the 26.3 client')
    if out.exists():
        raise FileExistsError(f'Stage into a fresh directory: {out}')
    material, angles, panels = LAYOUTS[split]
    width, height, front = 33, 13, 1
    wall = Building(0, split, 'relief', {})
    wall.cuboid((0, 0, -3), (width, height, front), 'facade', material)
    planned = {}
    for i, (x, y, w, h, delta, face) in enumerate(panels):
        cap = front + delta
        kind = 'protrusion' if delta > 0 else 'recess' if delta < 0 else 'flat'
        for dx in range(x, x+w):
            for dy in range(y, y+h):
                if (dx, dy) in planned or not (0 < dx < width-1 and 0 < dy < height-1):
                    raise ValueError('Panels must be isolated inside the facade')
                planned[dx, dy] = (cap, face)
                for z in range(front, cap):
                    wall.cells[dx, dy, z] = (face, 'return')
                wall.cells[dx, dy, cap-1] = (face, 'panel_face')
                # Explicit air readbacks distinguish an inset from an overlaid patch.
                for z in range(cap, 4):
                    wall.cells[dx, dy, z] = ('air', 'air_probe')
        wall.data['regions'].append(dict(id=f'panel-{i+1}', kind=kind, signed_depth=delta,
            face_material=face, footprint=quad(x, y, w, h, front), cap=quad(x, y, w, h, cap)))
    common = dict(kind=wall.data['kind'], page=0, building_id=wall.data['id'], split=split)
    wall.data.update(front_z=front, facade_material=material,
                     facade_bounds=[wall.xyz((0, 0, -3)), wall.xyz((width, height, front))])
    world = out/'fixture/saves/Palette Lab'
    create_world(world, meta, {'minecraft:random_tick_speed': 0}, level_name='Signed relief')
    (world/'datapacks').mkdir()
    commands, blocks = [], []
    for pos, (block, role) in sorted(wall.cells.items()):
        x, y, z = wall.xyz(pos)
        if block != 'air':
            commands.append(f'setblock {x} {y} {z} minecraft:{block} strict')
        blocks.append(dict(common, x=x, y=y, z=z, block_id=block, state=block, role=role))
    with ZipFile(world/'datapacks'/PACK, 'w', ZIP_DEFLATED) as pack:
        fmt = [meta['pack_version'][k] for k in ('data_major', 'data_minor')]
        pack.writestr('pack.mcmeta', json.dumps(dict(pack=dict(description='Signed relief',
                                                               min_format=fmt, max_format=fmt))))
        pack.writestr('data/palette_lab/function/page_0.mcfunction', '\n'.join(commands)+'\n')
    views = []
    target = [width/2, UP+height/2, front-1.]
    for side, degrees in zip(('left', 'right'), angles):
        angle = math.radians(degrees)
        name = f'oblique-{side}'
        views.append(dict(common, id=f"{wall.data['id']}-{name}", angle=name, target=target,
                          x=target[0]+34*math.sin(angle), feetY=target[1]+5.5-1.62,
                          z=target[2]+34*math.cos(angle)))
    layout = dict(version='26.3', ground_y=GROUND, time=6000, weather='clear', front='south',
                  client_sha256=hashlib.sha256(jar.read_bytes()).hexdigest(), blocks=blocks)
    for name, data in [('layout', layout), ('views', views), ('buildings', [wall.data])]:
        (out/f'{name}.json').write_text(json.dumps(data, separators=(',', ':')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=LAYOUTS, required=True)
    parser.add_argument('--jar', type=Path)
    args = parser.parse_args()
    generate(args.out, args.split, args.jar or discover('26.3'))
