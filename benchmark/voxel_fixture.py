"""Stage four withheld voxel volumes for evaluation from independent oblique views."""
import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark.fixtures import Building
from calibration.world import GROUND, PACK, create_world
from engine.atlas import discover


def massing(b):
    b.cuboid((0,0,-8), (13,4,1), 'base', 'andesite')
    b.cuboid((1,4,-8), (9,8,0), 'middle', 'andesite')
    b.cuboid((3,8,-7), (8,12,-2), 'upper', 'andesite')
    b.cuboid((9,4,-5), (13,6,-1), 'wing', 'tuff')
    b.probe('setback', (11.5,5.5,.5), False, 'air')
    b.probe('wing', (11.5,5.5,-2.5), True, 'surface')


def colonnade(b):
    for x in (1,7,15):
        for z in (-6,0):
            b.cuboid((x,0,z), (x+1,5,z+1), 'pillar', 'stone_bricks')
    b.cuboid((0,5,-7), (17,7,2), 'roof', 'stone_bricks')
    b.cuboid((7,0,-6), (16,2,-5), 'bench', 'tuff')
    b.probe('gap', (4.5,2.5,.5), False, 'air')
    b.probe('roof', (4.5,5.5,.5), True, 'surface')


def bays(b):
    b.cuboid((0,0,-4), (17,8,1), 'facade', 'tuff')
    for x0,x1,y0,y1,depth in ((2,6,0,5,2),(9,14,2,6,3)):
        for pos in product(range(x0,x1),range(y0,y1),range(1-depth,1)):
            b.cells.pop(pos)
        b.probe(f'indent-{x0}', (x0+.5,y0+.5,.5), False, 'air')
        b.probe(f'back-{x0}', (x0+.5,y0+.5,.5-depth), True, 'surface')
    b.cuboid((7,4,1), (9,7,2), 'cap', 'cobblestone')


def bridge(b):
    b.cuboid((0,0,-6), (5,10,1), 'west', 'cobblestone')
    b.cuboid((11,0,-4), (15,7,2), 'east', 'cobblestone')
    b.cuboid((5,5,-3), (11,7,0), 'bridge', 'andesite')
    b.cuboid((0,10,-6), (4,12,-2), 'crown', 'cobblestone')
    b.probe('under_bridge', (7.5,3.5,-1.5), False, 'air')
    b.probe('bridge_floor', (7.5,5.5,-1.5), True, 'surface')


def stage(jar, out):
    with ZipFile(jar) as archive:
        meta=json.loads(archive.read('version.json'))
    if meta['id']!='26.3':
        raise ValueError('Fixture requires Minecraft Java 26.3')
    out.mkdir(parents=True,exist_ok=False)
    world=out/'fixture/saves/Palette Lab'
    create_world(world,meta,{'minecraft:random_tick_speed':0},level_name='Voxel confirmation')
    (world/'datapacks').mkdir()
    blocks,records,views=[],[],[]
    with ZipFile(world/'datapacks'/PACK,'w',ZIP_DEFLATED) as pack:
        fmt=[meta['pack_version'][k] for k in ('data_major','data_minor')]
        pack.writestr('pack.mcmeta',json.dumps(dict(pack=dict(description='Voxel confirmation',
                                                              min_format=fmt,max_format=fmt))))
        for page,shape in enumerate((massing,colonnade,bays,bridge)):
            b=Building(page,'test',shape.__name__,{})
            shape(b)
            lo,hi=b.finish()
            records.append(b.data)
            common=dict(page=page,split='test',kind=b.data['kind'],building_id=b.data['id'])
            commands=[]
            # Read the entire bounding volume so misplaced solids and filled
            # openings are visible to the verifier as well as missing solids.
            for pos in product(*(range(a,z) for a,z in zip(lo,hi))):
                mat,role=b.cells.get(pos,('air','air_probe'))
                x,y,z=b.xyz(pos)
                if mat!='air':
                    commands.append(f'setblock {x} {y} {z} minecraft:{mat} strict')
                blocks.append(dict(common,x=x,y=y,z=z,block_id=mat,state=mat,role=role))
            pack.writestr(f'data/palette_lab/function/page_{page}.mcfunction','\n'.join(commands)+'\n')
            target=b.xyz([(a+z)/2 for a,z in zip(lo,hi)])
            distance=max(21,1.65*(hi[1]-lo[1]),1.32*(hi[0]-lo[0]))+.45*(hi[2]-lo[2])
            for side,degrees in (('left',-26),('right',31)):
                az=math.radians(degrees)
                views.append(dict(common,id=f"{b.data['id']}-oblique-{side}",angle=f'oblique-{side}',
                                  x=target[0]+distance*math.sin(az),z=target[2]+distance*math.cos(az),
                                  feetY=target[1]+distance*.15-1.62,target=target))
    layout=dict(version='26.3',ground_y=GROUND,time=6000,weather='clear',front='south',
                client_sha256=hashlib.sha256(jar.read_bytes()).hexdigest(),blocks=blocks)
    for filename,data in [('layout',layout),('views',views),('buildings',records)]:
        (out/f'{filename}.json').write_text(json.dumps(data,separators=(',',':')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--jar',type=Path)
    args=parser.parse_args()
    stage(args.jar or discover('26.3'),args.out)
