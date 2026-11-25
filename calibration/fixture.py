"""Generate all-block calibration pages and contiguous masonry seams from a client registry."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from .models import Models, INVISIBLE
from .world import GROUND, PACK, create_world
from runtime.mode import require_game_runtime

SPLITS = (('train',176,0,0,32), ('validation',581,7,3,34), ('test',911,-9,6,37))
MASONRY = ('stone','smooth_stone','andesite','diorite','polished_andesite','calcite','tuff',
           'cobblestone','mossy_cobblestone','stone_bricks','deepslate','cobbled_deepslate',
           'dead_fire_coral_block','dead_brain_coral_block','white_concrete','gray_concrete')
COLUMNS, ROWS = 20, 12


def panel(page, split, ids, adjacent, models, registry, facing):
    start=page*64
    spacing=1 if adjacent else 2
    bottom=66
    commands=[f'fill {start-1} 65 -1 {start+spacing*COLUMNS} {bottom+spacing*ROWS} -1 minecraft:gray_concrete']
    cells=[]
    for i,name in enumerate(ids):
        x=start+spacing*(i%COLUMNS)
        y=bottom+spacing*(i//COLUMNS)
        if not adjacent:
            commands.append(f'setblock {x} {y-1} 0 minecraft:gray_concrete strict')
            if name in {'water','lava','bubble_column'}:
                for dx,dz in ((-1,0),(1,0),(0,1)):
                    commands.append(f'setblock {x+dx} {y} {dz} minecraft:barrier strict')
            if name in {'cave_air','void_air'}:
                commands.append(f'setblock {x} {y} 0 minecraft:gray_concrete strict')
        commands.append(f'setblock {x} {y} 0 minecraft:{name} strict')
        cells.append(dict(x=x,y=y,z=0,block_id=name,state=name,page=page,split=split,
                          side_texture=models.front(name,registry[name]),invisible=name in INVISIBLE))
    tx=start+(COLUMNS-1)*spacing/2+.5
    ty=bottom+(ROWS-1)*spacing/2+.5
    offset,rise,distance=facing
    kind='adjacent' if adjacent else 'catalog'
    view=dict(id=f'{split}-{page:02d}',page=page,split=split,kind=kind,x=tx+offset,feetY=ty+rise-1.62,
              z=distance,target=[tx,ty,1])
    return commands,cells,view


def make_fixture(version, jar_path, out, registry_path):
    if registry_path is None:
        require_game_runtime('registry lookup')
    out.mkdir(parents=True,exist_ok=True)
    world=out/'fixture/saves/Palette Lab'
    if world.exists():
        raise FileExistsError(f'Fresh calibration needs a new run directory: {world}')
    if registry_path is None:
        registry_path=out/'registry.json'
        subprocess.run([sys.executable,'-m','runtime.capture','registry','--version',version,
                        '--jar',str(jar_path),'--out',str(registry_path)],check=True)
    registry=json.loads(registry_path.read_text(encoding='utf8'))
    if not registry or not all(isinstance(props,dict) for props in registry.values()):
        raise ValueError('Expected registry object mapping block IDs to default state properties')
    registry={k.removeprefix('minecraft:'):v for k,v in registry.items()}
    blocks=[]
    views=[]
    with ZipFile(jar_path) as jar:
        metadata=json.loads(jar.read('version.json'))
        if metadata['id']!=version:
            raise ValueError(f"Expected {version} jar, got {metadata['id']}")
        pack=metadata['pack_version']
        data_format=[pack['data_major'],pack['data_minor']]
        models=Models(jar)
        create_world(world,metadata,{'minecraft:random_tick_speed':0})
        packs=world/'datapacks'
        packs.mkdir(exist_ok=True)
        with ZipFile(packs/PACK,'w',ZIP_DEFLATED) as archive:
            archive.writestr('pack.mcmeta',json.dumps(dict(pack=dict(description='Vanilla block calibration',
                          min_format=data_format,max_format=data_format))))
            def add(split,ids,adjacent,camera):
                page=len(views)
                cmds,cells,view=panel(page,split,ids,adjacent,models,registry,camera)
                archive.writestr(f'data/palette_lab/function/page_{page}.mcfunction','\n'.join(cmds)+'\n')
                blocks.extend(cells)
                views.append(view)
            for split,seed,*camera in SPLITS:
                ids=sorted(registry)
                rng=random.Random(seed)
                rng.shuffle(ids)
                for offset in range(0,len(ids),COLUMNS*ROWS):
                    add(split,ids[offset:offset+COLUMNS*ROWS],False,camera)
                available=[name for name in MASONRY if name in registry]
                if len(available)<2:
                    raise ValueError('Target has too few supported masonry blocks for adjacency scenario')
                # Both homogeneous runs and unlike neighbors occur on each wall.
                ids=[]
                for y in range(ROWS):
                    for x in range(COLUMNS):
                        ids.append(ids[-1] if x and rng.random()<.26 else rng.choice(available))
                add(split,ids,True,camera)
    layout=dict(version=version,ground_y=GROUND,time=6000,weather='clear',front='south',blocks=blocks,
                registry_size=len(registry),client_sha256=hashlib.sha256(jar_path.read_bytes()).hexdigest())
    (out/'layout.json').write_text(json.dumps(layout))
    (out/'views.json').write_text(json.dumps(views,indent=1))
    print(f'{version}: {len(registry)} registry IDs, {len(blocks)} placements, {len(views)} views')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',required=True)
    p.add_argument('--jar',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--registry',type=Path,help='Previously exported default block states for the same version')
    args=p.parse_args()
    make_fixture(args.version,args.jar,args.out,args.registry)
