"""Load the actual exported pack, run its function and verify the server's block readbacks."""
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from calibration.world import create_world, GROUND, PACK
from runtime.mode import require_game_runtime

ROOT=Path(__file__).resolve().parents[1]


def verify(build, jar, run, runtime_args):
    require_game_runtime('placement verification')
    placements=json.loads((build/'placements.json').read_text())
    with ZipFile(jar) as z:
        release=json.loads(z.read('version.json'))
    if placements['version']!=release['id']:
        raise ValueError('Build and verification world must target the same release')
    world=run/'fixture/saves/Palette Lab'
    if world.exists() or (run/'client').exists():
        raise ValueError('Verification requires a fresh output directory')
    blocks=placements['blocks']
    xs,ys,zs=[[b[axis] for b in blocks] for axis in ('x','y','z')]
    offset=GROUND+1-min(ys)
    if max(ys)+offset>=320:
        raise ValueError('Structure exceeds the available height above the test floor')
    create_world(world,release,{'minecraft:random_tick_speed':0},level_name='Image Reconstruction')
    dst=world/'datapacks'/PACK
    dst.parent.mkdir(exist_ok=True)
    with ZipFile(build/'reconstruction.zip') as src, ZipFile(dst,'w',ZIP_DEFLATED) as pack:
        for name in src.namelist(): pack.writestr(name,src.read(name))
        pack.writestr('data/palette_lab/function/page_0.mcfunction',
                      f'execute positioned 0 {offset} 0 run function reconstruction:build\n')
    cells=[dict(b,y=b['y']+offset,page=0) for b in blocks]
    cx=(max(xs)+min(xs)+1)/2;cy=(max(ys)+min(ys)+1)/2+offset;cz=(max(zs)+min(zs)+1)/2
    extent=max(max(vals)-min(vals)+1 for vals in (xs,ys,zs))
    views=[dict(id='reconstruction',page=0,x=cx-.45*extent,feetY=cy+.1*extent,
                z=max(zs)+2+extent*1.4,target=[cx,cy,cz])]
    (run/'views.json').write_text(json.dumps(views))
    (run/'layout.json').write_text(json.dumps(dict(version=release['id'],blocks=cells)))
    subprocess.run([sys.executable,'-m','runtime.capture','capture',*runtime_args,
                    '--run',str(run.resolve())],cwd=ROOT,check=True)
    actual=json.loads((run/'captures/reconstruction-blocks.json').read_text())
    observed={(b['x'],b['y'],b['z']):b['block_id'] for b in actual}
    if len(observed)!=len(cells):
        raise ValueError('Incomplete server readback')
    errors=[dict(expected=b,actual=observed.get((b['x'],b['y'],b['z']))) for b in cells
            if observed.get((b['x'],b['y'],b['z']))!=b['block_id']]
    result=dict(version=release['id'],expected=len(cells),matched=len(cells)-len(errors),differences=errors)
    (run/'verification.json').write_text(json.dumps(result,indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} placements differ; see {run/'verification.json'}")
    print(f"Verified {len(cells)} blocks in {world.name}: {run/'captures/reconstruction.png'}")
    return result
