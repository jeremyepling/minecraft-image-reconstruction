"""Reconstruct visible Minecraft geometry or calibrate against a selected Java release."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from engine import catalog
from engine.reconstruction import reconstruct, analyze

ROOT=Path(__file__).resolve().parent


def runtime_options(args,jar):
    options=['--version',args.version,'--jar',str(jar.resolve())]
    for arg in ('minecraft_dir','manifest','java_home'):
        path=getattr(args,arg)
        if path: options.extend(['--'+arg.replace('_','-'),str(path.resolve())])
    return options


def calibrate(args):
    from calibration.fixture import make_fixture
    from calibration.evaluate import evaluate
    run=args.out.resolve()
    run.mkdir(parents=True,exist_ok=True)
    if args.install_dir:
        from runtime.install import install
        install(args.version,args.install_dir)
        args.minecraft_dir=args.install_dir
        base=args.install_dir/'versions'/args.version
        args.manifest=base/f'{args.version}.json'
        args.jar=base/f'{args.version}.jar'
    jar=args.jar or catalog.discover(args.version)
    runtime=[sys.executable,'-m','runtime.capture']
    options=runtime_options(args,jar)
    if (run/'fixture').exists() or (run/'client').exists():
        raise ValueError('Choose a fresh calibration directory to preserve independent captures')
    registry=run/'registry.json'
    subprocess.run(runtime+['registry',*options,'--out',str(registry)],cwd=ROOT,check=True)
    catalog.build(args.version,run/'atlas',jar,json.loads(registry.read_text()))
    make_fixture(args.version,jar.resolve(),run,registry)
    subprocess.run(runtime+['capture',*options,'--run',str(run)],cwd=ROOT,check=True)
    evaluate(run,run/'atlas/atlas.json')
    from calibration.verify_build import verify
    case=ROOT/'examples/repair-wall'
    exported=run/'independent-reconstruction'
    reconstruct(case/'wall-only.png',run/'atlas',None,run/'face_textures/softened-library.npz',exported)
    checked=verify(exported,jar,run/'placement-check',options)
    predictions=json.loads((exported/'placements.json').read_text())['blocks']
    actual={(b['x'],b['y'],b['z']):b['block_id'] for b in predictions}
    truth=json.loads((case/'truth.json').read_text())
    correct=sum(actual.get((b['x'],b['y'],b['z']))==b['block_id'] for b in truth)
    (run/'independent-check.json').write_text(json.dumps(dict(reference_cells=len(truth),
        predicted_cells=len(actual),correct_id_at_coordinate=correct,placed_as_predicted=checked['matched']),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    cmds=parser.add_subparsers(dest='command',required=True)
    for command in ('atlas','calibrate'):
        p=cmds.add_parser(command)
        p.add_argument('--version',required=True)
        p.add_argument('--jar',type=Path)
        p.add_argument('--out',type=Path,required=True)
        if command=='calibrate':
            for key in ('minecraft-dir','manifest','java-home','install-dir'): p.add_argument('--'+key,type=Path)
        else:
            p.add_argument('--registry',type=Path,help='Include default-state faces from a matching runtime registry')
    p=cmds.add_parser('analyze')
    p.add_argument('--geometry-method',choices=['baseline','texel','remap'],default='baseline')
    p.add_argument('image',type=Path)
    p.add_argument('--scene',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p=cmds.add_parser('verify-build')
    p.add_argument('build',type=Path)
    p.add_argument('--version',required=True)
    p.add_argument('--jar',type=Path)
    p.add_argument('--out',type=Path,required=True)
    for key in ('minecraft-dir','manifest','java-home'): p.add_argument('--'+key,type=Path)
    p=cmds.add_parser('build')
    p.add_argument('--geometry-method',choices=['baseline','texel','remap'],default='baseline')
    p.add_argument('image',type=Path)
    for key in ('atlas','out'): p.add_argument('--'+key,type=Path,required=True)
    shape=p.add_mutually_exclusive_group()
    shape.add_argument('--scene',type=Path)
    shape.add_argument('--geometry',type=Path,help='Use a previously measured grid for this image')
    p.add_argument('--white-balance',action='store_true',help='Estimate and correct a consistent neutral-patch color cast')
    p.add_argument('--match-appearance',choices=['local','shared','motif','adaptive'],help='Match calibrated face colors, share a palette, detect atlas motifs, or adapt to grid alignment')
    p.add_argument('--match-exposure',action='store_true',help='Allow bounded brightness differences when ranking candidate textures')
    p.add_argument('--library',type=Path,help='Calibrated face_textures/softened-library.npz')
    p.add_argument('--depth',type=int,help='Extrapolate a hollow shell behind one visible wall')
    args=parser.parse_args()
    if args.command=='atlas':
        defaults=json.loads(args.registry.read_text()) if args.registry else None
        info=catalog.build(args.version,args.out,args.jar,defaults)
        print(f"{args.version}: {len(info['choices']['south'])} placeable south-face textures")
    elif args.command=='calibrate':
        calibrate(args)
    elif args.command=='analyze':
        analyze(args.image,args.scene,args.out,args.geometry_method)
    elif args.command=='verify-build':
        from calibration.verify_build import verify
        jar=args.jar or catalog.discover(args.version)
        verify(args.build,jar,args.out,runtime_options(args,jar))
    else:
        reconstruct(args.image,args.atlas,args.scene,args.library,args.out,args.depth,args.geometry,
                    args.white_balance,match_exposure=args.match_exposure,geometry_method=args.geometry_method,match_appearance=args.match_appearance)


if __name__=='__main__':
    main()
