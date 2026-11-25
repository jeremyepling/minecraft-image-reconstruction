"""Reconstruct visible Minecraft geometry or calibrate against a selected Java release."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from engine import catalog
from engine.reconstruction import reconstruct, analyze
from runtime.mode import image_only, require_game_runtime

ROOT=Path(__file__).resolve().parent


def runtime_options(args,jar):
    options=['--version',args.version,'--jar',str(jar.resolve())]
    for arg in ('minecraft_dir','manifest','java_home'):
        path=getattr(args,arg)
        if path: options.extend(['--'+arg.replace('_','-'),str(path.resolve())])
    return options


def calibrate(args):
    require_game_runtime('live calibration')
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
    p=cmds.add_parser('inspect-openings',help='Run the experimental entrance check and write its decisions and review overlay')
    p.add_argument('image',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p=cmds.add_parser('inspect-form',help='Trace the whole-frame outline, then inspect smaller line junctions')
    p.add_argument('--cube-turns',action='store_true',help='Fit projected block axes to the outline, face corners and depth edges')
    p.add_argument('image',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p=cmds.add_parser('fit-voxels',help='Fit experimental solid geometry to a single screenshot')
    p.add_argument('image',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--atlas',type=Path,help='Also export a smooth-stone structure pack for this atlas version')
    p.add_argument('--method',choices=['sheet','cone','symmetric','boxes','carving','layers','recess'],default='symmetric')
    p.add_argument('--relative',action='store_true',help='Allow an arbitrary model scale when the metric ground grid is missing')
    p=cmds.add_parser('analyze')
    p.add_argument('--geometry-method',choices=['baseline','texel','remap','cube'],default='baseline')
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
    p.add_argument('--through',choices=['camera','surfaces','counts','assembly'],default='assembly',
                   help='Fit the camera, find visible surfaces, measure block counts, or build a datapack')
    p.add_argument('--surface-outlines',choices=['mask-first','edges+shade','normal-rank','normal-regions'],
                   help='Replay an outline method from the visible-plane comparison')
    p.add_argument('--surface-order',choices=['geometry','depth-anything'],
                   help='Select the relative depth ordering method for a surface build')
    p.add_argument('--block-counter', choices=['legacy-repeat','masked-repeat','atlas-repeat'],
                   help='Select the measuring method for --through counts')
    p.add_argument('--material-match', choices=['atlas'],
                   help='Identify complete visible cells against the versioned atlas with --through counts')
    p.add_argument('--version',help='Target Java version')
    p.add_argument('--geometry-method',choices=['baseline','texel','remap','cube'],default='baseline')
    p.add_argument('image',type=Path)
    p.add_argument('--atlas',type=Path,help='Versioned atlas directory for assembly or counting with material matching')
    p.add_argument('--out',type=Path,required=True)
    shape=p.add_mutually_exclusive_group()
    shape.add_argument('--scene',type=Path)
    shape.add_argument('--geometry',type=Path,help='Use a previously measured grid for this image')
    p.add_argument('--white-balance',action='store_true',help='Estimate and correct a consistent neutral-patch color cast')
    p.add_argument('--match-appearance',choices=['local','shared','motif','adaptive'],help='Match calibrated face colors, share a palette, detect atlas motifs, or adapt to grid alignment')
    p.add_argument('--match-exposure',action='store_true',help='Allow bounded brightness differences when ranking candidate textures')
    p.add_argument('--library',type=Path,help='Calibrated face_textures/softened-library.npz')
    p.add_argument('--depth',type=int,help='Extrapolate a hollow shell behind one visible wall')
    args=parser.parse_args()
    if args.command=='calibrate':
        return calibrate(args)
    if args.command=='verify-build':
        from calibration.verify_build import verify
        require_game_runtime('placement verification')
        jar=args.jar or catalog.discover(args.version)
        return verify(args.build,jar,args.out,runtime_options(args,jar))
    run_image_command(parser,args)


@image_only()
def run_image_command(parser,args):
    if (args.command=='build' and args.through not in ('surfaces','counts')
            and (args.surface_outlines is not None or args.surface_order is not None)):
        parser.error('Choose --through surfaces or counts when selecting a surface method')
    if args.command=='build' and args.through!='counts' and (args.block_counter or args.material_match):
        parser.error('--block-counter and --material-match apply to --through counts')
    if args.command=='fit-voxels':
        from engine.voxel_fit import fit
        fit(args.image,args.method,args.out,args.atlas,args.relative)
    elif args.command=='inspect-form':
        from benchmark.line_form import inspect_file
        inspect_file(args.image,args.out,cube_turns=args.cube_turns)
    elif args.command=='inspect-openings':
        from benchmark.entrances import inspect_to_folder
        inspect_to_folder(args.image,args.out)
    elif args.command=='atlas':
        defaults=json.loads(args.registry.read_text()) if args.registry else None
        info=catalog.build(args.version,args.out,args.jar,defaults)
        print(f"{args.version}: {len(info['choices']['south'])} placeable south-face textures")
    elif args.command=='analyze':
        analyze(args.image,args.scene,args.out,args.geometry_method)
    elif args.through in ('camera', 'surfaces', 'counts'):
        if not args.version:
            parser.error('Camera and surface builds need --version')
        if (args.scene or args.geometry or args.geometry_method!='baseline' or args.depth is not None
                or args.library or (args.atlas and args.through!='counts') or args.match_exposure
                or args.white_balance or args.match_appearance):
            parser.error('Choose assembly for the supplied geometry or material placement options')
        if args.through == 'camera':
            from engine.camera_grid import build_camera
            build_camera(args.image,args.version,args.out)
        else:
            from engine.surface_model import build
            build(args.image,args.version,args.out,outlines=args.surface_outlines,order_method=args.surface_order,
                  through=args.through,counter=args.block_counter,atlas_path=args.atlas,material_match=args.material_match)
    else:
        if not args.atlas:
            parser.error('An assembly build needs --atlas')
        if args.version and args.version!=json.loads((args.atlas/'placement-index.json').read_text())['version']:
            parser.error('--version and the atlas version differ')
        reconstruct(args.image,args.atlas,args.scene,args.library,args.out,args.depth,args.geometry,
                    args.white_balance,match_exposure=args.match_exposure,geometry_method=args.geometry_method,match_appearance=args.match_appearance)


if __name__=='__main__':
    main()
