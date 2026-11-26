"""Compare native vertical texture evidence against the archived production placement."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from benchmark import nearby_face_trial as fixture
from benchmark.mask_blocks import _write_read, registered
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import STABLE, checked, digest, read, save
from benchmark.visible_recovery_report import summarize
from engine.camera_atlas_floor import _atlas
from engine.vertical_atlas_faces import observations, select

CURRENT = STABLE.parent/'minecraft-image-reconstruction-worktrees/nearby-thin-geometry/runs/release'


def selected(image,parts,counts,camera,prior):
    from engine.vertical_atlas_faces import recover
    return recover(image,parts,camera,prior,_atlas('26.3',STABLE/'runs/26.3/atlas'))


def run_frame(args):
    job, out, modes, folds = args
    split,frame,source,base,_ = job
    key=frame['id']
    base=CURRENT/split
    original=next(r for r in read(base/'cases.json') if r['id']==key)
    job=(split,frame,source,base,original)
    image,parts,camera,prior=fixture.load_prediction(job)
    assert original['rgb_sha256']==frame['files']['rgb.png']
    ctx,rows=observations(image,parts,camera,prior,_atlas('26.3',STABLE/'runs/26.3/atlas'))
    policies=dict.fromkeys(modes)
    for mode in modes:
        if mode.startswith('fold'):
            from engine.vertical_atlas_faces import weights
            path = folds/(frame['family']+'.json')
            policies[mode]=(weights(path,path.stat().st_mtime_ns),int(mode[4:])/100)
        else:
            policies[mode]=mode
    predicted={k:select(ctx,rows,prior,policy) for k,policy in policies.items()}
    transform,truth,occupied=fixture.labels(job,camera['P'])
    before=registered(prior,transform)
    results={}
    for mode,cells in predicted.items():
        dest=out/mode/split/key
        dest.mkdir(parents=True,exist_ok=True)
        pack=dest/'reconstruction.zip'
        after=registered(_write_read(pack,cells),transform)
        result=placement_scores(after,before,truth,occupied)
        results[mode]=dict(**result, sha256=digest(pack))
    return dict(id=key,family=frame.get('family',frame['layout']),rgb=original['rgb_sha256'],prior=original['pack_sha256'],methods=results)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--modes',nargs='+',default=['strict','balanced','wide'])
    cli.add_argument('--workers',type=int,default=4)
    cli.add_argument('--folds',type=Path)
    cli.add_argument('--limit',type=int)
    args=cli.parse_args()
    jobs=fixture.cases('train')[:args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(run_frame,[(j,args.out,args.modes,args.folds) for j in jobs]))
    save(args.out/'scores.json',rows)
    for mode in args.modes:
        print(mode,summarize([r['methods'][mode] for r in rows]))
