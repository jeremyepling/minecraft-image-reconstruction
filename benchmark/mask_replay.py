"""Resolve immutable model outputs and run the image-only surface comparison."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4
from zipfile import BadZipFile

from benchmark.camera_round import ROOT, read
from benchmark.normal_fields import cached_fields
from runtime.mode import image_only


def cache_for_run(digests, sources, private):
    """Reuse a complete cache in place. Never add files to a shared cache."""
    found={}
    for folder in sources:
        available=set()
        for sha in digests:
            try:
                compatible=cached_fields(sha,folder,kinds=('normal','depth')) is not None
            except (BadZipFile, EOFError, KeyError, ValueError):
                compatible=False
            if compatible:
                available.add(sha)
                found.setdefault(sha, folder/(sha+'.npz'))
        if len(available)==len(digests):
            return folder, False
    private.mkdir(parents=True,exist_ok=False)
    for path in found.values():
        shutil.copyfile(path,private/path.name)
    return private, len(found)!=len(digests)


def inference_env(private):
    """Use the suite's ONNX runtime, or install its exact release into the run's private dependency folder."""
    env=os.environ.copy()
    probe='import onnxruntime; assert onnxruntime.__version__=="1.23.2"'
    def available():
        return subprocess.run([sys.executable,'-c',probe],env=env,capture_output=True).returncode==0
    if available():
        return env
    sibling=ROOT.parent/'surface-round/runs/depth-probe/runtime'
    if sibling.is_dir():
        env['PYTHONPATH']=os.pathsep.join(filter(None,(str(sibling),env.get('PYTHONPATH'))))
        if available():
            return env
    # pip resolves public wheels for this Python and leaves other checkouts and installations untouched.
    private.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--only-binary',':all:',
                    '--target',str(private),'numpy==2.3.5','onnxruntime==1.23.2'],check=True)
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join(filter(None,(str(private),env.get('PYTHONPATH'))))
    if not available():
        raise RuntimeError('The pinned ONNX Runtime could not be loaded with this Python interpreter.')
    return env


@image_only()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('first','full'),default='full')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--fields',type=Path,help='Read-only cache of numerical model outputs')
    parser.add_argument('--lines',type=Path,default=ROOT/'benchmark/data/mask-lines-v1')
    args=parser.parse_args()
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%SZ')
    out=(args.out or ROOT/'runs'/f'mask-{timestamp}-{uuid4().hex[:8]}').resolve()
    if out.exists():
        parser.error(f'Output already exists: {out}')
    if args.phase=='full':
        for method in ('scalelsd','deeplsd'):
            if not (args.lines/method/'manifest.json').is_file():
                parser.error(f'Missing frozen line release for {method}: {args.lines}')
    cases=read(ROOT/'benchmark/reviews/normals-01/cases.json')
    digests={r['sha256'] for r in cases if args.phase=='full' or r['cohort']=='development'}
    sources=[args.fields] if args.fields is not None else [ROOT/'runs/normal-fields',ROOT.parent/'normal-round/runs/normal-fields']
    cache,missing=cache_for_run(digests,sources,out.with_name(out.name+'-fields'))
    env=os.environ.copy()
    cmd=[sys.executable,'-m','benchmark.mask_round','--phase',args.phase,'--fields',str(cache),
         '--lines',str(args.lines.resolve()),'--out',str(out)]
    if missing:
        env=inference_env(out.with_name(out.name+'-onnx'))
        subprocess.run([sys.executable,'-m','benchmark.normal_assets','--download'],env=env,check=True,cwd=ROOT)
        cmd.append('--generate-fields')
    print(f'Image-only comparison · {len(digests)} unique screenshot hashes · cache {cache}',flush=True)
    subprocess.run(cmd,env=env,check=True,cwd=ROOT)


if __name__=='__main__':
    main()
