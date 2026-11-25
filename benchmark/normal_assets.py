"""Install the two pinned ONNX data files into this checkout for offline surface builds."""
import argparse
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from engine.atlas import digest
from engine.surface_fields import MODELS


ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'normal':('runs/moge-probe/model.onnx','https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/resolve/e50ffda41565591092adea54c6ac83d6212e1e23/model.onnx'),
    'depth':('runs/depth-probe/model_uint8.onnx','https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/34f40a4c93991327074907b51792dbc27e7d6f1c/onnx/model_uint8.onnx'),
}


def prepare(download=False):
    for kind,(env,file_name,sha) in MODELS.items():
        dest=ROOT/'.deps'/file_name
        if dest.is_file() and digest(dest)==sha:
            continue
        relative,url=SOURCES[kind]
        candidates=[ROOT/relative,ROOT.parent/'surface-round'/relative]
        if os.environ.get(env):
            candidates.insert(0,Path(os.environ[env]))
        local=next((p for p in candidates if p.is_file() and digest(p)==sha),None)
        if local is None and not download:
            raise FileNotFoundError(f'Run python -m benchmark.normal_assets --download to get {url}')
        dest.parent.mkdir(exist_ok=True)
        with NamedTemporaryFile(dir=dest.parent,delete=False) as tmp:
            name=Path(tmp.name)
        try:
            with (local.open('rb') if local else urlopen(url,timeout=180)) as src, name.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            if digest(name)!=sha:
                raise ValueError(f'Checkpoint SHA-256 mismatch for {kind}')
            name.replace(dest)
        finally:
            name.unlink(missing_ok=True)
        print(f'{kind}: verified {dest}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    prepare(parser.parse_args().download)
