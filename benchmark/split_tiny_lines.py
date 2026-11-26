"""Freeze the previously tested M-LSD Tiny source segments for all exterior fixtures."""
import argparse
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark.camera_round import ROOT, digest, read
from benchmark.mlsd_probe import probe
from benchmark.normal_round import groups
from runtime.mode import image_only
from engine.mlsd_tiny import MODEL


ARCHIVE = ROOT/'benchmark/data/mlsd-tiny-lines-v1.json.gz'
CHECKPOINT_SHA256 = MODEL[2]


@image_only()
def freeze(checkpoint, output):
    if digest(checkpoint)!=CHECKPOINT_SHA256:
        raise ValueError('M-LSD Tiny checkpoint differs from the original probe')
    cohorts,_,fixtures = groups()
    frames = {}
    elapsed = 0.
    with TemporaryDirectory() as work:
        for cohort in cohorts.values():
            for case in cohort:
                sha = case.frame['image_sha256']
                if sha in frames:
                    continue
                destination = Path(work)/sha
                probe(checkpoint,case.path('image'),destination)
                data = read(destination/'segments.json')
                if data['image_sha256']!=sha or data['runtime']!='ai-edge-litert 2.4.0':
                    raise ValueError('M-LSD frame or runtime differs from the frozen recipe')
                frames[sha] = dict(image_sha256=sha,image_size=data['size'],junctions=[],
                                   segments=[s['p']+s['q'] for s in data['segments']])
                elapsed += data['first_invoke_ms']/1000
    info = dict(model=data['model'],checkpoint_sha256=CHECKPOINT_SHA256,fixtures=fixtures,runtime=data['runtime'],
                preprocessing=data['preprocessing'],score_threshold=data['score_threshold'],
                min_length_output_pixels=data['min_length_output_pixels'],inference_seconds=elapsed,
                decoder_sha256=digest(ROOT/'engine/mlsd_tiny.py'),frames=frames)
    with output.open('xb') as file:
        file.write(gzip.compress(json.dumps(info,allow_nan=False,separators=(',',':')).encode(),mtime=0))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    freeze(args.checkpoint,args.out)
