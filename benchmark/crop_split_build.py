"""Run the split-only candidate through the ordinary single-image Surface Finder and Block Counter."""
import argparse
from pathlib import Path
from time import perf_counter

from benchmark.camera_round import read, revision, save
from engine.surface_model import CROP_SPLIT, build
from runtime.mode import image_only


@image_only()
def run(image, out):
    out.mkdir(parents=True, exist_ok=False)
    code = revision()
    start = perf_counter()
    build(image, '26.3', out, through='counts', boundary=CROP_SPLIT)
    path = out / 'build-manifest.json'
    manifest = read(path)
    manifest['candidate_verification'] = dict(input_basename=image.name, code=code, elapsed_s=perf_counter()-start,
                                              predictor_inputs=['source RGB', 'Java 26.3', 'pinned model weights'])
    save(path, manifest)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('image', type=Path)
    cli.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    run(args.image, args.out)
