"""Save Pixel-wise Planarity segment IDs before the ordinary four-pixel registration."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.wall_join_test import capture_index, checked
from engine import planarity_source as plane
from runtime.mode import image_only


class RawSegments:
    def __init__(self, segmenter):
        self.segmenter = segmenter

    def compute_planar_segments(self, *args):
        labels, count = self.segmenter.compute_planar_segments(*args)
        if labels.max() >= 65536 or labels.min() < 0:
            raise ValueError('Expected uint16 plane IDs')
        self.labels = labels.astype(np.uint16)
        return labels, count


@image_only()
def run(out, assets, tokens, split):
    import torch
    cases = read(ROOT/'benchmark/reviews/sky-gap-01/cases.json')
    captures = capture_index(ROOT/'benchmark/data')
    model, delegate, load_s = plane.load_model(assets/'PixelwisePlanarity', assets/'utils3d', assets/'model.pt', 'cuda')
    segmenter = RawSegments(delegate)
    plane.SETTINGS['tokens'] = tokens
    out.mkdir(parents=True, exist_ok=False)
    times, files = {}, {}
    for row in cases:
        if split != 'all' and row['cohort'] != split:
            continue
        sha = row['sha256']
        if sha in times:
            continue
        source = checked(captures[row['key']].path('image'),sha)
        with Image.open(source) as opened:
            _, times[sha] = plane.infer_image(opened.convert('RGB'),model,segmenter,'cuda',['relaxed'])
        dest = out/(sha+'.png')
        Image.fromarray(segmenter.labels).save(dest)
        files[dest.name] = digest(dest)
        if len(times)%10==1:
            print(len(times), sha, flush=True)
    save(out/'manifest.json', dict(release=plane.MODEL_REV,checkpoint=plane.MODEL_SHA,upstream=plane.UPSTREAM,
        moge=plane.MOGE,utils3d=plane.UTILS3D,device=torch.cuda.get_device_name(0),settings=plane.SETTINGS,
        torch=torch.__version__,segmentation='relaxed',load_seconds=load_s,timing=times,artifacts=files))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--assets',type=Path,required=True)
    cli.add_argument('--tokens',type=int,default=1600)
    cli.add_argument('--split',default='all')
    args=cli.parse_args()
    run(args.out,args.assets.resolve(),args.tokens,args.split)
