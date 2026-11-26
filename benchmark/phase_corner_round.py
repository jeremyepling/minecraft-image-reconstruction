"""Score texture-guided corner fitting using the shared frozen count protocol."""
import argparse
from pathlib import Path
from benchmark import texture_count_round as runner
from benchmark.camera_round import digest, read, save
from benchmark.portico_atlas import atlas


def run(out,split,workers):
    # Reuse the surface-aware protocol; the private runner name keeps the ordinary default fixed.
    runner.MODULES['phase']='phase_corners'
    runner.run(out,['phase'],split,workers,finder='photo')
    manifest=read(out/'manifest.json')
    manifest.update(atlas=atlas().provenance,reproduce=f'python -m benchmark.phase_corner_round --split {split} --out runs/my-phase-corners')
    save(out/'manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=['development','previously_reviewed_regression','all'],default='all')
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    run(args.out.resolve(),args.split,args.workers)
