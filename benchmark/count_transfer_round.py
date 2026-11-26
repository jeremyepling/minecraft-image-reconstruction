"""Replay both frozen counting experiments, independent families and the visual review."""
import argparse
from pathlib import Path

from benchmark.camera_round import ROOT
from benchmark import (coplanar_count_round, count_transfer_check,
                       count_transfer_review, phase_corner_round)


def run(out,dataset,workers):
    out.mkdir(parents=True,exist_ok=False)
    phase,donor=out/'phase',out/'coplanar'
    phase_corner_round.run(phase,'all',workers)
    coplanar_count_round.run(donor,'strict','all',workers)
    count_transfer_check.run(dataset,out/'new-families',workers,['refit','phase','coplanar'])
    count_transfer_review.run(phase,donor,out/'comparison','phase','coplanar')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--dataset',type=Path,default=ROOT/'runs/visible-builds-v1')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    run(args.out.resolve(),args.dataset,args.workers)
