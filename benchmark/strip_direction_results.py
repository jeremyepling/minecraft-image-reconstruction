"""Collect scored export pairs for all three frozen strip-direction decisions."""
import argparse
from pathlib import Path

from benchmark.roof_span_results import collect
from benchmark.strip_direction_opportunity import MODES


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs'))
    parser.add_argument('--baseline',type=Path,default=Path('../strip-orientation-audit/runs'))
    parser.add_argument('--out',type=Path,default=Path('benchmark/reviews/strip-face-directions-01'))
    args=parser.parse_args()
    folders={name:args.baseline/name for name in ('control','transfer-control')}
    collect(args.runs,['control',*MODES],args.out,'control','edb10a3',folders=folders)
