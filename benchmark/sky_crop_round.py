"""Score a frozen crop candidate on pixels, counts, joins and final placements."""
import argparse
from pathlib import Path
import re
import subprocess
import sys

from benchmark.expanded_baseline import ROOT, WORKTREES


def run(checkout, module, *args):
    subprocess.run([sys.executable,'-m','benchmark.'+module,*map(str,args)],cwd=checkout,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm')
    parser.add_argument('packet',type=Path)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*",args.arm):
        parser.error("Use a descriptive arm name with letters, numbers, underscores or hyphens")
    packet=args.packet.resolve(strict=True)
    output=ROOT/'runs/sky-crop-downstream'/args.arm/'surface-score'
    run(WORKTREES/'sky-crop-reference','sky_crop_reference','--scores-only','--output',output,
        '--candidates',args.arm+'='+str(packet))
    run(ROOT,'sky_crop_downstream',args.arm,'--packet',packet,'--workers',args.workers)
    run(ROOT,'sky_crop_count_checks',args.arm,'--reference',output/'report.json','--output',
        output/'counts-and-joins.json.gz')
    run(ROOT,'sky_crop_chart_choice',args.arm)
    run(ROOT,'sky_crop_chart_choice','--preserve',args.arm)


if __name__=='__main__':
    main()
