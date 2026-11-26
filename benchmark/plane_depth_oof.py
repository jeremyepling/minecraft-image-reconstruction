"""Decode placements with each evaluation building family excluded from model fitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark.atlas_plane_round import read


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--scorer-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--families-at-once',type=int,default=4)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=a.out.resolve()
    fit=read(a.fit)
    if len(fit['frames'])!=72:
        raise ValueError('The final fit must use the 72 training frames')
    def fold(family):
        env=dict(os.environ, MCIR_OFFSET_FIT=str(a.fit.resolve()),MCIR_OFFSET_EXCLUDE_FAMILY=family,
                 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        subprocess.run([sys.executable,'-m','benchmark.visible_depth_score','--candidate-root',str(root),
                        '--candidate','benchmark.plane_ranker_replay','--split','train','--out',str(out/family),
                        '--workers','1','--families',family],env=env,cwd=a.scorer_root,check=True,
                       stdout=subprocess.DEVNULL)
        return read(out/family/'train/manifest.json')['summary']
    families=sorted(fit['folds'])
    with ThreadPoolExecutor(a.families_at_once) as pool:
        result=dict(zip(families,pool.map(fold,families)))
    total={key:sum(s[key] for s in result.values()) for key in result[families[0]]
           if key not in ('precision','recall','f1')}
    total.update(precision=total['correct']/total['placed'],recall=total['correct']/total['target'],
                 f1=2*total['correct']/(total['target']+total['placed']))
    if total['frames']!=72:
        raise ValueError('Train families did not cover 72 distinct frames')
    (out/'all.json').write_text(json.dumps(dict(families=result,summary=total),indent=2)+'\n')
    print(total)


if __name__=='__main__':
    main()
