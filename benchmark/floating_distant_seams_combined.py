"""Test adding the frozen distant-seam decisions to the chart contraction gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from benchmark.floating_distant_seams import (OUT,REVIEW,GROUPS,locations,data,base,combined,read,save,summarize)


def run(job):
    g,c = job
    n = c['id']
    loc = OUT/g
    rel = Path('sky')/n/'charts.json'
    old = read(data(g)[2]/rel)
    new = read(locations(g)/'charts'/rel)
    decisions = {r['id']:r['accepted'] for r in read(loc/'audit'/(n+'.json'))['charts']}
    for arm in ('contraction','combined'):
        charts = []
        for a,b in zip(old['charts'],new['charts'],strict=True):
            assert a['id']==b['id']
            shrink = a['id'] in decisions and len(a['cells'])>=256 and 4*len(b['cells'])<=len(a['cells'])
            use = shrink or (arm=='combined' and decisions.get(a['id'],False))
            charts.append(b if use else a)
        dest = loc/arm
        save(dest/'charts'/rel,dict(old,charts=charts))
        with patch.object(combined,'inputs',data):
            combined.run((g,c,dest/'charts',dest/'assembly'))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('groups',nargs='*',default=GROUPS)
    args = p.parse_args()
    jobs = [(g,c) for g in args.groups for c in data(g)[3]]
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(run,jobs))
    for g in args.groups:
        roots = dict(horizontal=base(g) if g!='holdout5' else data(g)[0].parent/'horizontal-cells/assembly',
            distant=OUT/g/'assembly',contraction=OUT/g/'contraction/assembly',combined=OUT/g/'combined/assembly')
        res = summarize(data(g)[3],roots,'connected-atlas/sky')
        print(g,{k:round(100*v['geometry_strict']['f1'],2) for k,v in res['cohorts']['all'].items()},flush=True)
        save(REVIEW/(g+'-combined.json.gz'),res)
