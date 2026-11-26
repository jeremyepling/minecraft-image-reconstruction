"""Package the development comparisons and their per-building held-out model weights."""
import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

from benchmark.texture_camera_eval import read,save,digest

RUNS=('train0','family-geometry','family-round','family-repeat','train-final','train-iteration','train-wide')


def build(runs,out):
    out.mkdir(exist_ok=True,parents=True)
    summary={}
    with ZipFile(out/'development.zip','w',ZIP_DEFLATED) as archive:
        for dirname in RUNS:
            folder=runs/dirname
            manifest=folder/'scores.json'
            rows=read(manifest)['rows']
            archive.write(manifest,dirname+'/scores.json')
            for method in rows[0]['methods']:
                groups={}
                for r in rows:
                    pack=folder/'train'/r['id']/(method+'.zip')
                    if digest(pack)!=r['methods'][method]['sha256']:
                        raise ValueError(r['id'])
                    archive.write(pack,f"{dirname}/{method}/{r['id']}.zip")
                    family=r['id'].split('-')[1].rstrip('0123456789')
                    groups.setdefault(family,Counter()).update(r['methods'][method]['metrics'])
                totals=sum(groups.values(),Counter())
                summary[f'{dirname}/{method}']=dict(
                    cohorts={k:{s:v[s] for s in ('correct','extra_added','gained','hidden_added')} for k,v in groups.items()},
                    total={s:totals[s] for s in ('correct','extra_added','gained','hidden_added','placed','target')},
                    f1=2*totals['correct']/(totals['placed']+totals['target']))
        for folder in ('folds','geometry-folds'):
            for p in sorted((runs/folder).glob('*.json')):
                archive.write(p,folder+'/'+p.name)
        for name in ('ridge-fit.json','boost-fit.json','geometry-fit.json','ridge.json','boost.json','geometry.json'):
            archive.write(runs/name,name)
    save(out/'development.json',dict(comparisons=summary,archive_sha256=digest(out/'development.zip')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    build(args.runs,args.out)
