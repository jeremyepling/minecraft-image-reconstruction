"""Score texture-guided corner fitting using the shared frozen count protocol."""
import argparse
from pathlib import Path
from benchmark import texture_count_round as runner
from benchmark.camera_round import digest, read, save
from benchmark.portico_atlas import atlas
from benchmark.wall_join_test import read_gzip
from benchmark.normal_round import groups
from engine.surface_dimensions import dimensions
from PIL import Image


def finish(out,split):
    """Measure what the texture alone would have returned before the fixed atlas gate."""
    manifest=read(out/'manifest.json')
    rows=read(out/'cases.json')
    source,source_method=runner.FINDERS['photo']
    captures={c.floor+'/'+c.id:c for cohort in groups()[0].values() for c in cohort}
    for row in rows:
        with Image.open(captures[row['key']].path('image')) as im:
            size=im.size
        parts={part['id']:part for part in read_gzip(source/row['predictions'])[source_method]}
        preds={pred['proposal']:pred for pred in read_gzip(out/row['predictions'])}
        for surface in row['surfaces']:
            pid=surface['proposal']
            guess=preds[pid]['methods']['phase']['evidence'].get('self_repeat')
            surface['guess']['self_repeat']=(dimensions(parts[pid],guess['quad'],size,guess['counts'])['blocks']
                                              if guess else surface['guess']['current'])
        row['scores']=runner.summarize([row],['current','self_repeat','phase'])
    original=read(out/'scores.json')
    scores={k:runner.summarize([r for r in rows if k in ('all',r['cohort'],r['layout'])],['current','self_repeat','phase'])
            for k in original}
    assert all(scores[k][m]==v for k,x in original.items() for m,v in x.items())
    save(out/'cases.json',rows)
    save(out/'scores.json',scores)
    manifest.update(atlas=atlas().provenance,reproduce=f'python -m benchmark.phase_corner_round --split {split} --out runs/my-phase-corners',
                    ablation='self_repeat uses the same chosen chart and independent center strip with the final atlas gate omitted')
    manifest['artifacts'].update({key:digest(out/key) for key in ('cases.json','scores.json')})
    save(out/'manifest.json',manifest)
    print(scores['all']['self_repeat'])


def run(out,split,workers):
    runner.run(out,['phase'],split,workers,finder='photo')
    finish(out,split)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',choices=['development','previously_reviewed_regression','all'],default='all')
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    run(args.out.resolve(),args.split,args.workers)
