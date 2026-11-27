"""Isolate finite pixel-area integration from native projection and extra depth/phase search."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
from collections import Counter
import argparse
import numpy as np
from PIL import Image
from benchmark import view_atlas_lattice as concept,view_atlas_lattice_run as run
from benchmark import full_block_lattice_run as ordinary,full_block_lattice_groups as groups
from benchmark.view_conditioned_atlas import footprint_uv
from benchmark.expanded_baseline import read,save


OUT=run.RUN.parent/'view-atlas-lattice-point-01'


def center(P,xy,cell,axis,*,samples=1):
    return footprint_uv(P,xy,cell,axis,samples=1)


def one(entry):
    path=OUT/entry['dataset']/entry['id']
    if (path/'matched-point.json').exists():
        return
    with patch.object(run,'RUN',OUT),patch.object(concept,'footprint_uv',center):
        if not (path/'result.json').exists():
            run.frame(entry)
        if entry['dataset']=='decorative-builds-v1':
            concept._LAST['rgb']=np.asarray(Image.open(Path(entry['frame'])/'rgb.png').convert('RGB'))
            with patch.object(groups.run,'RUN',OUT),patch.object(groups,'occupancy',concept.group):
                groups.one(entry)
    point=read(path/'result.json')
    joint=read(run.RUN/entry['dataset']/entry['id']/'result.json')
    data=dict(id=entry['id'],dataset=entry['dataset'],split=entry['split'],
        source_sha256=entry['rgb_sha256'],point=point['source'],finite=joint['source'],
        point_confirmed=point['evidence'].get('confirmed',0),finite_confirmed=joint['evidence'].get('confirmed',0),
        point_cell_moves=point['evidence'].get('cuts',{}).get('changed_owner',0),
        finite_cell_moves=joint['evidence'].get('cuts',{}).get('changed_owner',0),
        point_rescued=point['rescued'],finite_rescued=joint['rescued'])
    if entry['dataset']=='decorative-builds-v1':
        data['point_groups']=read(path/'grouping.json')['grouped']
        data['finite_groups']=read(run.RUN/entry['dataset']/entry['id']/'grouping.json')['grouped']
    save(path/'matched-point.json',data)
    print(entry['id'],flush=True)


def main():
    with ProcessPoolExecutor(2) as p:
        for _ in p.map(one,run.all_frames()):pass
    rows=[read(p) for p in OUT.glob('*/*/matched-point.json')]
    summary={}
    for ds in (ordinary.FIXED,*ordinary.COHORTS):
        group=[r for r in rows if r['dataset']==ds]
        item={'tested':len(group)}
        for name in ('point','finite'):
            res={k:ordinary.typed.pooled_scores([r[name][k] for r in group]) for k in ('geometry','scores','registered','registered_scores')}
            res.update(anchors=sum(r[name+'_confirmed'] for r in group),cell_moves=sum(r[name+'_cell_moves'] for r in group),
                       rescued=sum(r[name+'_rescued'] for r in group))
            if ds=='decorative-builds-v1':
                for split in ('all','test'):
                    counts=Counter()
                    for r in group:
                        if split=='all' or r['split']==split:counts.update(r[name+'_groups'])
                    res['grouping_'+split]=dict(counts)
            item[name]=res
        summary[ds]=item
    save(run.REVIEW/'upstream-point-control.json',summary)
    save(run.REVIEW/'upstream-point-cases.json.gz',rows)


if __name__=='__main__':
    main()
