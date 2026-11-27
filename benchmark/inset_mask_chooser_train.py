"""Fit a depth-two tree on surface identity and overlap; geometry depth is never its label."""
from collections import defaultdict
from pathlib import Path

import benchmark
from benchmark.texture_camera_eval import ROOT, checked, digest, read, save

COORD=ROOT.parent/'facade-inset-round'
benchmark.__path__.append(str(COORD/'benchmark'))
from benchmark import inset_decision_model as model

OUT=ROOT/'benchmark/reviews/inset-mask-chooser-01'
SOURCE=COORD/'runs/chooser/rows.json'


def target(grade):
    return bool(grade['novel'] and grade['purity']>=.80 and grade['iou']>=.50
                and grade['actual_depth'] is not None and grade['actual_depth']>0)


def cv(rows,col):
    selected=[]
    for value in sorted({r[col] for r in rows}):
        train=[r for r in rows if r[col]!=value]
        held=[r for r in rows if r[col]==value]
        tree=model.train(train)
        grouped=defaultdict(list)
        for r in held:
            grouped[r['id'],tuple(r['group'])].append(r)
        for group in grouped.values():
            if any(r['strict'] for r in group):
                continue
            usable=[r for r in group if r['features']['rim_min']>=.55 and r['features']['host_fraction']>=.90
                    and model.probability(tree,r['features'])>=.90]
            if usable:
                selected.append(max(usable,key=lambda r:(model.probability(tree,r['features']),r['features']['depth_margin'])))
    return dict(accepted=len(selected),correct=sum(r['grade']['correct'] for r in selected),
                layout_ids=sorted({r['layout'] for r in selected}))


def main():
    expected=read(COORD/'benchmark/reviews/facade-inset-round-01/model.json')['source_sha256']
    rows=[r for r in model.flatten(read(checked(SOURCE,expected))) if r['split']=='train']
    for r in rows:
        r['grade']=dict(r['grade'],correct=target(r['grade']))
    tree=model.train(rows)
    result=dict(train_views=72,proposals=len(rows),positive_proposals=sum(r['grade']['correct'] for r in rows),
                contributing_layouts=sorted({r['layout'] for r in rows}), source_sha256=digest(SOURCE),
                tree=tree,probability=.90,min_rim_side=.55,min_host_share=.90,min_remaining_share=.90,
                max_other_foreground_overlap=.10,
                cross_validation={key:cv(rows,key) for key in ['layout','family']})
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'model.json',result)
    print(result)


if __name__=='__main__':
    main()
