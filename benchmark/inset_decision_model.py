"""Fit a small tree on development buildings to admit additional photographed recesses."""
import argparse
from collections import defaultdict
from pathlib import Path

from benchmark.texture_camera_eval import digest, read, save


# Features are entirely image-derived. Restrict the tree to independently
# measured edges, rim coverage, and the margin between alternative depths.
FEATURES = ('weaker_return_iou','side_iou','sill_iou','rim_min','rim_mean','host_fraction',
            'outward_margin','depth_margin','back_halo_luma_step','back_side_step','back_sill_step')
SPLITS = {'development':{'train'}, 'fixed':{'validation','test','regression','native','old','fresh'},
          'alternate':{'extra'}, 'new':{'new'}, 'shelter_gate':{'shelter'}}


def flatten(rows):
    for row in rows:
        for s in row['samples']:
            yield dict(s, id=row['id'], layout=row['layout'], family=row['family'], split=row['split'])


def train(rows, depth=0):
    good = sum(v['grade']['correct'] for v in rows)
    labels = {v['layout'] for v in rows}
    node = dict(n=len(rows), good=good, layouts=len(labels), probability=(good+1)/(len(rows)+2))
    if depth == 2 or min(good,len(rows)-good)==0 or len(labels)<2:
        return node
    def cost(n, yes):
        return yes*(n-yes)/max(n,1)
    errors = cost(len(rows),good)
    best = None
    for key in FEATURES:
        values = sorted({r['features'][key] for r in rows})
        for a,b in zip(values,values[1:]):
            cut=(a+b)/2
            lo=[r for r in rows if r['features'][key]<=cut]
            hi=[r for r in rows if r['features'][key]>cut]
            if min(len(lo),len(hi))<3 or min(len({r['layout'] for r in g}) for g in (lo,hi))<2:
                continue
            score=sum(cost(len(g),sum(r['grade']['correct'] for r in g)) for g in (lo,hi))
            if score < errors-1.e-8:
                errors=score
                best=(key,cut,lo,hi)
    if best:
        key,cut,lo,hi=best
        node.update(feature=key,threshold=cut,low=train(lo,depth+1),high=train(hi,depth+1))
    return node


def probability(tree, features):
    while 'feature' in tree:
        tree=tree['low'] if features[tree['feature']]<=tree['threshold'] else tree['high']
    return tree['probability']


def select_group(samples, mode, tree=None, threshold=1):
    strict=[r for r in samples if r['strict']]
    if mode=='oracle':
        options=[r for r in samples if r['grade']['correct']]
        return max(options,key=lambda r:r['grade']['iou'],default=None)
    if strict:
        return max(strict,key=lambda r:r['features']['depth_margin'])
    if mode=='strict':
        return None
    best=max(samples,key=lambda r:(probability(tree,r['features']),r['features']['depth_margin']))
    return best if probability(tree,best['features'])>=threshold else None


def selection(rows,mode,tree=None,threshold=1):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['id'],tuple(r['group']))].append(r)
    return [chosen for group in groups.values() if (chosen:=select_group(group,mode,tree,threshold)) is not None]


def score(rows):
    return dict(accepted=len(rows),correct=sum(r['grade']['correct'] for r in rows),
                wrong=sum(not r['grade']['correct'] for r in rows),
                distinct_correct_surfaces=len({(r['id'],r['grade']['reference']) for r in rows if r['grade']['correct']}))


def cross_validate(train_rows, column, threshold):
    selected=[]
    for group in sorted({r[column] for r in train_rows}):
        tree=train([r for r in train_rows if r[column]!=group])
        selected.extend(selection([r for r in train_rows if r[column]==group],'tree',tree,threshold))
    return score(selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    all_rows=list(flatten(read(args.data)))
    rows=[r for r in all_rows if r['split']=='train']
    choices=[]
    for threshold in (.60,.75,.85,.90,.95,1.0):
        layouts=cross_validate(rows,'layout',threshold)
        families=cross_validate(rows,'family',threshold)
        utility=lambda s:s['distinct_correct_surfaces']-4*s['wrong']
        # For tied validation results choose a higher threshold. The baseline
        # strict rule is always retained when the supplemental classifier abstains.
        choices.append((min(utility(layouts),utility(families)),utility(layouts)+utility(families),
                        threshold,layouts,families))
    _,_,chosen,*_=max(choices,key=lambda v:v[:3])
    # Also replay the less conservative selection by layout alone. This
    # makes the cost of transferring a forced learned choice measurable.
    exploratory=max(choices,key=lambda v:(v[3]['distinct_correct_surfaces']-4*v[3]['wrong'],v[2]))[2]
    tree=train(rows)
    result=dict(source_sha256=digest(args.data),train_proposals=len(rows),threshold=chosen,
                layout_only_threshold=exploratory,tree=tree,
                validation=[dict(threshold=t,by_layout=l,by_family=f) for _,_,t,l,f in choices],cohorts={})
    for name,splits in SPLITS.items():
        tests=[r for r in all_rows if r['split'] in splits]
        result['cohorts'][name]=dict(proposals=len(tests),**{mode:score(selection(tests,mode,tree,chosen))
                                             for mode in ('strict','tree','oracle')})
        result['cohorts'][name]['layout_only_tree']=score(selection(tests,'tree',tree,exploratory))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    save(args.out,result)
    print(chosen,result['cohorts'])


if __name__=='__main__':
    main()
