"""Leave-one-building-family-out conditional model for visible plane depth choice."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.atlas_plane_round import OUT, read
from engine.atlas_plane_depth import choose
from engine.plane_offset_features import features, NAMES, ranked_choice


def examples(frames, cell_target=False):
    for frame in frames:
        for row in frame['rows']:
            options = row['choices']
            if len(options)<2:
                continue
            truth=row['truth_cells'] if cell_target else row['truth']
            y = np.array([truth.get(str(o['level']),0) for o in options],float)
            if y.sum() < (1.5 if cell_target else 24) or (not cell_target and y.sum()<.2*row['n']):
                continue
            # Only rays hitting labeled visible faces on this axis supervise offset.
            yield features(options,row['baseline']), y/y.sum(), min(y.sum()**.5,60.)


def fit_examples(items, lam=.002):
    items=list(items)
    if not items:
        raise ValueError('No visible labeled face samples for fitting')
    n=items[0][0].shape[1]
    w=np.zeros(n)
    total=sum(a for x,y,a in items)
    for _ in range(20):
        grad=lam*w
        hessian=lam*np.eye(n)
        for x,y,a in items:
            scores=x@w
            probs=np.exp(scores-scores.max())
            probs/=probs.sum()
            xc=x-probs@x
            weight=a/total
            grad+=weight*((probs-y)@x)
            hessian+=weight*(xc.T@(probs[:,None]*xc))
        step=np.linalg.solve(hessian,grad)
        w-=step
        if abs(step).max()<.00001:
            break
    return w


def fit(frames, lam=.002, cell_target=False):
    return fit_examples(examples(frames,cell_target),lam)


def select(row,w,margin):
    return ranked_choice(row['choices'],row['baseline'],w,margin)


def evaluate(frames,w,margins):
    totals=dict.fromkeys(['oracle','baseline','combined',*margins],0)
    for f in frames:
        for row in f['rows']:
            q=row['choices']
            if not q: continue
            truth=lambda level:row['truth'].get(str(level),0)
            totals['oracle']+=max(truth(o['level']) for o in q)
            totals['baseline']+=truth(row['baseline'])
            totals['combined']+=truth(choose(q,row['baseline'],'combined'))
            for m in margins:
                totals[m]+=truth(select(row,w,m))
    return totals


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cell-target',action='store_true')
    args=p.parse_args()
    source=OUT/('cell-labels' if args.cell_target else 'wide-features')/'train'
    frames=[read(f) for f in sorted(source.glob('*.json'))]
    families=sorted({f['family'] for f in frames})
    margins=[0., .5, 1., 1.5]
    result,folds={},{}
    for family in families:
        w=fit([f for f in frames if f['family']!=family],cell_target=args.cell_target)
        folds[family]=w.tolist()
        result[family]=evaluate([f for f in frames if f['family']==family],w,margins)
    w=fit(frames,cell_target=args.cell_target)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(frames=[f['id'] for f in frames],features=NAMES,weights=w.tolist(),
        folds=folds,oof=result),indent=2)+'\n')
    totals={k:sum(v[k] for v in result.values()) for k in result[families[0]]}
    print(len(frames),totals)
    print(result)


if __name__=='__main__':
    main()
