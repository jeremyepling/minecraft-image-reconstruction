"""Test a confidence switch using nested building-family exclusion."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np

from benchmark.texture_camera_eval import read, save, digest
from engine.plane_repeat_evidence import feature_rows


def examples(folder):
    rows = []
    for path in sorted(folder.glob('*.json')):
        f = read(path)
        for r in f['rows']:
            q = r['choices']
            if not q:
                continue
            b = r['baseline']
            narrow = np.array([min(abs(o['level']-b),o['scene_delta']) <= 6 for o in q])
            levels = [o['level'] for o in q]
            rows.append(dict(name=f['family'], frame=f['id'], id=r['id'], baseline=b, n=r['n'], axis=r['axis'],
                             ordinary=narrow, levels=levels, baseline_samples=int(r['truth'].get(str(b),0)), x=feature_rows(q,b),
                             y=np.array([r['truth'].get(str(k),0) for k in levels]), raw=q))
    return rows


def rank_fit(rows, narrow):
    x,y,weight,starts=[],[],[],[]
    start=0
    for r in rows:
        mask = r['ordinary'] if narrow else np.ones(len(r['y']),bool)
        target=r['y'][mask]
        if len(target)<2 or target.sum()<max(24,.2*r['n']):
            continue
        starts.append(start)
        start+=len(target)
        x.extend(r['x'][mask,:12 if narrow else None])
        y.extend(target/target.sum())
        weight.append(min(target.sum()**.5,60.))
    x,y,starts=np.array(x),np.array(y),np.array(starts)
    weight=np.array(weight)/sum(weight)
    reps=np.diff(np.append(starts,len(y)))
    scale=np.repeat(weight,reps)
    w=np.zeros(x.shape[1])
    for _ in range(20):
        scores=x@w
        prob=np.exp(scores-np.repeat(np.maximum.reduceat(scores,starts),reps))
        prob/=np.repeat(np.add.reduceat(prob,starts),reps)
        mean=np.add.reduceat(prob[:,None]*x,starts,axis=0)
        grad=.002*w+x.T@(scale*(prob-y))
        hess=.002*np.eye(len(w))+x.T@((scale*prob)[:,None]*x)-mean.T@(weight[:,None]*mean)
        step=np.linalg.solve(hess,grad)
        w-=step/max(1.,abs(step).max()/2.)
        if abs(step).max()<1.e-5:break
    return w


def decisions(rows, old_w, new_w):
    values=[]
    for r in rows:
        x=r['x']
        scores=x@new_w
        old_scores=x[:,:12]@old_w
        i=int(np.argmax(np.where(r['ordinary'],old_scores,-np.inf)))
        j=int(np.argmax(scores))
        if r['baseline'] not in r['levels']:
            values.append(dict(frame=r['frame'],id=r['id'],old=r['baseline'],new=r['levels'][j],
                               old_samples=r['baseline_samples'],new_samples=int(r['y'][j]),x=None))
            continue
        ib=r['levels'].index(r['baseline'])
        if old_scores[i] <= old_scores[ib]:
            i=ib
        gap=float(scores[j]-scores[i])
        ob,nb=r['raw'][i],r['raw'][j]
        features=[min(gap,8.),np.clip(old_scores[i]-old_scores[j],-8,8),
                  ob['atlas']-nb['atlas'],ob['repeat_high']-nb['repeat_high'],
                  ob['edge']-nb['edge'],ob['boundary']-nb['boundary'],
                  float(r['levels'][i]==r['baseline'])-float(r['levels'][j]==r['baseline']),
                  float(r['axis']==1),np.log1p(r['n'])/8]
        values.append(dict(frame=r['frame'],id=r['id'],old=r['levels'][i],new=r['levels'][j],
                           old_samples=int(r['y'][i]),new_samples=int(r['y'][j]),x=np.asarray(features)))
    return values


def gate_fit(rows):
    chosen=[r for r in rows if r['x'] is not None and r['old_samples']!=r['new_samples']]
    x=np.array([r['x'] for r in chosen])
    y=np.array([r['old_samples']>r['new_samples'] for r in chosen],float)
    importance=np.sqrt([abs(r['old_samples']-r['new_samples']) for r in chosen])
    importance/=importance.sum()
    mean,std=x.mean(0),np.maximum(.05,x.std(0))
    x=np.column_stack((np.ones(len(x)),(x-mean)/std))
    w=np.zeros(x.shape[1])
    reg=np.array([.001]+[.10]*(len(w)-1))
    for _ in range(20):
        p=1./(1.+np.exp(-x@w))
        grad=reg*w+x.T@(importance*(p-y))
        hess=np.diag(reg)+x.T@((importance*p*(1-p))[:,None]*x)
        step=np.linalg.solve(hess,grad)
        w-=step
        if abs(step).max()<1.e-6:break
    return w,mean,std


def evaluate(rows, fit):
    w,mean,std=fit
    out=Counter()
    decisions_out=[]
    for r in rows:
        a,b=r['old_samples'],r['new_samples']
        change=r['old']!=r['new']
        out.update(n=1,old=a,new=b,oracle=max(a,b),changed=change,old_wins=a>b,new_wins=b>a)
        x=r['x']
        p=float(1./(1.+np.exp(-(w[0]+(x-mean)/std@w[1:])))) if x is not None else 0.
        switches=dict(conservative=p>=.75, majority=p>=.5,
                      narrow_margin=x is not None and x[0]<.5 and x[1]>1. and x[6]>=0.)
        for k,v in switches.items():
            out[k]+=a if v else b
            if v and change:
                out[k+'_switches']+=1
                out[k+'_added']+=max(0,a-b)
                out[k+'_lost']+=max(0,b-a)
        if change:
            decisions_out.append({k:v for k,v in r.items() if k!='x'} | dict(features=x.tolist() if x is not None else None,p=p))
    return dict(totals=dict(out), changed=decisions_out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('features','old','new','out'):
        p.add_argument('--'+arg, type=Path,required=True)
    a=p.parse_args()
    train=examples(a.features/'train')
    groups=sorted({r['name'] for r in train})
    outer_old=read(a.old)['folds'];outer_new=read(a.new)['folds']
    inner={k:[] for k in groups}
    for i,j in combinations(groups,2):
        rest=[r for r in train if r['name'] not in (i,j)]
        wo,wn=rank_fit(rest,True),rank_fit(rest,False)
        inner[i].extend(decisions([r for r in train if r['name']==j],wo,wn))
        inner[j].extend(decisions([r for r in train if r['name']==i],wo,wn))
    reports={}
    all_oof=[]
    for g in groups:
        fit=gate_fit(inner[g])
        held=decisions([r for r in train if r['name']==g],outer_old[g],outer_new[g])
        all_oof.extend(held)
        reports[g]=evaluate(held,fit)
    result=Counter()
    for g in reports.values():result.update(g['totals'])
    full_fit=gate_fit(all_oof)
    report=dict(totals=dict(result),families=reports,feature_files={f.name:digest(f) for f in (a.features/'train').glob('*.json')},
                old_model=digest(a.old),new_model=digest(a.new),weights=[s.tolist() for s in full_fit])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    save(a.out,report)
    print(dict(result))


if __name__=='__main__':
    main()
