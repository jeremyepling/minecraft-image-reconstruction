"""Select an entrance finder on labeled development views and evaluate a frozen policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
from PIL import Image
import numpy as np

from benchmark.entrances import POLICIES,measure
from benchmark.oblique_recess import rectify_wall
from benchmark.recess_review import known_openings
from benchmark.textures import area,intersection
from engine.geometry import project


def grade(result,truth):
    depths=[d for _,d in truth.values()]
    candidates=result['candidates']
    matches=[]
    for j,c in enumerate(candidates):
        for name,(inv,d) in truth.items():
            quad=project(inv,c['rim'])
            shared=intersection(quad,(0,0))
            overlap=shared/max(1e-9,area(quad)+1-shared)
            if overlap>=.5:
                matches.append((overlap,j,name))
    used=set();paired={}
    for overlap,j,name in sorted(matches,reverse=True):
        if j not in paired and name not in used:
            paired[j]=(name,overlap);used.add(name)
    metrics=Counter(real=sum(d>0 for d in depths),candidates=len(candidates),localized=0,exact=0,
                    wrong_depth=0,false_recess=0,other_proposals=0,unresolved=0)
    for j,c in enumerate(candidates):
        name,overlap=paired.get(j,(None,0.))
        expected=truth[name][1] if name is not None else None
        if expected and expected>0:
            metrics['localized']+=1
        else:
            metrics['other_proposals']+=1
        if c['depth']:
            if not expected:metrics['false_recess']+=1
            elif expected==c['depth']:metrics['exact']+=1
            else:metrics['wrong_depth']+=1
        c.update(target=name,truth_depth=expected,rim_iou=round(overlap,3))
    metrics['unresolved']=metrics['real']-metrics['exact']-metrics['wrong_depth']
    return metrics


def run(roots,split):
    views={};totals={m:Counter() for m in POLICIES}
    for root in roots:
        buildings={b['id']:b for b in json.loads((root/'buildings.json').read_text())}
        for view in json.loads((root/'views.json').read_text()):
            if not any(s in view['angle'] for s in ('left','right')):continue
            if view['kind']=='recess':
                if view['split']!=split:continue
            elif view['kind'] not in ('heldout-facade','dark-inlay','physical-shade','entrance-holdout'):continue
            vid=view['id'];stem=root/'captures'/vid
            with Image.open(stem.with_suffix('.png')) as im:wall=rectify_wall(im)
            truth=known_openings(buildings[view['building_id']],json.loads(stem.with_suffix('.json').read_text()))
            results={}
            for m in POLICIES:
                prediction=measure(wall,m)
                counts=grade(prediction,truth)
                totals[m].update(counts)
                results[m]=dict(counts=counts,report=prediction)
            if vid in views:raise ValueError(f'Duplicate view: {vid}')
            views[vid]=results
    if not views or not totals[POLICIES[0]]['real']:raise ValueError('No eligible positive views')
    return totals,views


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',nargs='+',type=Path)
    p.add_argument('--split',choices=('dev','test'),required=True)
    p.add_argument('--policy',type=Path,required=True,help='Written on development, read on test')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--extra-dev',nargs='+',type=Path,help='Previously reviewed test runs to include as development data')
    a=p.parse_args()
    totals,views=run(a.roots,a.split)
    if a.extra_dev:
        if a.split!='dev':p.error('--extra-dev is only for development')
        extra,reviewed=run(a.extra_dev,'test')
        if views.keys() & reviewed.keys():raise ValueError('Repeated development view')
        views.update(reviewed)
        for method in POLICIES:totals[method].update(extra[method])
    if a.split=='dev':
        def utility(m):
            c=totals[m]
            return (c['exact']-c['wrong_depth']-2*c['false_recess'],c['localized'],-c['other_proposals'])
        selected=max(POLICIES,key=utility)
        a.policy.write_text(json.dumps(dict(selected=selected,development_views=list(views),
            rule='maximize exact minus wrong depth minus twice false recesses; break ties by localized real rims, then fewer other proposals; stable method order'),indent=2)+'\n')
    else:
        selected=json.loads(a.policy.read_text())['selected']
    report=dict(selected=selected,metrics=totals,views=views)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(selected,totals)
