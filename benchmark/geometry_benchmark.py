"""Score image-only geometry and outline-assisted controls, with building-grouped splits."""
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from PIL import Image

from benchmark.geometry_metrics import score,outline_prediction
from engine.propose import estimate_grid
from engine.plane_candidates import repeat_grid
from engine.texel_grid import texel_grid

METHODS=('baseline','partition','texel','common','remap','ground','outlined-baseline','outlined-partition','outlined-texel')


def load(path):
    return json.loads(path.read_text())


def prediction_directories(roots, dirs):
    if dirs is None:
        return [root/'geometry-predictions' for root in roots]
    if len(dirs)!=len(roots) or len({d.resolve() for d in dirs})!=len(roots):
        raise ValueError('Supply one distinct prediction directory per fixture root, in the same order')
    return dirs


def load_prediction(directory, method, view_id, image_hash):
    pred=load(directory/method/(view_id+'.json'))
    if pred['image_hash']!=image_hash:
        raise ValueError(f"Geometry predictions refer to a different image: {view_id}")
    return pred


@lru_cache(maxsize=4)
def fixture(root):
    buildings={b['id']:b for b in load(root/'buildings.json')}
    cells=defaultdict(set)
    for c in load(root/'layout.json')['blocks']: cells[c['page']].add(tuple(c[k] for k in ('x','y','z')))
    return buildings,cells


def score_view(task):
    root,v,methods,prediction_dir=task
    building,occupied=fixture(root); b=building[v['building_id']]; voxels=occupied[b['page']]
    path=root/'captures'/v['id']
    digest=hashlib.sha256(path.with_suffix('.png').read_bytes()).hexdigest()
    pose=load(path.with_suffix('.json'))
    raw=np.asarray(Image.open(path.with_suffix('.png')).convert('RGB'))
    entries=[]
    for method in methods:
        if method.startswith('outlined'):
            estimator={'outlined-baseline':estimate_grid,'outlined-partition':repeat_grid,'outlined-texel':texel_grid}[method]
            planes,cells=outline_prediction(raw,b,pose,voxels,estimator)
        else:
            pred=load_prediction(prediction_dir,method,v['id'],digest)
            planes,cells=pred['planes'],pred['cells']
        metrics=score(b,pose,voxels,planes,cells)
        metrics.pop('air_probes', None)
        entries.append(dict(view=v['id'],fixture=root.name,building=b['id'],family=b['kind'],angle=v['angle'],
                            method=method,n_planes=len(planes),**metrics))
    return entries


def distribution(values):
    values=[float(v) for v in values if v is not None]
    if not values: return dict(n=0,mean=None)
    return dict(n=len(values),mean=round(float(np.mean(values)),4),median=round(float(np.median(values)),4),
                p90=round(float(np.percentile(values,90)),4))


def group_report(rows):
    auto=defaultdict(list); landmarks=defaultdict(list); statuses=defaultdict(Counter);corners=defaultdict(Counter)
    for r in rows:
        for key,res in r['automatic'].items(): auto[key].append(res['abs_error'])
        for m in r['landmarks']:
            name=re.sub(r'\d+','',m['id']).rstrip('_')
            landmarks[name].append(m['abs_error']);statuses[name][m['status']]+=1
        for k,c in r['corner'].items():corners[k][str(c['prediction'])]+=1
    n=sum(r['air_evaluable'] for r in rows); fp=sum(r['air_false_positive'] for r in rows)
    return dict(views=len(rows),silhouette_iou=distribution([r['silhouette_iou'] for r in rows]),
                automatic={k:dict(total=len(v),**distribution(v)) for k,v in auto.items()},
                air=dict(probes=n,incorrectly_filled=fp),corners=corners,
                assisted_lengths={k:dict(status=statuses[k],**distribution(v)) for k,v in landmarks.items()})


def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows: grouped[r['method']].append(r)
    report={}
    for method,rs in grouped.items():
        by_family={f:group_report([r for r in rs if r['family']==f]) for f in sorted({r['family'] for r in rs})}
        by_angle={a:group_report([r for r in rs if r['angle']==a]) for a in sorted({r['angle'] for r in rs})}
        # Give each family one vote. Unknown width/height receives zero credit.
        def utility(r):
            v=[r['automatic'][k]['abs_error'] for k in ('height','width')]
            return .5*r['silhouette_iou']+.25*sum(1/(1+e) if e is not None else 0 for e in v)
        selection=float(np.mean([np.mean([utility(r) for r in rs if r['family']==f]) for f in by_family]))
        report[method]=dict(utility=selection,overall=group_report(rs),by_family=by_family,by_angle=by_angle)
    return report


def run(roots,split,out,workers,methods,selection,prediction_dirs=None):
    frozen=None
    if split=='test':
        if selection is None or not selection.is_file():
            raise ValueError('Select on development buildings before testing')
        frozen=load(selection)
        if frozen['selected'] not in methods or set(methods)-set(frozen['dev']):
            raise ValueError('The holdout must evaluate the development selection and declared methods')
    dirs=prediction_directories(roots,prediction_dirs)
    tasks=[(r,v,methods,d) for r,d in zip(roots,dirs) for v in load(r/'views.json')
           if v['split']==split and v['kind']!='adjacency']
    with ProcessPoolExecutor(workers) as pool:
        rows=[x for group in pool.map(score_view,tasks,chunksize=2) for x in group]
    report=aggregate(rows)
    out.mkdir(exist_ok=True,parents=True)
    (out/f'{split}-cases.json').write_text(json.dumps(rows,separators=(',',':')))
    (out/f'{split}-geometry.json').write_text(json.dumps(report,indent=1))
    if frozen:
        (out/'tested-selection.json').write_text(json.dumps(dict(selected=frozen['selected'],
            dev_utility=frozen['dev'][frozen['selected']],test_utility=report[frozen['selected']]['utility'],
            selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest()),indent=1))
    if split=='dev':
        selected=max((m for m in methods if not m.startswith('outlined')),key=lambda m:report[m]['utility'])
        (out/'selection.json').write_text(json.dumps(dict(selected=selected,rule='equal-family mean of .5 silhouette IoU + .25/(1+width error) + .25/(1+height error); unknown size gets zero',dev={k:v['utility'] for k,v in report.items()}),indent=1))
    print(split,[(m,round(v['utility'],3),v['overall']['automatic']) for m,v in report.items()])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',nargs='+',type=Path);p.add_argument('--split',choices=('train','dev','test'),required=True)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--methods',choices=METHODS,nargs='+',default=METHODS)
    p.add_argument('--selection',type=Path)
    p.add_argument('--prediction-dirs',type=Path,nargs='+',help='One prediction directory per fixture root, in order')
    a=p.parse_args();run(a.roots,a.split,a.out,a.workers,a.methods,a.selection,a.prediction_dirs)
