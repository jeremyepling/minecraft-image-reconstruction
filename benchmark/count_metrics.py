"""Evaluation only: true plane dimensions, count errors and block-ID agreement."""
from collections import Counter, defaultdict

import numpy as np

from benchmark.surface_metrics import _components
from engine.geometry import UNIT, project
from engine.surface_model import canonical


def reference_planes(ref, faces):
    """Recover the exact labeled cells of the same connected planes as the frozen ID map."""
    planes = defaultdict(dict)
    for f in faces:
        normal = tuple(f['normal'])
        axis = int(np.argmax(np.abs(normal)))
        level = f['world_corners'][0][axis]
        cell = tuple(x for k,x in enumerate(f['xyz']) if k!=axis)
        planes[(normal,level)][cell] = f
    result = {}
    for (normal,level),cells in sorted(planes.items()):
        for group in _components(cells):
            pid = len(result)+1
            patch = ref['patches'][pid]
            if tuple(patch['normal'])!=normal or level!=patch['offset'] or len(group)!=patch['faces']:
                raise ValueError('Reference patch changed')
            coords = np.asarray(sorted(group))
            lo,hi = coords.min(0),coords.max(0)+1
            chart = ref['charts'][pid]
            # Place the world directions in the same screen-right / screen-down convention as the finder.
            q = canonical(project(chart,lo+UNIT*(hi-lo)))
            corners = project(np.linalg.inv(chart),q)
            first_axis = int(np.argmax(abs(corners[1]-corners[0])))
            size = (hi-lo)[[first_axis,1-first_axis]].tolist()
            result[pid] = dict(size=size,faces={c:cells[c] for c in group})
    return result


def chart_spans(chart,quad):
    if quad is None:
        return None
    uv = project(np.linalg.inv(chart),quad)
    lengths = np.linalg.norm(np.roll(uv,-1,axis=0)-uv,axis=1)
    return (.5*(lengths[:2]+lengths[2:])).tolist()


def count_totals(rows, method, diagnostic=False):
    total=Counter(eligible=0,matched=0,metered=0,returned=0,correct=0,incorrect=0,
                  unmatched_returned=0,across_right=0,down_right=0,across_returned=0,down_returned=0,
                  chart_both_within_half=0,chart_both_returned=0)
    errors=[]
    for row in rows:
        records=row['known_rectangles'] if diagnostic else row['surfaces']
        total['eligible']+=len(records) if diagnostic else row['eligible']
        for rec in records:
            pred=rec['measures'][method]
            guess=pred['counts'] if diagnostic else pred['dimensions']['blocks']
            both=all(n is not None for n in guess)
            truth=rec.get('expected')
            if truth is None:
                total['unmatched_returned']+=int(both)
                continue
            total['matched']+=1
            total['metered']+=int(rec['quad'] is not None)
            ok=[]
            for i,axis in enumerate(('across','down')):
                err=abs(guess[i]-truth[i]) if guess[i] is not None else None
                close=err is not None and err <= (.5000001 if diagnostic else 0.)
                total[axis+'_returned']+=int(err is not None)
                total[axis+'_right']+=int(close)
                ok.append(close)
            if both:
                total['returned']+=1
                total['correct']+=int(all(ok))
                total['incorrect']+=int(not all(ok))
                errors.extend(abs(n-e) for n,e in zip(guess,truth))
            chart=rec['expected_chart']
            if chart is not None and all(v is not None for v in pred['counts']):
                total['chart_both_returned']+=1
                total['chart_both_within_half']+=int(max(abs(n-e) for n,e in zip(pred['counts'],chart))<=.5000001)
    return dict(total,missing=total['eligible']-total['returned'],
                mean_axis_error_on_returned=float(np.mean(errors)) if errors else None)


def select(dev,methods):
    """Choose on the two development buildings, counting a wrong size against a correct size."""
    scores={}
    for method in methods:
        tables=[count_totals([r for r in dev if r['layout']==layout],method)
                for layout in sorted({r['layout'] for r in dev})]
        scores[method]=dict(balanced_score=float(np.mean([(s['correct']-s['incorrect'])/s['eligible'] for s in tables])),
                            correct=sum(s['correct'] for s in tables),wrong=sum(s['incorrect'] for s in tables))
    return dict(selected=max(methods,key=lambda m:(scores[m]['balanced_score'],scores[m]['correct'])),scores=scores,
                rule='On each development layout compute (both dimensions correct minus both attempted with an error) / eligible surfaces. Average layouts equally. Break ties by total correct then registry order.')
