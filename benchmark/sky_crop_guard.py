"""Use source-only component and atlas evidence to choose between two complete exports."""
import argparse
from pathlib import Path

from benchmark.expanded_baseline import ROOT, WORKTREES, read, save
from benchmark.relative_face_assembly import assemble

BASE = WORKTREES/'floating-visible-builds/runs/floating-cloud-gate-01'


def evidence(folder, name):
    packet = read(folder/'garden-anchor-extension/connected/sky'/name/'charts.json')
    assembly = assemble(packet['charts'],min_edges=1,edge_pixels=2.5,coplanar_pixels=.5)
    if not assembly['groups']:
        return dict(assigned=0,blocks=0,joins=[],conflicts=0)
    component = assembly['groups'][0]
    ids = set(component['origins'])
    joins = sorted(sorted(e['chart_ids']) for e in assembly['joins'] if set(e['chart_ids']) <= ids)
    return dict(assigned=len(component['assigned']),blocks=len(component['blocks']),joins=joins,
                conflicts=sum(set(e['chart_ids'])<=ids for e in assembly['conflicts']))


def accept(before, after):
    # Require additional observed material assignments, preserve established joins,
    # keep the number of fallback cells from growing, and reject additional conflicts.
    return (after['assigned']>before['assigned']
            and after['blocks']-after['assigned']<=before['blocks']-before['assigned']
            and all(j in after['joins'] for j in before['joins'])
            and after['conflicts']<=before['conflicts'])


def pooled(records):
    result = {}
    for group in ('geometry','scores'):
        for scope in ('strict','facade'):
            totals = {k:sum(r[group][scope][k] for r in records) for k in ('correct','missed','extras')}
            c,m,e = totals.values()
            result[group+'_'+scope] = totals | dict(f1=2*c/(2*c+m+e))
    return result


def compare(root):
    frames=[]
    scores = dict(baseline=[], candidate=[], guarded=[])
    for frame in sorted((BASE/'sky').iterdir()):
        name=frame.name
        before, after = evidence(BASE,name), evidence(root,name)
        chosen = accept(before,after)
        # Only now open reference-derived evaluation.
        receipt=Path('relative-face-assembly/connected-atlas/sky')/name/'score.json'
        baseline, candidate=[read(p/receipt)['score']['fixed'] for p in (BASE,root)]
        scores['baseline'].append(baseline)
        scores['candidate'].append(candidate)
        scores['guarded'].append(candidate if chosen else baseline)
        frames.append(dict(id=name,accept=chosen,before=before,after=after,
                           baseline=baseline['geometry']['strict'],candidate=candidate['geometry']['strict']))
    summary={key:pooled(rows) for key,rows in scores.items()}
    save(root/'guarded-comparison.json',dict(frames=frames,pooled=summary))
    for key,metrics in summary.items():
        print(key,[(k,round(v['f1']*100,3)) for k,v in metrics.items()])


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('arm')
    compare(ROOT/'runs/sky-crop-downstream'/cli.parse_args().arm)
