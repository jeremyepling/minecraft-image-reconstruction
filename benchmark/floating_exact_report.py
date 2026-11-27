"""Collect paired floating-grid placement counts, gauge metamorphisms and oracle gains."""
from collections import defaultdict

from benchmark.floating_exact import METHODS, OUTPUT, SHARED, read, write


def confusions(row):
    return [row['correct'],row['missed'],row['extras']]


def f1(values):
    correct,missed,extras=values
    return 2*correct/max(1,2*correct+missed+extras)


def aggregate(rows):
    totals=[sum(group) for group in zip(*(confusions(s['geometry']['strict']) for s in rows))]
    return dict(n=len(rows),strict_counts=totals,strict_f1=f1(totals) if totals else None)


def local_world(prediction,origin):
    return {tuple(a+b for a,b in zip(p,origin)):name for p,name in prediction['local_blocks']}


def report():
    cases=read(SHARED/'fixtures.json')['cases']
    results=defaultdict(list)
    comparisons=[]
    invariant=[]
    for entry in cases:
        for variant in entry['variants']:
            scores,predictions={},{}
            for method in (*METHODS,'exact-distance-sky'):
                folder=OUTPUT/method/variant/entry['id']
                if not (folder/'score.json').exists():
                    continue
                s=read(folder/'score.json')
                assert s['source_sha256']==entry['variants'][variant]['source_sha256']
                assert s['reference_sha256']==entry['reference_sha256']
                scores[method]=s
                predictions[method]=read(folder/'prediction.json')
                results[method,variant].append(s)
            for root in ('exact-floor','exact-floating'):
                a,b=root+'-base',root+'-roof'
                if not {a,b}<=scores.keys():continue
                ab,bb=(local_world(predictions[m], scores[m]['projection_checks']['origin_world']) for m in (a,b))
                invariant.append(dict(id=entry['id'],variant=variant,policy=root,common=len(ab.keys()&bb.keys()),
                    lost=len(ab.keys()-bb.keys()),gained=len(bb.keys()-ab.keys()),id_changes=sum(ab[k]!=bb[k] for k in ab.keys()&bb.keys())))
            for a,b in (('exact-floor-base','exact-floating-base'),('exact-floating-base','exact-distance-sky')):
                if not {a,b}<=scores.keys():continue
                from benchmark.expanded_scores import load_reference
                truth=load_reference(entry).truth
                aa,bb=({tuple(p) for p,_ in predictions[m]['aligned']} for m in (a,b))
                comparisons.append(dict(id=entry['id'],variant=variant,start=a,end=b,
                    recovered=len((bb-aa)&truth.keys()),lost=len((aa-bb)&truth.keys()),
                    extra_added=len((bb-aa)-truth.keys()),extra_removed=len((aa-bb)-truth.keys())))
    data=dict(summary={f'{m}/{v}':aggregate(rows) for (m,v),rows in sorted(results.items())},
              comparisons=comparisons,origin_pairs=invariant)
    write(OUTPUT/'summary.json',data)
    print(data['summary'])


if __name__=='__main__':
    report()
