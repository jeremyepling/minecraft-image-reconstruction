"""Compare actual exported sites for the surface feedback interventions."""
from collections import Counter, defaultdict
import argparse
import gzip
import json
from pathlib import Path

from benchmark.reference_surface_feedback import MODES, ROOT, releases, typed, SUITE


def compare(before, after):
    def sets(row):
        placed = {tuple(p) for p, _, actual, _ in row['points'] if actual}
        extras = {tuple(p) for p, _, actual, _ in row['points'] if actual is None}
        right = {tuple(p) for p, name, actual, _ in row['points'] if name == actual}
        return placed, extras, right
    a, x, c = sets(before)
    b, y, d = sets(after)
    return dict(gained=len(b-a), lost=len(a-b), extras_added=len(y-x), extras_removed=len(x-y),
                typed_gained=len(d-c), typed_lost=len(c-d))


def summarize(folder,out,count):
    expected={}
    for j in SUITE.jobs([*SUITE.framed.SIZES,'shelter']):
        expected.setdefault(j[0],[]).append(j[1])
    ids={f['id'] for frames in expected.values() for f in sorted(frames,key=lambda r:r['files']['rgb.png'])[:count]}
    grouped=defaultdict(dict)
    for path in folder.glob('*.json'):
        row=json.loads(path.read_bytes())
        if row['id'] in ids:
            grouped[row['id']][row['mode']]=row
    if grouped.keys()!=ids or any(set(group)!=set(MODES) for group in grouped.values()):
        raise ValueError('Missing or additional preregistered view or intervention')
    release,_=releases()
    cases=[]
    for k,group in sorted(grouped.items()):
        ctl=group['baseline']
        ref=release[k]
        assert ctl['source_sha256']==ref['source_hash']
        case=dict(id=k, split=ctl['split'],source_sha256=ctl['source_sha256'],published_pack=ref['pack_sha256'],
                  reproduced_pack=ctl['pack_sha256']==ref['pack_sha256'],published_geometry=ref['geometry'],
                  published_scores=ref['result'], modes={})
        for m,row in group.items():
            assert row['source_sha256']==ctl['source_sha256']
            case['modes'][m] = {key:row[key] for key in ('pack_sha256','regions','no_overlap_donor','natural_span_exact',
                                                       'geometry','scores','translation')}
            case['modes'][m]['changes']=compare(ctl,row)
        cases.append(case)
    result={}
    for cohort in ('all','fixed','other'):
        frames=[c for c in cases if cohort=='all' or (c['split'] in SUITE.framed.EVALUATION)==(cohort=='fixed')]
        group={}
        for m in MODES:
            records=[c['modes'][m] for c in frames]
            geom=typed.pooled_scores([r['geometry'] for r in records])
            strict=typed.pooled_scores([{'strict':r['scores']['strict'],'facade':r['scores']['facade']} for r in records])
            correct=sum(r['scores']['material']['correct'] for r in records)
            positions=sum(r['scores']['material']['positions'] for r in records)
            group[m]=dict(views=len(frames),geometry=geom,typed=strict,material={'correct':correct,'positions':positions,'accuracy':correct/positions if positions else None},
                          changes=dict(sum((Counter(r['changes']) for r in records),Counter())),
                          exact_spans=sum(r['natural_span_exact'] or 0 for r in records),
                          regions=sum(r['regions'] for r in records),
                          improved=sum(r['geometry']['strict']['f1']>c['modes']['baseline']['geometry']['strict']['f1'] for r,c in zip(records,frames)),
                          worse=sum(r['geometry']['strict']['f1']<c['modes']['baseline']['geometry']['strict']['f1'] for r,c in zip(records,frames)))
        result[cohort]=group
    handoffs={}
    for a,b in zip(('mask','directions','chart','grids'),('directions','chart','grids','joint')):
        handoffs[a+' -> '+b]=dict(sum((Counter(compare(g[a],g[b])) for g in grouped.values()),Counter()))
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('cases',cases),('summary',dict(cohorts=result,paired_handoffs=handoffs,verified_packs=sum(c['reproduced_pack'] for c in cases)))]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    points={k:{m:r['points'] for m,r in sorted(g.items())} for k,g in sorted(grouped.items())}
    payload=json.dumps(points,separators=(',',':')).encode()
    (out/'scored-sites.json.gz').write_bytes(gzip.compress(payload,mtime=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--in', dest='folder',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT)
    p.add_argument('--per-split',type=int,default=2)
    a=p.parse_args()
    summarize(a.folder,a.out,a.per_split)

