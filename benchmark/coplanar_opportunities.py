"""Scoring-only inventory of opportunities to borrow a complete dimension pair."""
import argparse
from collections import Counter
from pathlib import Path

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.normal_round import groups
from benchmark.surface_reference import recorded
from benchmark.wall_join_test import checked

BASE=ROOT/'benchmark/reviews/texture-photo-count-01'


def run(out):
    captures={c.floor+'/'+c.id:c for group in groups()[0].values() for c in group}
    rows=read(checked(BASE/'cases.json',read(BASE/'manifest.json')['artifacts']['cases.json']))
    cases=[]
    for frame in rows:
        _,ref=recorded(captures[frame['key']])
        known=[s for s in frame['surfaces'] if 'actual' in s]
        plane={s['reference']:(tuple(ref['patches'][s['reference']]['normal']),
                               ref['patches'][s['reference']]['offset']) for s in known}
        examples=[]
        for target in known:
            if None not in target['guess']['current']:
                continue
            rid=target['reference']
            siblings=[s for s in known if s is not target and plane[s['reference']]==plane[rid]]
            complete=[s for s in siblings if None not in s['guess']['current']]
            exact=[s for s in complete if s['guess']['current']==s['actual']]
            different=[s for s in exact if target['materials'].keys().isdisjoint(s['materials'])]
            examples.append(dict(reference=rid,same_plane=[s['reference'] for s in siblings],
                                 complete=[s['reference'] for s in complete],exact=[s['reference'] for s in exact],
                                 different_material_exact=[s['reference'] for s in different]))
        cases.append(dict(key=frame['key'],layout=frame['layout'],sha256=frame['sha256'],targets=examples))
    totals={}
    for group in ['all',*sorted({f['layout'] for f in cases})]:
        counts=Counter()
        for frame in cases:
            if group not in ('all',frame['layout']):
                continue
            for t in frame['targets']:
                counts['unresolved_matched']+=1
                for k in ('same_plane','complete','exact','different_material_exact'):
                    counts[k]+=int(bool(t[k]))
        totals[group]=dict(counts)
    out.parent.mkdir(parents=True,exist_ok=True)
    save(out,dict(scope='Scoring only. Distinct matched connected components on the same oriented Minecraft plane in the same screenshot. Counts recipient appearances with at least one donor. Correct donor size need not imply correct grid origin. Other types of donor are outside this inventory.',
                  source_cases_sha256=digest(BASE/'cases.json'),totals=totals,cases=cases))
    print(totals['all'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args().out)
