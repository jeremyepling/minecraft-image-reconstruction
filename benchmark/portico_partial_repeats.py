"""Retrospective diagnostic of unpaired repeat estimates on previously unresolved dimensions."""
import argparse
from collections import Counter
from pathlib import Path

from benchmark.camera_round import read,save
from benchmark.texture_count_round import SOURCE
from benchmark.wall_join_test import checked,read_gzip
from engine.surface_dimensions import dimensions


def run(root,out):
    source=read(SOURCE/'manifest.json')
    rows={r['key']:r for r in read(checked(SOURCE/'cases.json',source['artifacts']['cases.json']))}
    manifest=read(root/'manifest.json')
    measured=read(checked(root/'cases.json',manifest['artifacts']['cases.json']))
    counts=Counter()
    by_axis=[Counter(),Counter()]
    examples=[]
    for frame in measured:
        row=rows[frame['key']]
        assert row['sha256']==frame['sha256']
        parts={p['id']:p for p in read_gzip(checked(SOURCE/row['predictions'],source['artifacts'][row['predictions']]))['current']}
        preds=read_gzip(checked(root/frame['predictions'],manifest['artifacts'][frame['predictions']]))
        for item,p in zip(frame['surfaces'],preds):
            if 'actual' not in item:
                continue
            assert item['proposal']==p['proposal']
            picks=p['methods']['repeat']['evidence'].get('selected',[None,None])
            raw=[x['count'] if x is not None else None for x in picks]
            # Same dimensions call as the runner. No cross-axis transfer, and never combine hints into a joint guess.
            given=dimensions(parts[p['proposal']],p['chart'],(1440,900),raw)['blocks']
            for k in (0,1):
                if item['guess']['current'][k] is None:
                    counts['previously_unresolved']+=1
                    by_axis[k]['previously_unresolved']+=1
                    if given[k] is not None:
                        is_correct = given[k]==item['actual'][k]
                        name='correct' if is_correct else 'incorrect'
                        counts[name]+=1
                        by_axis[k][name]+=1
                        counts['proposed']+=1
                        by_axis[k]['proposed']+=1
                        examples.append(dict(key=row['key'],frame_sha256=row['sha256'],proposal=p['proposal'],
                                             reference=item['reference'],axis=k,guess=given[k],actual=item['actual'][k],
                                             pitch=picks[k]['pitch'],correlation=picks[k]['correlation']))
    save(out,dict(scope='Retrospective diagnostic. Report each direct translation as an independent hypothesis for '
                        'previously unresolved matched-surface axes; do not add a complete block size or transfer across axes.',
                  counts=dict(counts),horizontal=dict(by_axis[0]),vertical=dict(by_axis[1]),examples=examples))
    print(counts,by_axis)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.run,args.out)
