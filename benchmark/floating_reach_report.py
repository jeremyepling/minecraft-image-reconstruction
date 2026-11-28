"""Audit decoded wall-reach exports against visible frame labels, then pool the four placement metrics."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from benchmark.floating_grid_reach import ARMS, OUT, ROOT, inputs, digest, read, save
from benchmark.mask_blocks import decoded_pack
from benchmark.sky_crop_guard import pooled

REVIEW=ROOT/'benchmark/reviews/floating-grid-reach-01'
METRICS=('geometry_strict','geometry_facade','scores_strict','scores_facade')
ALL=('ordinary',*ARMS)


def report(cohort):
    rows=[]
    root=OUT/cohort
    for case in inputs(cohort)[3]:
        name=case['id']
        frame=Path(case['frame'])
        truth={tuple(b['xyz_world']):b['block_id'] for b in read(frame/'labels.json')['blocks']}
        scores,geos,paired={},{},{}
        e=read(root/'evidence'/(name+'.json'))
        for a in ALL:
            folder=root/a/'assembly/connected-atlas/sky'/name
            rec=read(folder/'score.json')
            coords=read(folder/'placements.json')
            assert {tuple(p):m for p,m in coords['local']}==decoded_pack(folder/'reconstruction.zip')
            assert rec['pack_sha256']==digest(folder/'reconstruction.zip') and rec['source_sha256']==e['source_sha256']
            board={tuple(p):m for p,m in coords['aligned']}
            s=rec['score']['fixed'] or rec['score']['relaxed_yaw']
            assert sum(p in truth for p in board)==s['geometry']['strict']['correct']
            assert sum(m==truth.get(p) for p,m in board.items())==s['scores']['strict']['correct']
            scores[a]=dict(score=s,pack_sha256=rec['pack_sha256'],chart_sha256=rec['chart_sha256'])
            geos[a]=set(board)
        for a in ALL:
            b=geos['whole']
            v=geos[a]
            paired[a]=dict(recovered=len((v-b)&truth.keys()),lost=len((b-v)&truth.keys()),extras_added=len((v-b)-truth.keys()),extras_removed=len((b-v)-truth.keys()))
        rows.append(dict(**e,layout=case.get('layout',name),dataset=case['dataset'],split=case['split'],arms=scores,paired=paired,reference_sha256=rec['reference_sha256']))
    groups={'all':rows}
    for r in rows:
        for k in ('dataset','split','layout'):
            groups.setdefault(k+'/'+r[k],[]).append(r)
    summary={}
    for k,rs in groups.items():
        arms={a:pooled([r['arms'][a]['score'] for r in rs]) for a in ALL}
        paired={}
        for a in ALL:
            counts=Counter()
            for r in rs:
                counts.update(r['paired'][a])
            for metric in ('geometry','scores'):
                d=[r['arms'][a]['score'][metric]['strict']['f1']-r['arms']['whole']['score'][metric]['strict']['f1'] for r in rs]
                counts.update({metric+'_gains':sum(x>0 for x in d),metric+'_losses':sum(x<0 for x in d),metric+'_ties':sum(x==0 for x in d)})
            paired[a]=counts
        summary[k]=dict(views=len(rs),arms=arms,paired=paired)
    save(REVIEW/(cohort+'.json.gz'),dict(frozen='d6133fe',frames=rows,groups=summary))
    for k,g in summary.items():
        if not k.startswith('layout/'):
            print(cohort,k,g['views'],{a:[round(100*m[n]['f1'],2) for n in METRICS] for a,m in g['arms'].items()})
    print('Sparse2 vs whole',summary['all']['paired']['sparse2'])


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('cohorts',nargs='+')
    for c in parser.parse_args().cohorts:
        report(c)
