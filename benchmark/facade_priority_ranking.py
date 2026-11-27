"""Rank the same complete layouts after neutralizing only individually thin horizontal faces."""
from pathlib import Path

from benchmark.facade_visibility import rates
from benchmark.texture_camera_eval import digest,read,save

FOLDER=Path('benchmark/reviews/strip-orientation-01')
LAYOUTS=Path('benchmark/reviews/facade-placement-01/worst/ranking.json')


def rank(audit_file,out):
    audit=read(audit_file)
    cases={r['id']:r for r in audit['frames'] if r['split']!='extra'}
    layouts=[]
    for layout in read(LAYOUTS)['all_layouts']:
        frames=[]
        for key in layout['frame_ids']:
            r=cases[key]
            hit,miss,*_=r['stats']['face_4_horizontal']
            correct=r['correct']-hit
            missed=r['missed']-miss
            frames.append(dict(id=key,correct=correct,missed=missed,extras=r['extras'],
                               neutral_hits=hit,neutral_misses=miss,f1=rates(correct,missed,r['extras'])['f1']))
        totals={k:sum(f[k] for f in frames) for k in ('correct','missed','extras','neutral_hits','neutral_misses')}
        f1=rates(totals['correct'],totals['missed'],totals['extras'])['f1']
        layouts.append(dict(layout=layout['name'],f1=f1,frames=frames,**totals))
    assert sum(len(v['frames']) for v in layouts)==len(cases)==104
    rows=sorted(layouts,key=lambda v:(v['f1'],v['layout']))
    save(out,dict(sliver_audit_sha256=digest(audit_file),layout_identity_sha256=digest(LAYOUTS),layouts=rows))
    for row in rows[:10]:
        print(row['layout'],len(row['frames']),round(100*row['f1'],2),row['missed'],row['extras'])


if __name__=='__main__':
    rank(FOLDER/'sliver-counts.json',FOLDER/'priority-layouts.json')
