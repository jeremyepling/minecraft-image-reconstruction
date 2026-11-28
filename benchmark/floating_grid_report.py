"""Score already exported packs. The finite oracle maximizes pooled strict position F1."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from benchmark.expanded_baseline import SOURCE, read, save, digest
from benchmark.floating_grid_evidence import ARMS, OUT, ROOT
from benchmark.mask_blocks import decoded_pack
from benchmark.sky_crop_guard import pooled

REVIEW=ROOT/'benchmark/reviews/floating-grid-evidence-01'


def summarize(rows):
    results={arm:pooled([r['arms'][arm]['score'] for r in rows]) for arm in (*ARMS,'guarded')}
    ratio=results['current']['geometry_strict']['f1']
    for i in range(100):
        chosen=[]
        for r in rows:
            def gain(arm):
                s=r['arms'][arm]['score']['geometry']['strict']
                return 2*s['correct']-ratio*(2*s['correct']+s['missed']+s['extras'])
            chosen.append(max((*ARMS,'guarded'),key=gain))
        oracle=pooled([r['arms'][arm]['score'] for r,arm in zip(rows,chosen)])
        new=oracle['geometry_strict']['f1']
        if abs(ratio-new)<1e-12:
            break
        ratio=new
    else:
        raise RuntimeError('Finite pooled oracle failed to converge')
    changes=Counter()
    for row in rows:
        changes.update(row['paired'])
    return dict(views=len(rows),arms=results|{'oracle':oracle},finite_oracle_choices=dict(zip([r['id'] for r in rows],chosen)),paired=changes)


def report(cohort):
    root=OUT/cohort
    rows=[]
    index={r['id']:(ds,r) for ds in ('visible-builds-v2','visible-builds-v3','decorative-builds-v1')
        for r in read(SOURCE/ds/'manifest.json')['frames']}
    for file in sorted((root/'evidence').glob('*.json')):
        name=file.stem
        ds,case=index[name]
        frame=SOURCE/ds/'data'/case['split']/'frames'/name
        reference={tuple(b['xyz_world']):b['block_id'] for b in read(frame/'labels.json')['blocks']}
        truth=set(reference)
        ev=read(file)
        assert ev['source_sha256']==digest(frame/'rgb.png')==case['rgb_sha256']
        arms,geos={},{}
        for arm in (*ARMS,'guarded'):
            pack=root/arm/'assembly/connected-atlas/sky'/name
            record=read(pack/'score.json')
            coords=read(pack/'placements.json')
            assert dict((tuple(p),m) for p,m in coords['local'])==decoded_pack(pack/'reconstruction.zip')
            assert record['pack_sha256']==digest(pack/'reconstruction.zip') and record['source_sha256']==case['rgb_sha256']
            score=record['score']['fixed'] or record['score']['relaxed_yaw']
            geos[arm]={tuple(p) for p,m in coords['aligned']}
            assert len(geos[arm]&truth)==score['geometry']['strict']['correct']
            assert sum(reference.get(tuple(p))==m for p,m in coords['aligned'])==score['scores']['strict']['correct']
            arms[arm]=dict(score=score,pack_sha256=record['pack_sha256'],chart_sha256=record['chart_sha256'])
        before,after=geos['current'],geos['selected']
        paired=dict(recovered=len((after-before)&truth),lost=len((before-after)&truth),
            extras_added=len((after-before)-truth),extras_removed=len((before-after)-truth))
        guard=read(root/'guarded/evidence'/(name+'.json'))
        if not any(c['accepted'] for c in guard):
            assert read(root/'current/assembly/connected-atlas/sky'/name/'placements.json')==read(root/'guarded/assembly/connected-atlas/sky'/name/'placements.json')
        rows.append(dict(guard=guard,id=name,dataset=ds,layout=case['layout'],split=case['split'],**ev,
            reference_sha256=record['reference_sha256'],arms=arms,paired=paired))
    assert len(rows)==(38 if cohort=='original' else 12)
    groups={'all':rows}
    for key in ('dataset','split','layout'):
        for r in rows:
            groups.setdefault(key+'/'+r[key],[]).append(r)
    summary={group:summarize(rs) for group,rs in groups.items()}
    save(REVIEW/(cohort+'.json.gz'),dict(frozen_selector='2d064e7',frames=rows,groups=summary))
    for g,s in summary.items():
        print(g,s['views'],{a:[round(100*m[v]['f1'],2) for v in ('geometry_strict','geometry_facade','scores_strict','scores_facade')] for a,m in s['arms'].items()})
    print(summary['all']['paired'],Counter(c['chosen'] for r in rows for c in r['charts']))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('cohort',choices=('original','fresh'))
    report(parser.parse_args().cohort)


