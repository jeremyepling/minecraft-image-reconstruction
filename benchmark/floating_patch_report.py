"""Validate decoded exports and summarize matched reference placements."""
from collections import Counter
import sys
from pathlib import Path

from benchmark.floating_patch_lattice import OUT, ROOT, digest, inputs, read, save
from benchmark.mask_blocks import decoded_pack
from benchmark.sky_crop_guard import pooled

REVIEW = ROOT/'benchmark/reviews/floating-patch-lattice-01'
ARMS = ('whole','trim','rechart','horizontal','trim-horizontal','rechart-horizontal')


def report(group):
    rows = []
    source,seed,whole,wall_pack,top_pack,cases = inputs(group)
    for case in cases:
        name = case['id']
        truth = {tuple(b['xyz_world']):b['block_id'] for b in read(Path(case['frame'])/'labels.json')['blocks']}
        ev = read(OUT/group/'evidence'/(name+'.json.gz'))
        s,boards,delta,packs = {},{},{},{}
        for arm in ARMS:
            tree = {'whole':wall_pack,'horizontal':top_pack}.get(arm,OUT/group/arm/'assembly')
            folder = tree/'connected-atlas/sky'/name
            rec = read(folder/'score.json')
            pos = read(folder/'placements.json')
            assert {tuple(p):m for p,m in pos['local']} == decoded_pack(folder/'reconstruction.zip')
            assert digest(folder/'reconstruction.zip') == rec['pack_sha256'] and rec['source_sha256'] == ev['source_sha256']
            score = rec['score']['fixed'] or rec['score']['relaxed_yaw']
            board = {tuple(p):m for p,m in pos['aligned']}
            assert sum(p in truth for p in board) == score['geometry']['strict']['correct']
            assert sum(truth.get(p)==m for p,m in board.items()) == score['scores']['strict']['correct']
            s[arm] = {k:score[k] for k in ('geometry','scores')}
            boards[arm] = set(board)
            packs[arm] = {'folder':str(folder),'sha256':rec['pack_sha256'],'origins':rec['selected_origins']}
            baseline='horizontal' if 'horizontal' in arm else 'whole'
            added = set(board)-boards[baseline]
            removed = boards[baseline]-set(board)
            delta[arm] = dict(recovered=len(added & truth.keys()),lost=len(removed & truth.keys()),
                              extras_added=len(added-truth.keys()),extras_removed=len(removed-truth.keys()))
        local = Counter()
        for c in ev['charts']:
            local.update(regions=len(c['regions']),rejected=sum(r['rejected'] for r in c['regions']),
                         source_cells_removed=c['before']-c['after'],new_charts=len(c['new_charts']),
                         new_cells=sum(r['cells'] for r in c['new_charts']))
        rows.append(dict(id=name,layout=case['layout'],dataset=case['dataset'],source=ev['source_image'],
                         hashes={k:ev[k] for k in ('source_sha256','seed_sha256','whole_sha256','mask_sha256')},
                         inputs={k:ev[k] for k in ('mask','seed_chart','whole_chart')},
                         evidence=str(OUT/group/'evidence'/(name+'.json.gz')),counts=local,
                         reference=rec['reference_sha256'],scores=s,packs=packs,delta=delta))
    def summarize(rs):
        scores={a:pooled([r['scores'][a] for r in rs]) for a in ARMS}
        counts=sum((Counter(r['counts']) for r in rs),Counter())
        pairs={}
        for a in ARMS:
            base='horizontal' if 'horizontal' in a else 'whole'
            d=[r['scores'][a]['geometry']['strict']['f1']-r['scores'][base]['geometry']['strict']['f1'] for r in rs]
            pairs[a]=dict(**sum((Counter(r['delta'][a]) for r in rs),Counter()),wins=sum(x>0 for x in d),losses=sum(x<0 for x in d),ties=sum(x==0 for x in d))
        return dict(scores=scores,counts=counts,pairs=pairs)
    summary=summarize(rows)
    summary['datasets']={ds:summarize([r for r in rows if r['dataset']==ds]) for ds in sorted({r['dataset'] for r in rows})}
    save(REVIEW/(group+'.json.gz'),dict(freeze='3630fc5',frames=rows,summary=summary))
    print(group,summary['counts'])
    for a,m in summary['scores'].items():
        print(a,[round(100*t['f1'],2) for t in m.values()],summary['pairs'][a])


if __name__=='__main__':
    for cohort in sys.argv[1:]:
        report(cohort)
