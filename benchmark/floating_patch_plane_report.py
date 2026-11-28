"""Audit each datapack and summarize paired visible positions and exact material IDs."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.expanded_baseline import digest, read, save
from benchmark.floating_patch_planes import COHORTS, OUT, ROOT, inputs
from benchmark.mask_blocks import decoded_pack
from benchmark.sky_crop_guard import pooled
from engine.geometry import project

REVIEW = ROOT/'benchmark/reviews/floating-patch-planes-01'
ARMS = ('whole','planes','opposed')
METRICS = ('geometry_strict','geometry_facade','scores_strict','scores_facade')


def score(row, name):
    return row['arms'][name]['score']


def summarize(rows):
    groups = {'all':rows}
    for row in rows:
        for category in ('dataset','split','layout'):
            groups.setdefault(category+'/'+row[category], []).append(row)
    summary = {}
    for name, group in groups.items():
        arms = {}
        for arm in ARMS:
            scores = [score(r,arm) for r in group]
            supported = [r for r in group if r['arms'][arm]['camera']=='resolved']
            failures = [r for r in group if r['arms'][arm]['camera']!='resolved']
            m = [s['scores'] for s in scores]
            hits = sum(s['material']['positions'] for s in m)
            ids = sum(s['material']['correct'] for s in m)
            metrics = pooled(scores)
            strict = metrics['geometry_strict']
            # Sensitivity with unresolved/absent upright orientation counted as a complete miss.
            failed_hits = sum(score(r,arm)['geometry']['strict']['correct'] for r in failures)
            correct = strict['correct']-failed_hits
            charged_f1 = 2*correct/(2*strict['correct']+strict['extras']+strict['missed'])
            arms[arm] = dict(metrics, assigned=sum(s['assigned'] for s in m),fallback=sum(s['fallback'] for s in m),
                             correct_ids=ids, exact_id_accuracy=ids/hits if hits else None,
                             supported_views=len(supported),failed_views=len(failures),
                             failed_visible_targets=sum(r['visible'] for r in failures),
                             failed_exported=sum(r['arms'][arm]['blocks'] for r in failures),
                             supported_empty=sum(r['arms'][arm]['blocks']==0 for r in supported),
                             supported_scores=pooled([score(r,arm) for r in supported]) if supported else None,
                             failure_charged_strict_f1=charged_f1)
        pairs = Counter()
        for r in group:
            for arm in ARMS[1:]:
                pairs.update({arm+'_'+key:count for key,count in r['paired'][arm].items()})
        summary[name] = dict(views=len(group),arms=arms,paired=pairs)
    return summary


def report(cohort):
    _, _, whole_charts, assembly, cases = inputs(cohort)
    candidate = OUT/cohort/'planes'
    rows=[]
    for case in cases:
        name = case['id']
        truth_path = Path(case['frame'])
        truth = {tuple(b['xyz_world']):b['block_id'] for b in read(truth_path/'labels.json')['blocks']}
        e = read(candidate/'evidence'/(name+'.json'))
        hashes = {f:digest(truth_path/f) for f in ('camera.json','labels.json','labels.npz')}
        assert e['source_sha256'] == digest(Path(case['variants']['sky']['image']))
        arms, boards = {}, {}
        locations = [('whole',assembly,whole_charts)]
        for arm in ARMS[1:]:
            arm_root = OUT/cohort/arm
            locations.append((arm,arm_root/'assembly',arm_root/'charts'))
        for arm, folder, chart_root in locations:
            source = folder/'connected-atlas/sky'/name
            chart = chart_root/'sky'/name/'charts.json'
            rec = read(source/'score.json')
            pos = read(source/'placements.json')
            assert rec['reference_sha256']==hashes and rec['source_sha256']==e['source_sha256']
            assert rec['chart_sha256']==digest(chart) and rec['pack_sha256']==digest(source/'reconstruction.zip')
            assert decoded_pack(source/'reconstruction.zip')=={tuple(x):m for x,m in pos['local']}
            board = {tuple(x):m for x,m in pos['aligned']}
            s = rec['score']['fixed'] or rec['score']['relaxed_yaw']
            hit = board.keys() & truth.keys()
            assert len(hit)==s['geometry']['strict']['correct'] and sum(board[p]==truth[p] for p in hit)==s['scores']['strict']['correct']
            camera = 'absent' if rec['yaw'] is None else 'ambiguous' if rec['yaw']['yaw_quarters'] is None else 'resolved'
            arms[arm]=dict(score=s,camera=camera,blocks=len(board),chart_sha256=rec['chart_sha256'],pack_sha256=rec['pack_sha256'])
            boards[arm]=board
        paired = {}
        for arm in ARMS[1:]:
            before,after = boards['whole'].keys(), boards[arm].keys()
            gained, lost = after-before, before-after
            pair=dict(recovered=len(gained&truth.keys()),lost=len(lost&truth.keys()),extras_added=len(gained-truth.keys()),extras_removed=len(lost-truth.keys()))
            for metric in ('geometry','scores'):
                delta=arms[arm]['score'][metric]['strict']['f1']-arms['whole']['score'][metric]['strict']['f1']
                pair[metric+('_gain' if delta>0 else '_loss' if delta<0 else '_tie')]=1
            paired[arm] = pair
        rows.append(dict(**e,dataset=case['dataset'],layout=case.get('layout',name),split=case['split'],visible=len(truth),reference_sha256=hashes,arms=arms,paired=paired))
    groups=summarize(rows)
    save(REVIEW/(cohort+'.json.gz'),dict(frozen=dict(planes='e9f2683',opposed='47d0603'),frames=rows,groups=groups))
    for name in ('all',):
        g=groups[name]
        print(cohort,g['views'],{arm:[round(100*g['arms'][arm][metric]['f1'],2) for metric in METRICS] for arm in ARMS},g['paired'])


def overlay():
    cases=[('sky','validation-hillside-chapel-1-warm-low-left'),('supplemental','train-observatory-02-verdigris-high-right')]
    sheet=Image.new('RGB',(920,320),'white')
    draw=ImageDraw.Draw(sheet)
    for i,(cohort,name) in enumerate(cases):
        _,_,base,_,rows=inputs(cohort)
        case=next(r for r in rows if r['id']==name)
        im=Image.open(case['variants']['sky']['image']).convert('RGB')
        before=read(base/'sky'/name/'charts.json')
        after=read(OUT/cohort/'planes/charts/sky'/name/'charts.json')
        d=ImageDraw.Draw(im)
        points=[]
        for b,a in zip(before['charts'],after['charts']):
            assert b['id']==a['id']
            kept={(c['u'],c['v']) for c in a['cells']}
            for c in b['cells']:
                key=(c['u'],c['v'])
                x,y=project(np.array(b['H']),np.array(key)+.5)
                points.append((x,y))
                color='#16936e' if key in kept else '#fc724b'
                d.ellipse((x-2,y-2,x+2,y+2), fill=color)
        xy=np.array(points)
        center=(xy.max(axis=0)+xy.min(axis=0))/2
        scale=max(np.ptp(xy[:,0])/420,np.ptp(xy[:,1])/230)
        crop=tuple(np.r_[center-[224*scale,123*scale],center+[224*scale,123*scale]].round().astype(int))
        sheet.paste(im.crop(crop).resize((448,246),Image.Resampling.LANCZOS),(6+460*i,36))
        draw.text((12+460*i,12),['Chapel, sky composite','Observatory, original RGB'][i],fill='#222222')
    draw.text((12,294),'Green: retained wall cells. Orange: removed wall cells. Dots mark projected cell centers.',fill='#222222')
    sheet.save(REVIEW/'source-overlays.jpg',quality=91)


def main():
    for cohort in COHORTS:
        report(cohort)
    overlay()


if __name__=='__main__':
    main()
