"""Report paired joined-wall exports, including an oracle over the measured alternatives."""
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

from benchmark.expanded_baseline import ROOT, SOURCE, digest, read, save
from benchmark.floating_join_evidence import ARMS, OUTPUT
from benchmark.sky_crop_guard import pooled


REVIEW = ROOT/'benchmark/reviews/floating-join-evidence-01'


def best_pooled(rows, group):
    # Maximize pooled F1 as one ratio, allowing a separate observed option
    # per image. Price each additional predicted block at the current ratio.
    value = 0.
    while True:
        def objective(option):
            s = option['score'][group]['strict']
            return 2*s['correct'] - value*(2*s['correct']+s['missed']+s['extras'])
        winners = [max(r['scores'], key=lambda a:objective(r['scores'][a])) for r in rows]
        score = pooled([r['scores'][a]['score'] for r,a in zip(rows,winners)])[group+'_strict']['f1']
        if abs(score-value)<1e-12:
            return winners
        value = score


def summarize(rows):
    result = dict(views=len(rows), proposals=sum(len(r['proposals']) for r in rows),
                  frames_without_perpendicular_joins=sum(not r['proposals'] for r in rows),
                  missing_from_every_option=sum(r['missing_from_every_option'] for r in rows), arms={})
    choices = {a:[a]*len(rows) for a in ARMS}
    for g in ('geometry','scores'):
        choices['oracle_'+g] = best_pooled(rows,g)
    for arm,winners in choices.items():
        selected = [r['scores'][a]['score'] for r,a in zip(rows,winners)]
        comparison = Counter()
        for row, winner in zip(rows,winners):
            chosen = row['selected'][winner]
            for k in ('gained','lost','extras_added','extras_removed','components','joins','conflicts'):
                comparison[k] += chosen[k]
            comparison['kept_proposals'] += len(row['options'][winner])
            comparison['kept_charts'] += len(chosen['origins'])
            for g in ('geometry','scores'):
                delta = row['scores'][winner]['score'][g]['strict']['f1'] - row['scores']['existing']['score'][g]['strict']['f1']
                comparison[g+('_improved' if delta>0 else '_worsened' if delta<0 else '_tied')] += 1
        tally = pooled(selected)
        for k in ('assigned','fallback'):
            tally[k] = sum(s['scores'][k] for s in selected)
        tally['correct_ids_on_correct_positions'] = sum(s['scores']['material']['correct'] for s in selected)
        tally['paired'] = dict(comparison)
        if arm.startswith('oracle'):
            tally['chosen'] = dict(zip((r['id'] for r in rows), winners))
        result['arms'][arm] = tally
    return result


def draw_sheet(groups):
    samples = [('prior','test-twinspire3-cool-high-right'), ('prior','test-twinspire3-earth-low-left')]
    garden = [r for r in groups['ideal'] if r['dataset']=='decorative-builds-v1']
    samples.append(('ideal',max(garden,key=lambda r:r['scores']['existing']['score']['geometry']['strict']['f1'] -
                                  r['scores']['two_distinct']['score']['geometry']['strict']['f1'])['id']))
    if 'fresh' in groups:
        samples.append(('fresh', max(groups['fresh'],key=lambda r:abs(r['scores']['existing']['score']['geometry']['strict']['f1'] -
                           r['scores']['supported']['score']['geometry']['strict']['f1']))['id']))
    sheet = Image.new('RGB',(950,len(samples)*310+38),'#182026')
    caption = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    labels = dict(existing='Current one-edge rule',two_distinct='Two distinct edges',
                  separated='Two separated edges',supported='Mask and depth evidence',
                  bypass='Perpendicular joins disabled')
    scope = dict(prior='Original source grids',ideal='Exact wall grids and masks',fresh='New-layout source grids')
    caption.text((12,8),'Proposed shared edges       Green = passes the mask and depth check       Red = rejected',font=font,fill='white')
    for index,(group,name) in enumerate(samples):
        r = next(x for x in groups[group] if x['id']==name)
        src = SOURCE/r['dataset']/'data'/r['split']/'frames'/name/'rgb.png'
        assert digest(src)==r['source_sha256']
        with Image.open(src) as original:
            image = original.convert('RGB')
            image.thumbnail((515,265))
            scale = image.width/original.width
        d = ImageDraw.Draw(image)
        for p in r['proposals']:
            color = '#41f5b4' if p['supported'] else '#ff5f56'
            for segment in p['segments']:
                d.line([(x*scale,y*scale) for x,y in segment],fill=color,width=3)
        top=38+index*310
        sheet.paste(image,(12,top+31))
        caption.text((12,top+5),scope[group]+'  /  '+name,font=font,fill='white')
        for j,arm in enumerate(('existing','two_distinct','separated','supported','bypass')):
            s=r['scores'][arm]['score']['geometry']['strict']
            caption.text((480,top+47+j*43),labels[arm],font=font,fill='white')
            caption.text((706,top+47+j*43),f"{s['f1']*100:.2f}%   ({s['correct']} correct)",font=font,fill='white')
    sheet.save(REVIEW/'junctions.jpg',quality=90)


def report():
    groups={}
    for group in ('prior','ideal','fresh'):
        rows = [read(p) for p in sorted((OUTPUT/group).glob('*/result.json.gz'))]
        if not rows:
            continue
        groups[group] = rows
        cohorts = {key:summarize([r for r in rows if key in ('all',r['dataset'],r['layout'])])
                   for key in ('all',*sorted({r['dataset'] for r in rows}),*sorted({r['layout'] for r in rows}))}
        save(REVIEW/(group+'.json.gz'),dict(cases=rows,cohorts=cohorts))
        for key in ('all','visible-builds-v2','visible-builds-v3','decorative-builds-v1'):
            if key in cohorts:
                score=cohorts[key]
                print(group,key,score['views'],score['proposals'],'candidates;',{
                    a:tuple(round(100*record[k]['f1'],3) for k in ('geometry_strict','geometry_facade','scores_strict','scores_facade'))
                    for a,record in score['arms'].items()})
    draw_sheet(groups)


if __name__ == '__main__':
    report()
