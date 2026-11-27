"""Exact whole-view chooser for two frozen reconstruction options."""
from fractions import Fraction
from collections import Counter

from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.mask_blocks import decoded_pack
from benchmark.sky_crop_guard import pooled


def best_choice(rows):
    # For a proposed pooled ratio r, maximize sum(2TP - r * (2TP + FP + FN)).
    # Choices are independent per image. Zero maximum proves global optimality.
    ratio=Fraction(0)
    while True:
        chosen=[]
        for row in rows:
            def objective(arm):
                counts=row[arm]['score']['geometry']['strict']
                c,m,e=[counts[k] for k in ('correct','missed','extras')]
                return 2*c*ratio.denominator - ratio.numerator*(2*c+m+e)
            chosen.append('native' if objective('native')>objective('source') else 'source')
        counts=Counter()
        for row,arm in zip(rows,chosen):
            counts.update({k:row[arm]['score']['geometry']['strict'][k] for k in ('correct','missed','extras')})
        c,m,e=[counts[k] for k in ('correct','missed','extras')]
        improved=Fraction(2*c,2*c+m+e)
        if ratio==improved:
            return chosen
        ratio=improved


def summarize(folder):
    rows=[read(p) for p in sorted(folder.glob('pairs/*/*.json'))]
    if len(rows)!=38:
        raise ValueError(f'Expected 38 complete pairs, found {len(rows)}')
    for row in rows:
        for arm in ('source','native'):
            grouped=folder/arm
            receipt=read(grouped/'assembly/connected-atlas/sky'/row['id']/'score.json')
            export=grouped/'assembly/connected-atlas/sky'/row['id']
            pack=export/'reconstruction.zip'
            local=read(export/'placements.json')['local']
            assert digest(pack)==receipt['pack_sha256']==row[arm]['pack_sha256']
            assert decoded_pack(pack)=={tuple(xyz):material for xyz,material in local}
            counts=read(grouped/'sky'/row['id']/'source-handoff/counts.json')['surfaces']
            dimensions=[item.get('counts') or [None,None] for item in counts]
            row[arm]['counts_both_axes']=sum(all(v is not None for v in d) for d in dimensions)
            row[arm]['counts_any_axis']=sum(any(v is not None for v in d) for d in dimensions)
            row[arm]['kept_charts']=receipt['kept_charts']
            row[arm]['components']=receipt['components']
            row[arm]['yaw_estimated_for_score']=receipt['score']['fixed'] is not None
    by_dataset={}
    for cohort in ('all',*sorted({r['dataset'] for r in rows})):
        batch=[r for r in rows if cohort=='all' or r['dataset']==cohort]
        selected=best_choice(batch)
        ans={arm:pooled([r[arm]['score'] for r in batch]) for arm in ('source','native')}
        ans['chooser']=pooled([r[arm]['score'] for r,arm in zip(batch,selected)])
        by_dataset[cohort]=dict(views=len(batch),scores=ans,choose_native=[r['id'] for r,arm in zip(batch,selected) if arm=='native'])
        print(cohort,len(batch),{arm:[round(100*m['f1'],2) for m in score.values()] for arm,score in ans.items()})
    save(ROOT/'benchmark/reviews/expanded-crop-ceiling-01/scored-exports.json.gz',
         dict(cohorts=by_dataset,cases=rows,source_images='original captured test split RGB',
              geometry='visible frontmost captured surfaces including decorative quads',
              chooser='maximum pooled strict geometry F1 over one complete export per image'))


if __name__=='__main__':
    summarize(ROOT/'runs/native-crop-ceiling')
