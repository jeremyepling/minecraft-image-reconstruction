"""Publish the frozen candidate set and its attainable decoded-pack scores."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
from benchmark.floating_corner_overwrite import FREEZE, OUT, REVIEW, T1ROOT, WORKTREES, digest, jobs, read, save


def counts(rows, choices, field, kind):
    result = Counter()
    for row, arm in zip(rows, choices, strict=True):
        metric = row['score'][arm][field][kind]
        result.update({k: metric[k] for k in ('correct','missed','extras')})
    tp = result['correct']
    return dict(result,f1=2*tp/(2*tp+result['missed']+result['extras']))


def offered(row):
    return [arm for arm in row['score'] if arm!='noop']


def choose(rows, field, kind):
    # Maximize the corpus micro F1 over exactly one exported option per view.
    # Integer cross-products preserve exact ties and prefer keeping the original.
    rate = Fraction(0)
    while True:
        def objective(row, arm):
            m = row['score'][arm][field][kind]
            return 2*m['correct']*rate.denominator - rate.numerator*(2*m['correct']+m['missed']+m['extras'])
        choices = [max(offered(r), key=lambda a:objective(r,a)) for r in rows]
        total = counts(rows,choices,field,kind)
        new = Fraction(2*total['correct'],2*total['correct']+total['missed']+total['extras'])
        if new==rate:
            return choices
        rate = new


def compare(rows):
    choices = dict(keep=['keep']*len(rows),restore_all=[next((a for a in reversed(offered(r)) if a.startswith('restore-')),'keep') for r in rows])
    for field in ('geometry','scores'):
        for kind in ('strict','facade'):
            choices['chooser_'+field+'_'+kind]=choose(rows,field,kind)
    result = {}
    for key,arms in choices.items():
        result[key]=dict(
            metric={field:{kind:counts(rows,arms,field,kind) for kind in ('strict','facade')} for field in ('geometry','scores')},
            paired={k:sum(r['paired'][arm][k] for r,arm in zip(rows,arms)) for k in ('gained','lost','added','removed')},
            substitutions=[dict(panel=r['panel'],id=r['id'],arm=a) for r,a in zip(rows,arms) if a!='keep'],
            assigned=sum(r['score'][a]['scores']['assigned'] for r,a in zip(rows,arms)),
            fallback=sum(r['score'][a]['scores']['fallback'] for r,a in zip(rows,arms)))
    return result


def publish():
    assert digest(FREEZE)=='b71078163ce9f1ddfbc557fc828eff93442201d3cd24d0cc8dcee8775ff6f6bb'
    rows=[]
    statuses=Counter()
    singles=Counter()
    for frozen,special in jobs():
        name,panel=frozen['id'],frozen['panel']
        row=read(OUT/('exact-t1' if special else panel)/'rows'/(name+'.json'))
        assert row['source_sha256']==frozen['source_sha256']
        row['matches_depth_refit_pack_hash']=row['baseline_pack_sha256']==frozen['pack_sha256']['source-depth']
        # Machine-readable paths are relative to the worktree farm.
        for key in ('packet','files'):
            row[key]={k:Path(v).relative_to(WORKTREES).as_posix() for k,v in row[key].items()}
        row['base']=Path(row['base']).relative_to(WORKTREES).as_posix()
        if special:
            previous=read(T1ROOT/'t1-grid-diagnosis/bypass.json')
            for old,new in [('keep-shared','keep'),('bypass-shared','restore-mask-1')]:
                for k,s in [('geometry','geometry'),('scores','ids')]:
                    assert row['score'][new][k]==previous[old][s]
            save(REVIEW/'exact-t1.json.gz',row)
            continue
        statuses.update(c['status'] for c in row['charts'].values())
        for a in offered(row):
            if a.startswith('restore-mask'):
                for field in ('geometry','scores'):
                    x=row['score'][a][field]['strict']['f1']
                    y=row['score']['keep'][field]['strict']['f1']
                    singles[(field,'improve' if x>y else 'regress' if x<y else 'tie')]+=1
        rows.append(row)
    ignored=[dict(panel=r['panel'],id=r['id']) for r in read(FREEZE) if not r['active']]
    summary=dict(revision='5bd7908',fixture_sha256=digest(FREEZE),frozen_active=len(rows),frozen_inactive=ignored,
                 accepted_seams=sum(len(r['accepted_seams']) for r in rows),chart_status=statuses,frames_with_accepted_seams=sum(bool(r['charts']) for r in rows),
                 offered_frames=sum(len(offered(r))>1 for r in rows),unavailable_frames=0,
                 single_restore={f:{k:singles[f,k] for k in ('improve','tie','regress')} for f in ('geometry','scores')},
                 source_depth_archive_hash_matches=sum(r['matches_depth_refit_pack_hash'] for r in rows),
                 all=compare(rows),by_dataset={d:compare([r for r in rows if r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})})
    save(REVIEW/'conditions.json.gz',rows)
    save(REVIEW/'summary.json',summary)


if __name__=='__main__':
    publish()
