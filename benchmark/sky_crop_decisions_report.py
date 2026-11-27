"""Archive native support selections and compact held-out/building/crop score receipts."""
from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.sky_crop_options import OUTPUT
from benchmark.sky_crop_chooser import DEV


ARCHIVE = ROOT/'benchmark/reviews/sky-crop-chooser-01'


def summarize():
    sc = read(OUTPUT/'decisions/report.json')
    options = read(OUTPUT/'scores-all/report.json')
    choices = {n:read(OUTPUT/f'{n}-choices.json')['frames'] for n in ('learned','oracle')}
    cases = []
    for case in sc['cases']:
        ident = case['id']
        arms = case['arms']
        crops = []
        for current, learned, oracle in zip(*(arms[n]['proposals'] for n in ('baseline','learned','oracle'))):
            pid = current['id']
            if not (pid == learned['id'] == oracle['id'] and
                    current['target'] == learned['target'] == oracle['target'] and
                    oracle['iou'] >= max(current['iou'],learned['iou'])):
                raise ValueError('Inconsistent fixed target, proposal or chooser ceiling')
            crops.append(dict(id=pid,label=current['label'],target=current['target'],
                              current=current['iou'],learned=learned['iou'],oracle=oracle['iou'],
                              learned_choice=choices['learned'][ident][pid],oracle_choice=choices['oracle'][ident][pid]))
        option_case = next(c for c in options['cases'] if c['id'] == ident)
        bests = [max(t['best_iou'] for t in group) for group in zip(*(a['targets'] for a in option_case['arms'].values()))]
        cases.append(dict(id=ident,development=ident in DEV,source_sha256=case['source_sha256'],
                          reference_sha256=case['reference_sha256'],source_handoff_sha256=case['source_handoff_sha256'],
                          reached_by_any_option=sum(v>0 for v in bests),
                          matched_by_any_option=sum(v>=.50 for v in bests),
                          summary={n:arm['summary'] for n,arm in arms.items()},crops=crops))
    save(ARCHIVE/'scores.json',dict(fixture_sha256=sc['fixture_sha256'],scorer_sha256=sc['scorer_sha256'],
                                    candidate_sha256=sc['candidate_sha256'],pooled=sc['pooled'],cases=cases,
                                    option_summary={n:a for n,a in options['pooled'].items()},
                                    development=list(DEV)))
    save(ARCHIVE/'frozen.json',read(OUTPUT/'frozen.json'))
    save(ARCHIVE/'sources.json',read(OUTPUT/'sources.json'))
    for n in ('learned','oracle'):
        f = OUTPUT/f'{n}.json'
        if digest(f) != sc['candidate_sha256'][n]:
            raise ValueError(f'Changed support packet {f}')
        save(ARCHIVE/f'{n}.json.gz',read(f))


if __name__ == '__main__':
    summarize()
