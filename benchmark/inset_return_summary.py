"""Condense the independent fixed scorer outputs for strict photographed inset masks."""
from argparse import ArgumentParser
from pathlib import Path

from benchmark import framed_depth_eval as f
from benchmark.texture_camera_eval import checked,digest,read,save


def main(source,out):
    out.mkdir(parents=True,exist_ok=True)
    rows=read(checked(source/'cases.json',read(source/'summary.json')['cases_sha256']))
    groups={'development':['train'],'fixed':f.EVALUATION,'alternate':['extra'],'new':['new'],'shelter':['shelter']}
    result={}
    comp=[]
    for name,splits in groups.items():
        cases=[r for r in rows if r['split'] in splits]
        changed=[r for r in cases if r['changed']]
        result[name]=dict(views=len(cases),changed=len(changed),events=sum(len(r['events']) for r in changed),
            matched_gained=sum(len(set(r['after']['matched'])-set(r['before']['matched'])) for r in changed),
            matched_lost=sum(len(set(r['before']['matched'])-set(r['after']['matched'])) for r in changed),
            paired_px=sum(r['after']['paired_pixels']-r['before']['paired_pixels'] for r in changed),
            unmatched_delta=sum(r['after']['unmatched_predictions']-r['before']['unmatched_predictions'] for r in changed),
            background_delta=sum(r['after']['background_pixels']-r['before']['background_pixels'] for r in changed),
            **{k:sum(r['geometry'][k] for r in changed) for k in ['gained','lost','extra_added','extra_removed','hidden_added','hidden_removed']})
        for r in changed:
            if not r['geometry']['baseline_reproduced']:
                raise ValueError('Production replay differs: '+r['id'])
            comp.append(dict(split=r['split'],id=r['id'],events=r['events'],rgb_sha256=r['rgb_sha256'],
                gained_surfaces=sorted(set(r['after']['matched'])-set(r['before']['matched'])),
                lost_surfaces=sorted(set(r['before']['matched'])-set(r['after']['matched'])),
                geometry={k:r['geometry'][k] for k in ('correct','missed','extras','gained','lost','extra_added','extra_removed','hidden_extras')},
                new_back_counts={e['output']:r['geometry']['new_dims'][e['output']] for e in r['events']}))
    focus=next(r for r in rows if r['id']=='heldout-facade-oblique-right')
    save(out/'result.json',dict(candidate='15202d5',control='c589f1a',frames=len(rows),source_cases_sha256=digest(source/'cases.json'),
         groups=result,changed=comp,focus={s:{k:focus[k]['best'][s] for k in ('before','after')} for s in ['9','14','19']}))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('../facade-inset-round/runs/returns'))
    p.add_argument('--out',type=Path,default=Path('benchmark/reviews/facade-inset-returns-01'))
    args=p.parse_args()
    main(args.source,args.out)
