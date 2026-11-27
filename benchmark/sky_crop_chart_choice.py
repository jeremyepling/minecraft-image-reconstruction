"""Try an alternative counted chart per surface while preserving existing shared joins."""
import argparse
from pathlib import Path

from benchmark import relative_face_round as assembly
from benchmark.expanded_baseline import ROOT, read, save
from benchmark.relative_face_assembly import assemble
from benchmark.sky_crop_guard import BASE, pooled


def grade(charts):
    fitted=assemble(charts,min_edges=1,edge_pixels=2.5,coplanar_pixels=.5)
    if not fitted['groups']:
        return 0,0,set(),0,set()
    main=fitted['groups'][0]
    ids=set(main['origins'])
    joins={tuple(sorted(j['chart_ids'])) for j in fitted['joins'] if set(j['chart_ids'])<=ids}
    conflicts=sum(set(j['chart_ids'])<=ids for j in fitted['conflicts'])
    return len(main['assigned']),len(main['blocks'])-len(main['assigned']),joins,conflicts,set(main['blocks'])


def contains_build(old, new):
    """Find one integer translation preserving every previously placed position."""
    if not old:
        return True
    if len(new) < len(old):
        return False
    if old <= new:
        return True
    x,y,z=min(old)
    for nx,ny,nz in new:
        dx,dy,dz=nx-x,ny-y,nz-z
        if all((a+dx,b+dy,c+dz) in new for a,b,c in old):
            return True
    return False


def choose(base, candidate, preserve):
    live=list(base)
    ids={chart['id']:i for i,chart in enumerate(live)}
    current=grade(live)
    accepted=[]
    # Deterministic trial order. Keep the original if source evidence ties.
    for chart in sorted(candidate,key=lambda c:c['id']):
        index=ids.get(chart['id'],len(live))
        if index<len(live) and live[index]==chart:
            continue
        trial=live.copy()
        trial[index:index+1]=[chart]
        alt=grade(trial)
        if (alt[0]>current[0] and alt[1]<=current[1] and current[2]<=alt[2] and alt[3]<=current[3]
                and (not preserve or contains_build(current[4],alt[4]))):
            live=trial
            current=alt
            ids[chart['id']]=index
            accepted.append(chart['id'])
    return live,accepted


def main(names, preserve):
    for arm in names:
        root=ROOT/'runs/sky-crop-downstream'/arm
        mode='preserve-chart' if preserve else 'paired-chart'
        subdir='preserve' if preserve else 'paired'
        assembly.MODES[mode]=(root/subdir,'charts.json',False,1,2.5,.5,False)
        assembly.OUTPUT=root/'paired-export'
        results=[]
        for case in read(assembly.SHARED/'fixtures.json')['cases']:
            name=case['id']
            relative=Path('garden-anchor-extension/connected/sky')/name/'charts.json'
            base=read(BASE/relative)
            candidate=read(root/relative)
            if candidate['source_sha256']!=base['source_sha256']:
                raise ValueError(name)
            chosen,changes=choose(base['charts'],candidate['charts'],preserve)
            save(root/subdir/'sky'/name/'charts.json',dict(base,charts=chosen,accepted_alternatives=changes))
            assembly.evaluate((case,'sky',mode))
            score=read(assembly.OUTPUT/mode/'sky'/name/'score.json')['score']['fixed']
            results.append(dict(id=name,accepted=changes,score=score))
        metrics=pooled([r['score'] for r in results])
        save(root/(subdir+'-summary.json.gz'),dict(frames=results,pooled=metrics))
        print(arm,[(k,round(v['f1']*100,2)) for k,v in metrics.items()],
              [(r['id'],r['accepted']) for r in results if r['accepted']])


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('arms',nargs='+')
    cli.add_argument('--preserve',action='store_true')
    args=cli.parse_args()
    main(args.arms,args.preserve)
