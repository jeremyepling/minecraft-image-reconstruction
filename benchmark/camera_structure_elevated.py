"""Second camera rule. Validate phase on multiple photographed elevated corners above a wall foot."""
import itertools
from collections import defaultdict
from benchmark.camera_structure_run import BASE,COHORTS,REPORT,ROOT,RUN,SOURCE,frames,paths,read,save,summarize
import numpy as np
import sys


def options(source):
    scaffold=source.get('scaffold')
    reports=[]
    selected=[]
    for option in source.get('tentative',[]):
        contact=option.get('contact')
        if not contact:
            continue
        edges=scaffold['edges']
        families=defaultdict(set)
        for a,b,dim in edges:
            families[a].add(dim)
            families[b].add(dim)
        a,b=contact['edge']
        if len(families[a])<2:
            continue
        xyz=np.asarray(scaffold['xyz'])
        P=np.asarray(option['P'])
        eye=-np.linalg.solve(P[:,:3],P[:,3])
        # The measured floor junction is at (0,0,0) in the tentative camera.
        scale=-eye[1]/xyz[a,1]
        if scale<=0:
            continue
        metric=xyz*scale+eye
        junctions=[k for k in families if len(families[k])>=2 and metric[k,1]>1.5]
        good=[k for k in junctions if np.max(abs(metric[k]-np.rint(metric[k])))<=.20]
        separated=any(np.linalg.norm(metric[p]-metric[q])>=2 for p,q in itertools.combinations(good,2))
        fraction=len(good)/max(1,len(junctions))
        eligible=fraction>.5 and separated
        reports.append(dict(elevated=len(junctions),agree=len(good),fraction=fraction,
                            separated=separated,accepted=eligible))
        if eligible:
            selected.append(option['P'])
    return selected,reports


def main():
    controls=read(BASE/'scores/cases.json')
    entries=list(frames(SOURCE,COHORTS))
    mapping={(e['dataset'],e['id']):e for e in entries}
    cases=[]
    changed=[]
    sys.path.append(str(ROOT.parent/'camera-horizon-floor/benchmark'))
    from camera_horizon_replay import replay
    for base in controls:
        result=base
        metrics=[]
        if not base['camera_usable']:
            e=mapping[base['dataset'],base['id']]
            folder,out=paths(e)
            source=read(out/'source.json.gz')
            choices,metrics=options(source)
            if choices:
                P=choices[0]
                forced=out/'forced-grade.json'
                if forced.exists() and P==source['tentative'][0]['P']:
                    result=read(forced)
                else:
                    camera=dict(P=P,reason='counted wall base and multiple elevated turning corners',
                                scale_status='supported',camera_source='structure-elevated-corners')
                    result=replay(e,camera,folder,out/'elevated-build')
                changed.append(e['id'])
        cases.append(dict(dataset=base['dataset'],id=base['id'],split=base['split'],
                          source=result,changed=result is not base,metrics=metrics))
    save(REPORT/'elevated-corners.json.gz',cases)
    save(REPORT/'elevated-summary.json',dict(changed=changed,summary=summarize([r['source'] for r in cases],entries),
        cases=[dict(id=r['id'],metrics=r['metrics']) for r in cases if r['metrics']]))


if __name__=='__main__':
    main()
