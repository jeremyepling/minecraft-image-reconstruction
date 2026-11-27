"""Compose verified incumbent and empirical exports; references enter only in scoring."""
from collections import Counter, defaultdict

from PIL import Image, ImageDraw
from benchmark.camera_grid_joint import CONTROL,GEO,MOGE,RUN,REVIEW,Path,digest,read,save,entries
from benchmark.camera_grid_joint_replay import tasks
from benchmark.expanded_scores import summarize


def export(case_path,pack):
    case=read(case_path)
    if 'case' in case: case=case['case']
    assert case['pack_sha256']==digest(pack)
    return case


def main():
    frozen=read(CONTROL/'frames.json')
    for name,sha in read(CONTROL/'manifest.json')['hashes'].items():
        assert digest(CONTROL/name)==sha
    canonical=read(CONTROL/'baseline-cases.json.gz')['repeat']
    candidates={e['id']:e for e in entries()}
    review_rows=[]
    summary=defaultdict(list)
    arms_for=defaultdict(list)
    for e,arm in tasks(): arms_for[e['id']].append(arm)
    for base in canonical:
        key,ds=base['id'],base['dataset']
        matched=GEO/'matched-control'
        incumbent=base
        if (matched/'scores'/(key+'.json')).exists() and not base['camera_usable']:
            incumbent=export(matched/'scores'/(key+'.json'),matched/'builds'/ds/key/'reconstruction.zip')
        source=incumbent
        choices={'keep':incumbent}
        row=dict(id=key,dataset=ds,split=base['split'],incumbent_pack_sha256=incumbent['pack_sha256'])
        if key in candidates:
            folder=RUN/key
            p=read(folder/'choices.json')
            assert p['source_sha256']==incumbent['source_sha256']
            row['prediction']=p
            if (GEO/'scores'/(key+'.json')).exists():
                choices['raw-geocalib']=export(GEO/'scores'/(key+'.json'),GEO/'builds'/ds/key/'reconstruction.zip')
            if (MOGE/ds/key/'case.json').exists():
                choices['raw-moge']=export(MOGE/ds/key/'case.json',MOGE/ds/key/'build/reconstruction.zip')
        best_raw=max(choices,key=lambda k:choices[k]['geometry']['strict']['f1'])
        for arm in arms_for[key]:
            c=export(folder/(arm+'.json'),folder/arm/'reconstruction.zip')
            row.setdefault('ablations',{})[arm]=dict(geometry=c['geometry'],scores=c['scores'],pack_sha256=c['pack_sha256'],
                registered=c['registration']['geometry']['strict'],native=c['camera_validation'],calls=read(folder/(arm+'.json'))['calls'])
            manifest = folder/arm/'build-manifest.json'
            actual = read(manifest)['placement']
            measured = {r['id']:[r['axis'],r['level']] for r in actual['plane_offsets']}
            applied = row['ablations'][arm]['calls'][-1]['applied']
            assert all(measured[k][1]==v for k,v in applied.items())
            row['ablations'][arm].update(manifest_sha256=digest(manifest),actual_plane_offsets={k:measured[k] for k in applied})
            if arm.startswith('joint'): choices[arm]=c
            if p['accepted'] and arm=='joint1': source=c
        assert all(c['source_sha256']==incumbent['source_sha256'] for c in choices.values())
        best=max(choices,key=lambda k:choices[k]['geometry']['strict']['f1'])
        row.update(oracle=best,raw_oracle=best_raw,source_pack_sha256=source['pack_sha256'],
                   options={k:{f:c[f] for f in ('geometry','scores','pack_sha256')} for k,c in choices.items()})
        for name,c in [('incumbent',incumbent),('source',source),('raw-oracle',choices[best_raw]),('joint-oracle',choices[best])]:
            summary[name].append(c)
        review_rows.append(row)
    summaries={name:summarize(cases,frozen) for name,cases in summary.items()}
    save(REVIEW/'cases.json.gz',review_rows)
    save(REVIEW/'summary.json',summaries)
    for key,arms in arms_for.items():
        if not arms: continue
        e=candidates[key]
        p=read(RUN/key/'choices.json')['best'][0]
        im=Image.open(Path(e['frame'])/'rgb.png').convert('RGB')
        draw=ImageDraw.Draw(im)
        for qs,projected,fit in zip(p['native_quads'],p['reprojected'],p['fit']):
            for lines,color in [(qs,'cyan' if fit else 'lime'),(projected,'magenta')]:
                xy=[tuple(q) for q in lines]
                draw.line(xy+xy[:1],fill=color,width=2)
        draw.rectangle((0,0,im.width,32),fill='black')
        draw.text((6,4),key+'   cyan: fit sprite   green: held wall sprite   magenta: fitted lattice',size=18)
        im.save(REVIEW/(key+'.jpg'),quality=84)
    save(REVIEW/'manifest.json',dict(base='d9eb377',source_revision='experiment/camera-grid-joint',minecraft='26.3',
        frozen_frames_sha256=digest(CONTROL/'frames.json'),canonical_cases_sha256=digest(CONTROL/'baseline-cases.json.gz'),
        frame_count=len(frozen),residual_count=len(candidates),split_counts=dict(Counter(e['split'] for e in candidates.values())),
        raw_candidates='verified GeoCalib and MoGe exports on residuals',refined_exports=sum(arm.startswith('joint') for arms in arms_for.values() for arm in arms),
        ablation_exports=sum(arm in ('grid','camera') for arms in arms_for.values() for arm in arms),
        diagnostic_frames=[k for k,arms in arms_for.items() if arms],
        hashes={p.name:digest(p) for p in [REVIEW/'cases.json.gz',REVIEW/'summary.json',*sorted(REVIEW.glob('*.jpg'))]}))
    print('decision counts',Counter(r['oracle'] for r in review_rows))
    for name,cohorts in summaries.items():
        for dataset,s in cohorts.items():
            print(name,dataset,*[round(s[type_][mode]['f1']*100,2) for type_ in ('geometry','scores') for mode in ('strict','facade')],
                  s['camera']['unusable'],s['geometry']['strict'])


if __name__=='__main__':
    main()
