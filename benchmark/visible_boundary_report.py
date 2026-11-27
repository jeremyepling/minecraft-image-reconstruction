"""Pool identical released controls and newly decoded alternative packs."""
from collections import Counter
from pathlib import Path
import shutil
import hashlib
from io import BytesIO
from zipfile import ZipFile
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.typed_structures import pooled_scores
from benchmark.expanded_baseline import digest,frames,SOURCE,COHORTS,MEASURED_REVISION
from benchmark.visible_boundary_capture import RUN,BASE,CASES,read,save
from benchmark.visible_boundary_screen import inputs
from benchmark.visible_boundary_evaluate import audit
from benchmark import expanded_scores as grades
import numpy as np

REVIEW=fixed.ROOT/'benchmark/reviews/visible-boundary-direction-01'
DELTA=('gained','lost','extras_added','extras_removed')
ARMS=('baseline','corner_boundary','corner_direction','corner_both','parallel_boundary','source','strict_source','oracle','pooled_oracle')


def summarize(rows):
    def metrics(name):
        selected=[r['options'][r['choices'][name]] for r in rows]
        change=Counter()
        for r in rows:
            change.update(r['deltas'].get(r['choices'][name],{}))
        return dict(geometry=pooled_scores([r['geometry'] for r in selected]),
                    scores=pooled_scores([r['scores'] for r in selected]),
                    delta={key:change[key] for key in DELTA},
                    changed_packs=sum(s.get('decoded_sha256')!=r['options']['baseline'].get('decoded_sha256') for r,s in zip(rows,selected)))
    return dict(tested=len(rows),camera_supported=sum(r['camera_usable'] for r in rows),
                nominees=sum(bool(r['proposals']) for r in rows),**{name:metrics(name) for name in ARMS})


def decoded_digest(payload):
    names=fixed.decoded_pack(BytesIO(payload))
    text=''.join(f'{x},{y},{z},{names[(x,y,z)]}\n' for x,y,z in sorted(names))
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    REVIEW.mkdir(parents=True,exist_ok=True)
    indexed={(e['dataset'],e['id']):e for e in list(fixed.fixed_frames())+list(frames(SOURCE,COHORTS))}
    rows=[]
    for base in read(CASES):
        dataset,key=base['dataset'],base['id']
        e=indexed[dataset,key]
        assert e['rgb_sha256']==base['source_sha256']
        folder=RUN/dataset/key
        options={'baseline':base}
        deltas,proposals,stages={}, {}, {}
        for family,proposal,result in [('corner',folder/'proposal.json',folder),('parallel',folder/'parallel-proposal.json',folder/'parallel')]:
            if not proposal.exists():
                assert not base['camera_usable']
                continue
            if not read(proposal):
                continue
            data=read(result/'result.json')
            assert data['arms']['baseline']['geometry']==base['geometry']
            assert data['arms']['baseline']['scores']==base['scores']
            proposals[family]=read(proposal)
            im,parts,cam=inputs(e)
            if dataset==fixed.FIXED_DATASET:
                labels,ids,transform=fixed.fixed_reference(e)
            else:
                labels=read(Path(e['frame'])/'labels.json')
                with np.load(Path(e['frame'])/'labels.npz') as arrays:
                    ids=arrays['face_id']
                transform=grades.registration(np.asarray(cam['P']),read(Path(e['frame'])/'camera.json'))
            stage=audit(im,parts,cam,proposals[family],labels,ids,transform)
            stages[family]=stage
            if dataset!=fixed.FIXED_DATASET:
                host=proposals[family]['id']
                stage['public_counts']={}
                stage['camera_projection_same']={}
                for arm in data['arms']:
                    loc=BASE/dataset/key if arm=='baseline' else result/arm
                    counts=read(loc/'counts.json')['surfaces']
                    stage['public_counts'][arm]={
                        c['id']:c.get('counts') for c in counts if c['id']==host or c['id'].startswith(host+'-crease-')}
                    stage['camera_projection_same'][arm]=bool(np.array_equal(cam['P'],read(loc/'build-manifest.json')['placement']['P']))
            options.update({family+'_'+arm:row for arm,row in data['arms'].items() if arm!='baseline'})
            deltas.update({family+'_'+arm:delta for arm,delta in data['deltas'].items()})
            preview=REVIEW/'previews'/(key+'-'+family+'.jpg')
            preview.parent.mkdir(exist_ok=True)
            shutil.copyfile(result/'preview.jpg',preview)
        if proposals:
            if dataset==fixed.FIXED_DATASET:
                with ZipFile(fixed._release()[2]) as archive:
                    payload=archive.read(key+'/reconstruction.zip')
            else:
                payload=(BASE/dataset/key/'reconstruction.zip').read_bytes()
            base['decoded_sha256']=decoded_digest(payload)
            if dataset!=fixed.FIXED_DATASET:
                match=read(folder/'control/match.json')
                assert match['projection_match'] and match['decoded_sha256']==base['decoded_sha256']
                for stage in stages.values():
                    stage['unchanged_input_control']=match
            for option,row in options.items():
                if option=='baseline':
                    continue
                family,arm=option.split('_')
                location=(folder/'parallel') if family=='parallel' else folder
                pack=(location/(arm+'.zip')) if dataset==fixed.FIXED_DATASET else (location/arm/'reconstruction.zip')
                row['decoded_sha256']=decoded_digest(pack.read_bytes())
        choices={name:name if name in options else 'baseline' for name in ARMS}
        for name,strict in [('source',False),('strict_source',True)]:
            for family,option in [('corner','corner_both'),('parallel','parallel_boundary')]:
                if family in proposals and (not strict or proposals[family]['source']):
                    choices[name]=option
                    break
        choices['oracle']=max(options,key=lambda a:options[a]['geometry']['facade']['f1'] or 0)
        if dataset==fixed.FIXED_DATASET:
            metadata=fixed._jobs()[key][1]['files']
            ref={k:v for k,v in metadata.items() if k in ('camera.json','labels.json','labels.npz','world-readback.json')}
        else:
            ref=base['reference_sha256']
        rows.append(dict(id=key,dataset=dataset,split=base['split'],layout=base['layout'],camera_usable=base['camera_usable'],
            rgb_sha256=base['source_sha256'],reference_sha256=ref,dataset_manifest_sha256=e['dataset_manifest_sha256'],
            choices=choices,deltas=deltas,proposals=proposals,stages=stages,
            options={a:{k:v[k] for k in ('geometry','scores','pack_sha256','decoded_sha256','translation','exported') if k in v} for a,v in options.items()}))
    # Maximizing each image's F1 and maximizing the pooled F1 use different
    # weights. Also compute the exact fractionally optimized cohort ceiling.
    for dataset in (fixed.FIXED_DATASET,*COHORTS):
        group=[r for r in rows if r['dataset']==dataset]
        ratio=0.
        for _ in range(100):
            numerator,denominator=0,0
            for r in group:
                def terms(arm):
                    m=r['options'][arm]['geometry']['facade']
                    return 2*m['correct'],2*m['correct']+m['missed']+m['extras']
                selected=max(r['options'],key=lambda a:terms(a)[0]-ratio*terms(a)[1])
                r['choices']['pooled_oracle']=selected
                n,d=terms(selected)
                numerator+=n
                denominator+=d
            updated=numerator/denominator
            if updated==ratio:
                break
            ratio=updated
        else:
            raise RuntimeError('Pooled chooser did not converge')
    save(REVIEW/'cases.json.gz',rows)
    summary={}
    for dataset in [fixed.FIXED_DATASET,*COHORTS]:
        group=[r for r in rows if r['dataset']==dataset]
        out=summarize(group)
        out['splits']={s:summarize([r for r in group if r['split']==s]) for s in sorted({r['split'] for r in group})}
        out['camera_supported_metrics']=summarize([r for r in group if r['camera_usable']])
        if dataset!=fixed.FIXED_DATASET:
            out['held_out']=summarize([r for r in group if r['split']!='train'])
        summary[dataset]=out
    summary['production_revision']=MEASURED_REVISION
    summary['control_cases_sha256']=digest(CASES)
    summary['cases_sha256']=digest(REVIEW/'cases.json.gz')
    save(REVIEW/'summary.json',summary)


if __name__=='__main__':
    main()
