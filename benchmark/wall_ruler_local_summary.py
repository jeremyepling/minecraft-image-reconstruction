"""Decode controls and candidate exports and pool scores on identical image sets."""
import json
import shutil
from hashlib import sha256
from zipfile import ZipFile

from benchmark.wall_ruler_local_transfer import CONTROLS, ROOT, WORK, OLD, OUT, REVIEW, EXTENSION, read
from benchmark.local_wall_ruler_exports import entries
from benchmark.local_wall_ruler_summary import inventory
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.expanded_baseline import save
from benchmark.typed_structures import pooled_scores
import numpy as np

ARMS=['ordinary','whole','direct','connected']+(['overlap'] if EXTENSION else [])
DELTA=['recovered','lost','extra_added','extra_removed']


def main():
    items=entries()
    prior_path=ROOT/'benchmark/reviews/local-wall-rulers-01/exports.json'
    prior={r['id']:r for r in read(prior_path)['cases']}
    data=read(CONTROLS)
    frames=data['frames'] if EXTENSION else data['pilot']['frames']
    report_dir=REVIEW.parent/'wall-ruler-local-transfer-extension-01' if EXTENSION else REVIEW
    report_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for f in frames:
        key=f['id']
        directory=OUT/key
        base=f['baseline']
        p=WORK/f['pack']['path']
        if 'member' in f['pack']:
            with ZipFile(p) as z:
                raw=z.read(f['pack']['member'])
        else:
            raw=p.read_bytes()
        assert sha256(raw).hexdigest()==f['pack'].get('member_sha256',f['pack']['sha256'])
        if f['dataset']!='fixed-104':
            blocks=read(WORK/f['reference']['labels.json']['path'])['blocks']
            transform=np.array(base['registration']['transform'])
        else:
            labels,_,transform=fixed.fixed_reference(items[key])
            blocks=labels['blocks']
        truth={tuple(b['xyz_world']) for b in blocks}
        initial,local=inventory(raw,base,transform)
        records=read(directory/'regions.json')['records']
        row=dict(id=key,dataset=f['dataset'],split=f['split'],reference=f['reference'],
                 source_sha256=f['rgb']['sha256'],candidates_sha256=sha256((OLD/(key+'.json')).read_bytes()).hexdigest(),
                 ordinary=dict(geometry=base['geometry'],scores=base['scores'],pack_sha256=sha256(raw).hexdigest()),
                 regions=records)
        if not EXTENSION:
            row['whole']=prior[key]['arms']['native_single']
        row['camera_usable']=base.get('camera_usable')
        selected=['noop','direct','connected']
        if EXTENSION:
            selected.insert(1,'whole')
        if EXTENSION or (directory/'overlap'/'case.json').exists():
            selected.append('overlap')
        for arm in selected:
            source=arm
            if arm=='connected' and all(r[arm]==r['direct'] for r in records):
                source='direct'
            if source=='overlap' and not any(r['connected'] for r in records):
                source='noop'
            if source=='whole' and not records:
                source='noop'
            if source not in ('noop','whole','overlap') and not any(r[source] for r in records):
                source='noop'
            receipt=read(directory/source/'case.json')
            case=receipt['case']
            pack_source=receipt.get('reused',source)
            payload=(directory/pack_source/'reconstruction.zip').read_bytes()
            assert sha256(payload).hexdigest()==case['pack_sha256']
            placed,other=inventory(payload,case,transform)
            if arm=='noop':
                assert placed==initial and other==local
                assert all(base[k]==case[k] for k in ('geometry','scores','translation'))
            gains=placed.keys()-initial.keys()
            losses=initial.keys()-placed.keys()
            row[arm]=dict(source=pack_source,geometry=case['geometry'],scores=case['scores'],events=receipt['events'],
                          pack_sha256=case['pack_sha256'],recovered=len(gains & truth),lost=len(losses & truth),
                          extra_added=len(gains-truth),extra_removed=len(losses-truth),translation=case['translation'],
                          changed=placed!=initial)
        if (not EXTENSION and key.startswith(('train-viaduct-','test-castle-01','test-walled-garden-'))) or (EXTENSION and key.startswith(('validation-hillside-chapel','test-twinspire3'))):
            shutil.copy2(directory/'source.jpg',report_dir/(key+'.jpg'))
        rows.append(row)
    summary={}
    for cohort in ['fixed-104','visible-builds-v3','decorative-builds-v1','development','transfer','all']:
        batch=[r for r in rows if r['dataset']==cohort or cohort=='all' or
               (cohort in ('development','transfer') and (r['split']=='train')==(cohort=='development'))]
        if not batch:
            continue
        summary[cohort]={a:{k:pooled_scores([r[a][k] for r in batch]) for k in ('geometry','scores')} for a in ARMS}
        summary[cohort]['tested']=len(batch)
        for a in ARMS[1:]:
            summary[cohort][a].update({k:sum(r[a][k] for r in batch) for k in DELTA})
    save(report_dir/'exports.json',dict(cases=rows,summary=summary,noop_decoded_equal=len(rows),
         controls_sha256=sha256(CONTROLS.read_bytes()).hexdigest(),prior_exports_sha256=sha256(prior_path.read_bytes()).hexdigest(),method_sha256=sha256((ROOT/'benchmark/wall_ruler_local_transfer.py').read_bytes()).hexdigest()))
    for name,s in summary.items():
        print(name,*[a+' '+ '/'.join(f"{100*s[a][k][m]['f1']:.2f}" for k in ('geometry','scores') for m in ('strict','facade')) for a in ARMS],sep=' | ')
    for r in rows:
        print(r['id'], *['/'.join(str(r[a][k]) for k in DELTA) for a in ARMS[1:]],sep=' | ')


if __name__=='__main__':
    main()
