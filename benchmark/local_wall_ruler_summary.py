"""Pool actual decoded exports with no-change frames and verify stored carrier signatures."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from benchmark import expanded_fixed_handoffs as fixed
from benchmark.mask_blocks import decoded_pack
from benchmark.local_wall_rulers import ARMS,CONTROLS,ROOT,OUT,WORK,read
from benchmark.local_wall_ruler_exports import entries,options
from benchmark.local_wall_ruler_audit import REVIEW
from benchmark.typed_structures import aligned_sites,pooled_scores
from benchmark.full_block_lattice import atlas


def inventory(raw,case,transform):
    local=decoded_pack(BytesIO(raw))
    coords=aligned_sites(list(local),transform,np.array(case['translation']))
    return dict(zip(coords,local.values())),local


def main():
    releases=entries()
    arms=[*ARMS,'native_mixed']
    cases=[]
    for f in read(CONTROLS)['pilot']['frames']:
        key=f['id']
        p=WORK/f['pack']['path']
        if 'member' in f['pack']:
            with ZipFile(p) as z: raw=z.read(f['pack']['member'])
        else:
            raw=p.read_bytes()
        assert sha256(raw).hexdigest()==f['pack'].get('member_sha256',f['pack']['sha256'])
        base=f['baseline']
        if 'labels.json' in f['reference']:
            blocks=read(WORK/f['reference']['labels.json']['path'])['blocks']
            transform=np.array(base['registration']['transform'])
        else:
            labels,_,transform=fixed.fixed_reference(releases[key])
            blocks=labels['blocks']
        prior,local=inventory(raw,base,transform)
        truth={tuple(b['xyz_world']):b['block_id'] for b in blocks}
        row=dict(id=key,split=f['split'],dataset=f['dataset'],rgb_sha256=f['rgb']['sha256'],arms={},reference=f['reference'],
                 baseline={k:base[k] for k in ('geometry','scores','exported')},pack_sha256=sha256(raw).hexdigest())
        signatures=options(f)
        locs={}
        for folder in [OUT,ROOT/'runs/local-wall-rulers-01']:
            for path in (folder/'exports'/key).glob('*/case.json'):
                report=read(path)
                sig=tuple(tuple(x) for x in report['signature'])
                locs.setdefault(sig,(path,report))
        if any(signatures):
            path,r=locs[()]
            mapped,other=inventory((path.parent/'reconstruction.zip').read_bytes(),r['case'],transform)
            assert other==local and mapped==prior
            assert all(base[k]==r['case'][k] for k in ('geometry','scores','translation'))
            row['noop_decoded_equal']=True
        for sig,names in signatures.items():
            new=prior
            case=base
            packhash=row['pack_sha256']
            ev=[]
            if sig:
                # A receipt can be reused across native support repair only when its
                # entire ordered (part,axis,level) carrier signature still matches.
                path,receipt=locs[sig]
                assert receipt['case']['source_sha256']==f['rgb']['sha256']
                case=receipt['case']
                pp=path.parent/'reconstruction.zip'
                payload=pp.read_bytes()
                assert sha256(payload).hexdigest()==case['pack_sha256']
                new,_=inventory(payload,case,transform)
                packhash=case['pack_sha256']
                ev=receipt['events']
            gained=new.keys()-prior.keys()
            lost=prior.keys()-new.keys()
            record=dict(geometry=case['geometry'],scores=case['scores'],pack_sha256=packhash,
                        recovered=len(gained & truth.keys()),lost=len(lost & truth.keys()),
                        extra_added=len(gained-truth.keys()),extra_removed=len(lost-truth.keys()),
                        changed_ids_or_positions=new!=prior,changed_positions=new.keys()!=prior.keys(),
                        signature=sig,carrier_changes=ev,exported=case['exported'])
            row['arms'].update({n:record for n in names})
        cases.append(row)
    summary={}
    for key in ('all','fixed-104','visible-builds-v3','decorative-builds-v1','development','transfer'):
        group=[r for r in cases if r['dataset']==key or key=='all' or
               (key in ('development','transfer') and (r['split']=='train')==(key=='development'))]
        item={}
        for arm in ['baseline',*arms]:
            vals=[r['baseline'] if arm=='baseline' else r['arms'][arm] for r in group]
            stats={k:pooled_scores([v[k] for v in vals]) for k in ('geometry','scores')}
            if arm!='baseline':
                stats.update({k:sum(v[k] for v in vals) for k in ('recovered','lost','extra_added','extra_removed','changed_ids_or_positions','changed_positions')})
                stats['offered_frames']=sum(bool(v['signature']) for v in vals)
            item[arm]=stats
        # Reference-only ceiling over actual outputs and incumbent for this same frame.
        for metric in ('strict','facade'):
            best=[max([r['baseline'],*r['arms'].values()],key=lambda v:v['geometry'][metric]['f1']) for r in group]
            item['oracle_'+metric]={k:pooled_scores([v[k] for v in best]) for k in ('geometry','scores')}
        summary[key]=dict(tested=len(group),**item)
    (REVIEW/'exports.json').write_text(json.dumps(dict(cases=cases,summary=summary),separators=(',',':')))
    manifest=dict(base_revision='e3bd220',branch='experiment/local-atlas-wall-rulers',minecraft='26.3',model='deterministic vanilla atlas, ordinary learned surface fields',
        atlas=atlas().provenance,controls_sha256=sha256(CONTROLS.read_bytes()).hexdigest(),
        provenance={f['id']:dict(model_sha256=f['replay'].get('model_sha256'),source_sha256=f['rgb']['sha256'],
            pack_sha256=f['pack'].get('member_sha256',f['pack']['sha256']),visible_sha256=f['visible_ids_sha256']) for f in read(CONTROLS)['pilot']['frames']},
        implementation={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'benchmark').glob('local_wall_ruler*.py'))},
        artifacts={p.name:sha256(p.read_bytes()).hexdigest() for p in REVIEW.iterdir() if p.suffix in ('.jpg','.json') and p.name!='manifest.json'})
    (REVIEW/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for group,item in summary.items():
        print(group,[(k,round(item[k]['geometry']['strict']['f1']*100,2),round(item[k]['geometry']['facade']['f1']*100,2),
                    round(item[k]['scores']['strict']['f1']*100,2),round(item[k]['scores']['facade']['f1']*100,2))
                    for k in ['baseline',*arms]])


if __name__=='__main__':
    main()
