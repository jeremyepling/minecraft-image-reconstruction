"""Choose the minimum evidence for a single wall cut on the two development layouts."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.mask_metrics import footprint
from benchmark.mobile_region_probe import score
from benchmark.normal_fields import cached_fields
from benchmark.normal_round import groups
from benchmark.surface_reference import recorded
from engine import single_wall_gate as gate
from runtime.mode import image_only


BASE=ROOT/'benchmark/reviews/mobile-combined-01'
PROBE=ROOT/'benchmark/reviews/mobile-regions-01'


def verified(path,expected):
    if digest(path)!=expected:
        raise ValueError(f'Frozen input changed: {path}')
    return path


def packed_input(row,case,probe,field_dir):
    sha=row['sha256']
    with Image.open(verified(case.path('image'),sha)) as source_image:
        image=source_image.convert('RGB')
    with gzip.open(verified(BASE/row['predictions'],row['prediction_sha256']),'rt') as f:
        regions=json.load(f)
    def asset(suffix):
        rel=f'proposals/{sha}{suffix}'
        return verified(PROBE/rel,probe['artifacts'][rel])
    with gzip.open(asset('.json.gz'),'rt') as f:
        meta=json.load(f)
    with np.load(asset('.npz'),allow_pickle=False) as f:
        packed=f['packed']
        width=int(f['shape'][2])
    masks={i:np.unpackbits(packed[i],count=width,axis=1).astype(bool) for i in meta['selected']}
    field=cached_fields(sha,field_dir,('depth',))
    if field is None:
        raise ValueError(f'Missing compatible Depth Anything V2 field for {sha}')
    return image,regions,masks,meta,gate.depth_grid(field['depth'],image.size)


@image_only()
def run(out,field_dir):
    cohorts,_,fixtures=groups()
    dev={f'{c.floor}/{c.id}':c for c in cohorts['development']}
    manifest=read(BASE/'development/manifest.json')
    rows=read(verified(BASE/'development/cases.json',manifest['artifacts']['cases.json']))
    if {r['key'] for r in rows}!=dev.keys():
        raise ValueError('Development frames differ')
    probe=read(PROBE/'manifest.json')
    trials=[(jump,agree) for jump in (.001,.004,.010) for agree in (.55,.7,.85)]
    report={str(t):Counter() for t in trials}
    details=[]
    for row in rows:
        case=dev[row['key']]
        image,parents,masks,meta,field=packed_input(row,case,probe,field_dir)
        inventory=gate.prepare(image,parents,masks,meta['candidates'],meta['selected'])
        events=gate.votes(inventory,field)
        # Store and score every development candidate as a single replacement for diagnosis.
        parent_by_id={p['id']:p for p in parents}
        pieces={(e['parent'],e['proposal']):gate.partition(image,parent_by_id[e['parent']],e['proposal'],masks[e['proposal']])
                for e in events}
        _,ref=recorded(case)
        by_id={}
        def ids(parts):
            for p in parts:
                if p['id'] not in by_id:
                    by_id[p['id']]=footprint(p,ref['id_map'].shape,image.size,True)
            owned=[by_id[p['id']] for p in parts]
            return {int(i) for i in score(owned,range(len(owned)),ref)}
        before=ids(parents)
        if before!=set(row['assignments'].values()):
            raise ValueError(f"Development control mismatch: {row['key']}")
        for e in events:
            child=pieces[(e['parent'],e['proposal'])]
            after=ids([part for p in parents for part in (child if p['id']==e['parent'] else [p])])
            e.update(gained=sorted(after-before),lost=sorted(before-after),extra=len(child)-1)
        details.append(dict(frame=row['key'],events=events))
        for jump,agree in trials:
            selected={}
            for e in events:
                if abs(e['median_step'])<jump or e['sign_agreement']<agree:
                    continue
                rank=(abs(e['median_step'])*e['sign_agreement'],meta['candidates'][e['proposal']]['quality'],-e['proposal'])
                if e['parent'] not in selected or rank>selected[e['parent']][0]:
                    selected[e['parent']]=(rank,e['proposal'])
            proposed=[part for p in parents for part in (pieces[p['id'],selected[p['id']][1]] if p['id'] in selected else [p])]
            after=ids(proposed)
            report[str((jump,agree))].update(found=len(after),gained=len(after-before),lost=len(before-after),
                proposed=len(proposed),extra_proposals=len(proposed)-len(parents),cuts=len(selected))
    out.mkdir(parents=True,exist_ok=False)
    save(out/'scores.json',report)
    save(out/'diagnostics.json',details)
    save(out/'manifest.json',dict(frames=[dict(key=r['key'],sha256=r['sha256']) for r in rows],
         fixtures=fixtures,settings=gate.SETTINGS,source_manifests={'mobile-combined-development':
         digest(BASE/'development/manifest.json'),'mobile-region-probe':digest(PROBE/'manifest.json')},
         baseline_found=sum(len(row['assignments']) for row in rows),
         runner_sha256=digest(Path(__file__)),gate_sha256=digest(Path(gate.__file__)),
         artifacts={name:digest(out/name) for name in ('scores.json','diagnostics.json')},
         rule='Maximize development matches, break ties by fewer extra proposals, stricter sign agreement, then stronger step.',
         distance_trials='Inside/outside distances 0/2, 1/2 and 1/3 lattice cells; tightening or relaxing containment and minimum sample count added no development matches.'))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fields',type=Path,default=ROOT.parent/'normal-round/runs/normal-fields')
    args=parser.parse_args()
    print(json.dumps(run(args.out,args.fields),indent=2))
