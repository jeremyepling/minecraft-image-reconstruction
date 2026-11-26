"""Read-only diagnostic for the eight published surface-matching views."""
from collections import Counter
import numpy as np
from PIL import Image
from benchmark.camera_round import ROOT,digest,read,save
from benchmark.normal_fields import cached_fields
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE,recorded
from benchmark.perimeter_sheet import load_parts
from benchmark.mask_metrics import footprint
from engine.geometry import bilinear
from runtime.mode import image_only


@image_only()
def audit(output, fields):
    run=ROOT/'benchmark/reviews/perimeters-01'
    manifest=read(run/'manifest.json')
    sheet=read(run/'surface-detection.json')
    assert digest(run/'surface-detection.json')==manifest['artifacts']['surface-detection.json']
    assert digest(run/'cases.json')==manifest['artifacts']['cases.json']
    assert digest(ARCHIVE)==manifest['reference_sha256']
    cases={r['key']:r for r in read(run/'cases.json')}
    groups_by_kind,_,fixtures=groups()
    assert fixtures==manifest['fixtures']
    captures={f'{c.floor}/{c.id}':c for g in groups_by_kind.values() for c in g}
    stats=Counter()
    out=[]
    for view in sheet['views']:
        key=view['frame']
        case=captures[key]
        row=cases[key]
        sha=view['image_sha256']
        assert digest(case.path('image'))==sha==row['sha256']
        assert digest(run/row['predictions'])==manifest['artifacts'][row['predictions']]
        parts=load_parts(run,row,None,None,'color-edges')
        restored=cached_fields(sha,fields,kinds=('normal',))
        if restored is None:
            raise ValueError('Normal field cache unavailable')
        field=restored['normal']
        _,ref=recorded(case)
        ids=ref['id_map']
        patches=ref['patches']
        width,height=ref['image_size']
        y,x=np.indices(ids.shape)
        xy=(np.stack((x,y),axis=-1)+.5)*4
        vectors=bilinear(field,xy*np.array(field.shape[1::-1])/ref['image_size'])
        vectors/=np.maximum(np.linalg.norm(vectors,axis=2,keepdims=True),1.e-9)
        with Image.open(case.path('image')) as im:
            rgb=np.asarray(im.convert('RGB'),dtype=float)
        lum=bilinear(rgb,xy)@np.array([.2126,.7152,.0722])
        masks=[footprint(p,ids.shape,(width,height),True) for p in parts]
        sizes=np.bincount(ids.ravel(),minlength=1+max(patches))
        hits=np.array([np.bincount(ids[m],minlength=len(sizes)) for m in masks])
        for r in view['references']:
            rid=r['id']
            target=ids==rid
            p=patches[rid]
            world_normal=p['normal']
            status=r['status']
            kind='underside' if world_normal[1]<0 else 'top' if world_normal[1]>0 else 'vertical'
            stats[(status,kind)]+=1
            if status!='missed' and kind!='underside':
                continue
            other=r['best_prediction_matched_ref']
            pi=r['best_prediction']
            competing=[]
            if pi and status in ('found','missed'):
                assert np.isclose(hits[pi-1,rid]/sizes[rid],r['coverage'])
                for other_id,other_p in patches.items():
                    if other_id==rid or not other_p['eligible'] or hits[pi-1,other_id]<sizes[other_id]/2:
                        continue
                    competing.append(dict(ref=other_id,coverage=float(hits[pi-1,other_id]/sizes[other_id]),
                        normal=other_p['normal'],offset=other_p['offset']))
            angles=np.degrees(np.arccos(np.clip(vectors[target]@p['normal_camera'],-1,1)))
            contrast={}
            # Paired neighbor samples across an actual boundary, excluding open air and floor (ID 0).
            edges={}
            for a,b in [(np.s_[:-1,:],np.s_[1:,:]),(np.s_[:,:-1],np.s_[:,1:])]:
                for inside,outside in ((a,b),(b,a)):
                    neighbor_ids=ids[outside]
                    boundary=target[inside]&(neighbor_ids!=rid)&(neighbor_ids>0)
                    for oid in np.unique(neighbor_ids[boundary]):
                        crossing=boundary&(neighbor_ids==oid)
                        bucket=edges.setdefault(int(oid),[])
                        bucket.extend((lum[inside][crossing]-lum[outside][crossing]).tolist())
            for oid,steps in sorted(edges.items(),key=lambda kv:-len(kv[1])):
                contrast[oid]=dict(n=len(steps),median_signed=float(np.median(steps)),
                       median_abs=float(np.median(np.abs(steps))))
            rec=dict(frame=key,layout=view['layout'],ref=rid,kind=kind,status=status,normal=world_normal,
                     offset=p['offset'],pixels=r['pixels'],samples=r['samples'],proposal=pi,other=other,
                     coverage=r['coverage'],iou=r['iou'],others_half_covered=competing,
                     signed_normal_angle_median=float(np.median(angles)) if len(angles) else None,
                     normal_within_30=float((angles<=30).mean()) if len(angles) else None,
                     native_brightness_steps_4px=contrast)
            out.append(rec)
    missed=[r for r in out if r['status']=='missed']
    stats2=Counter()
    for r in missed:
        others=r['others_half_covered']
        high=r['coverage']>=.5
        typ=('matched other' if r['other'] else 'unmatched')
        stats2[typ]+=1
        stats2['coverage>=50%']+=high
        stats2['own>=50%,same proposal >=50% second scored surface']+=bool(high and others)
        stats2['own>=50%,same proposal >=50% parallel second scored surface at another depth']+=bool(high and any(r['normal']==c['normal'] and r['offset']!=c['offset'] for c in others))
        stats2['own>=50%,same proposal >=50% parallel second scored surface at same depth']+=bool(high and any(r['normal']==c['normal'] and r['offset']==c['offset'] for c in others))
    result=dict(schema=1,run='perimeters-01',method='color-edges',frames=len(sheet['views']),
                archive_sha256=manifest['reference_sha256'],case_sha256=manifest['artifacts']['cases.json'],
                summary={a+':'+b:n for (a,b),n in stats.items()},miss_types=dict(stats2),
                notes='Other half-covered references are eligible and cover >=50% of that reference, evaluated on the primary stride-4 grid. Threshold is an explanatory diagnostic. Raw normal angles are signed and reference-conditioned. Brightness steps are native RGB luminance sampled four pixels apart (inside minus neighboring surface).',
                references=out)
    save(output,result)
