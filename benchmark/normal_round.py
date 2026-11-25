"""Paired one-image comparisons of visible plane outlines and relative depth ordering."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from PIL import __version__ as pillow_version

from benchmark.camera_round import ROOT, digest, fixture_digest, read, revision, save
from benchmark.depth_probe_score import occlusion_pairs
from benchmark.exterior_set import Capture, check_capture_conditions, load_suite
from benchmark.normal_fields import fields
from benchmark.surface_metrics import score_pitch, score_planes, DOWNSAMPLE, observed_support
from benchmark.surface_round import (image_of, recorded, outline_summary, depth_summary,
                                     lift_scores, pitch_summary, artifact_path)
from benchmark.surface_topology import opening_mask, opening_coverage
from engine.geometry import bilinear
from engine.surface_fields import MODELS, SETTINGS as INFERENCE
from engine.surface_model import assemble, intrinsic, pitch, proposals_for
from engine.surface_order import order, SETTINGS as ORDER_SETTINGS


METHODS=('edges+shade','normal-rank','normal-regions')
BASE=METHODS[0]
RULE='Freeze highest mean layout F1 at IoU >= 0.5 on shopfront and wing. On a tie retain registry order.'
ADOPTION='Promote if fresh layout-mean F1 exceeds baseline and pooled opening fill grows by <= 3 percentage points.'
ORDER_RULE=('Keep relative ordering experimental unless Depth Anything yields more correct pairs than geometry, '
            'with at least half of all emitted relations verified as correct, on development and blind cohorts.')


def groups():
    from benchmark.surface_fixture import FROZEN as old, FROZEN_SHA256 as old_hash
    from benchmark.normal_fixture import FROZEN as fresh, FROZEN_SHA256 as fresh_hash
    suite,paired=load_suite()
    cohorts={'development':[],'regression':[],'blind':[]}
    for variants in paired.values():
        for c in variants.values():
            cohorts['development' if c.frame['split']=='dev' else 'regression'].append(c)
    seen={c.frame['layout'] for cases in cohorts.values() for c in cases}
    for root,expected,cohort in ((old,old_hash,'regression'),(fresh,fresh_hash,'blind')):
        index=read(root/'index.json')
        if fixture_digest(root,index)!=expected or index['client_sha256']!=suite['client_sha256']:
            raise ValueError(f'Fixture or client changed for {root.name}')
        layouts={f['layout'] for f in index['frames']}
        if layouts & seen or any(f['split']!='holdout' for f in index['frames']):
            raise ValueError('Underlying layouts cross cohort boundaries')
        seen.update(layouts)
        for f in index['frames']:
            c=Capture(f['floor'],root,f)
            check_capture_conditions(read(c.path('pose')),suite['capture_conditions'],c.id)
            cohorts[cohort].append(c)
    hashes={name:spec['sha256'] for name,spec in suite['conditions'].items()}
    return cohorts,suite,dict(**hashes,surface_confirm_v1=old_hash,normal_confirm_v1=fresh_hash)


def verify_baseline(case,scores):
    previous=ROOT/'benchmark/reviews/surfaces-01/cases'/case.floor/case.id/'outlines.json'
    if previous.is_file():
        ref=read(previous)[BASE]['scores']
        for key in ('matched','eligible','proposed','spill_pixels','proposed_pixels'):
            if scores[key]!=ref[key]:
                raise ValueError(f'Published baseline changed for {case.id}/{case.floor}: {key}')


def relation_score(relations,parts,matches,truth_pairs,eligible_patches=None):
    raw={(r['near_patch'],r['far_patch']) for r in truth_pairs['pairs']}
    supported={(a,b) for a,b in raw if eligible_patches is None or {a,b}<=eligible_patches}
    targets={(a,b) for a,b in supported if (b,a) not in supported}
    keys={parts[m['proposal_index']]['id']:m['patch_id'] for m in matches}
    found={(keys[r['near']],keys[r['far']]) for r in relations if r['near'] in keys and r['far'] in keys}
    wrong=sum((far,near) in targets for near,far in found)
    return dict(eligible=len(targets),excluded_small=len(raw)-len(supported),ambiguous=len(supported)-len(targets),
                proposed=len(relations),correct=len(found & targets),reversed=wrong,
                other=len(relations)-len(found & targets)-wrong)


def normal_diagnostic(t,normal,image_size):
    ids=t['id_map']
    y,x=np.where(ids>0)
    xy=(np.column_stack((x,y))+.5)*DOWNSAMPLE*np.array(normal.shape[1::-1])/image_size
    samples=bilinear(normal,xy)
    norm=samples/np.maximum(np.linalg.norm(samples,axis=1,keepdims=True),1.e-10)
    expected=np.array([[0,0,0]]+[t['patches'][i]['normal_camera'] for i in sorted(t['patches'])])
    cos=np.clip(np.sum(norm*expected[ids[y,x]],axis=1),-1,1)
    soffits=[i for i,p in t['patches'].items() if p['normal']==[0,-1,0]]
    underside=np.isin(ids[y,x],soffits)
    ok=cos>=np.cos(np.radians(15))
    return dict(samples=len(cos),within_15=int(ok.sum()),soffit_samples=int(underside.sum()),
                soffit_within_15=int(ok[underside].sum()))


def support_gap(openings,parts,outline_gap,size):
    if not any('support' in p for p in parts):
        return outline_gap
    covered=np.zeros_like(openings)
    for p in parts:
        if 'support' not in p:
            from benchmark.camera_labels import rasterize
            q=np.array(p['quad'])/DOWNSAMPLE
            hit=rasterize(np.column_stack([q,np.ones(4)]),openings.shape[1],openings.shape[0])
            if hit:
                covered[hit[0]]|=hit[1]
            continue
        covered|=observed_support(p['support'],openings.shape,size)
    return dict(opening_pixels=int(openings.sum()),filled_pixels=int((covered&openings).sum()))


def evaluate(job):
    case,cohort,cache,out=job
    source=case.path('image')
    sha=case.frame['image_sha256']
    normal,depth=fields(source,sha,cache)
    image=image_of(case)
    # All three predictions are complete before any pose, state, or label is opened.
    pred={}
    for method in METHODS:
        parts=proposals_for(image,'26.3',method,normal)
        pitches=[pitch(image,p['quad'],'26.3','repeat') for p in parts]
        model=assemble(parts,pitches,image.size)
        relations={key:order(model['planes'],image.size,field) for key,field in
                   (('geometry',None),('depth-anything',depth))}
        pred[method]=dict(model=model,relations=relations)
    pose,t=recorded(case)
    from benchmark.camera_metrics import truth_camera
    P=truth_camera(pose)
    axis=P[2,:3]/np.linalg.norm(P[2,:3])
    focal=np.linalg.norm(P[0,:3]-(P[0,:3]@axis)*axis)/np.linalg.norm(P[2,:3])
    known_K=intrinsic(image.size,focal)
    gap=opening_mask(t['id_map'])
    depth_truth=occlusion_pairs(t,depth)
    results={}
    for method,d in pred.items():
        parts=d['model']['planes']
        scores=score_planes(t,parts)
        if method==BASE and cohort!='blind':
            verify_baseline(case,scores)
        matches=scores['matches']
        count_scores=[]
        for m in matches:
            p=parts[m['proposal_index']]
            count_scores.append(score_pitch(t,m['patch_id'],p['quad'],p['pitch']))
        eligible={i for i,p in t['patches'].items() if p['eligible']}
        relations={k:relation_score(v,parts,matches,depth_truth,eligible) for k,v in d['relations'].items()}
        known=assemble([{key:p[key] for key in ('id','quad')} for p in parts],
                       [p['pitch'] for p in parts],image.size,known_K)
        openings=opening_coverage(gap,parts)
        support=score_planes(t,parts,supported_pixels=True) if any('support' in p for p in parts) else scores
        angles=[]
        for match in matches:
            p=parts[match['proposal_index']]
            if p.get('observed_normal_camera') is not None:
                cosine=np.dot(match['normal_camera'],p['observed_normal_camera'])
                angles.append(float(np.degrees(np.arccos(np.clip(cosine,-1,1)))))
        results[method]=dict(outlines=scores,gaps=openings,support_matches=support['matched'],
                             support_gaps=support_gap(gap,parts,openings,image.size),observed_angles=angles,pitch=count_scores,
                             lift=lift_scores(d['model'],matches),known_lift=lift_scores(known,matches),order=relations,
                             soffits_matched=sum(t['patches'][m['patch_id']]['normal']==[0,-1,0] for m in matches))
    name=f'{case.floor}/{case.id}'
    folder=out/'predictions'/case.floor
    folder.mkdir(exist_ok=True,parents=True)
    target=folder/(case.id+'.json.gz')
    target.write_bytes(gzip.compress(json.dumps(pred,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    return dict(id=case.id,floor=case.floor,layout=case.frame['layout'],cohort=cohort,
                source=artifact_path(source,out),sha256=sha,key=name,prediction=artifact_path(target,out),
                methods=results,depth_diagnostic=depth_truth,normal_diagnostic=normal_diagnostic(t,normal,image.size),
                soffits_eligible=sum(p['normal']==[0,-1,0] and p['eligible'] for p in t['patches'].values()))


def totals(rows,keys):
    return {k:sum(r[k] for r in rows) for k in keys}


def summary(rows):
    result=dict(frames=len(rows),methods={})
    for method in METHODS:
        data=[r['methods'][method] for r in rows]
        o=outline_summary([d['outlines'] for d in data])
        gaps=totals([d['gaps'] for d in data],('filled_pixels','opening_pixels'))
        gaps['fill_rate']=gaps['filled_pixels']/gaps['opening_pixels'] if gaps['opening_pixels'] else None
        support_gaps=totals([d['support_gaps'] for d in data],('filled_pixels','opening_pixels'))
        angles=[a for d in data for a in d['observed_angles']]
        result['methods'][method]=dict(outlines=o,gaps=gaps,support_matches=sum(d['support_matches'] for d in data),
            observed_normal_median_deg=float(np.median(angles)) if angles else None,support_gaps=support_gaps,
            pitch=pitch_summary([p for d in data for p in d['pitch']]),
            lift=depth_summary([m for d in data for m in d['lift']]),
            known_lift=depth_summary([m for d in data for m in d['known_lift']]),soffits_matched=sum(d['soffits_matched'] for d in data),
            order={m:totals([d['order'][m] for d in data],('eligible','proposed','correct','reversed','other'))
                   for m in ('geometry','depth-anything')})
    result['soffits_eligible']=sum(r['soffits_eligible'] for r in rows)
    result['oracle_depth_order']=totals([r['depth_diagnostic'] for r in rows],('eligible_pairs','correct_order'))
    result['oracle_normal_samples']=totals([r['normal_diagnostic'] for r in rows],('samples','within_15','soffit_samples','soffit_within_15'))
    return result


def layout_means(rows):
    layouts=sorted({r['layout'] for r in rows})
    by_layout=[summary([r for r in rows if r['layout']==layout]) for layout in layouts]
    return {m:float(np.mean([s['methods'][m]['outlines']['f1'] for s in by_layout])) for m in METHODS}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--cache',type=Path,default=ROOT/'runs/normal-fields')
    ap.add_argument('--workers',type=int,default=3)
    args=ap.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=(args.out or ROOT/'runs'/('normals-'+stamp)).resolve()
    out.mkdir(parents=True,exist_ok=False)
    cohorts,suite,hashes=groups()
    archive_dir=ROOT/'benchmark/worlds/normal-confirm-v1'
    world_index=read(archive_dir/'world-index.json')
    world=world_index['worlds'][0]
    if digest(archive_dir/world['archive'])!=world['sha256'] or world['clock_ticks']!=6000 or world['raining'] or world['thundering']:
        raise ValueError('Frozen clear-noon confirmation world differs from its index.')
    rows=[]
    code=revision()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for cohort,cases in cohorts.items():
            print(f'{cohort}: evaluating {len(cases)} single-frame cases',flush=True)
            batch=list(pool.map(evaluate,[(c,cohort,args.cache,out) for c in cases]))
            rows.extend(batch)
            if cohort=='development':
                means=layout_means(batch)
                winner=max(METHODS,key=means.get)
                save(out/'selection.json',dict(rule=RULE,adoption=ADOPTION,order_rule=ORDER_RULE,mean_layout_f1=means,selected=winner))
                print(f'Frozen development choice: {winner}',flush=True)
    six=[r for r in rows if r['cohort']!='blind']
    blind=[r for r in rows if r['cohort']=='blind']
    report=dict(six=summary(six),blind=summary(blind),layouts={k:summary([r for r in rows if r['layout']==k])
                for k in sorted({r['layout'] for r in rows})},floors={floor:summary([r for r in six if r['floor']==floor])
                for floor in sorted({r['floor'] for r in six})},cohorts={k:summary([r for r in rows if r['cohort']==k]) for k in cohorts})
    fresh_means=layout_means(blind)
    fresh=report['blind']['methods']
    gap=lambda m:fresh[m]['gaps']['fill_rate'] or 0.
    adopted=winner if fresh_means[winner]>fresh_means[BASE] and gap(winner)-gap(BASE)<=.03 else BASE
    def improves_order(cohort):
        scores=report['cohorts'][cohort]['methods'][adopted]['order']
        learned=scores['depth-anything']
        return learned['correct']>scores['geometry']['correct'] and learned['correct']*2>=learned['proposed']
    adopted_order='depth-anything' if all(improves_order(k) for k in ('development','blind')) else 'geometry'
    save(out/'scores.json',report)
    save(out/'cases.json',rows)
    (out/'feedback.jsonl').touch()
    from benchmark.normal_sheet import render
    render(out,rows,winner)
    import onnxruntime as ort
    from engine.normal_rank import SETTINGS as ranking
    from engine.normal_regions import SETTINGS as regions
    manifest=dict(run_id=out.name,suite='normal-round',code=code,version=suite['version'],client_sha256=suite['client_sha256'],
       capture=suite['capture_conditions'],fixtures=hashes,split={k:sorted({c.frame['layout'] for c in cs}) for k,cs in cohorts.items()},
       blind_world=dict(path=artifact_path(archive_dir/world['archive'],out),sha256=world['sha256']),
       methods=dict(outlines=list(METHODS),pitch=['repeat'],order=['geometry','depth-anything'],lift='shared-edges-v1'),
       selected_on_development=winner,recommended_outlines=adopted,recommended_depth_order=adopted_order,
       selection_rule=RULE,adoption_rule=ADOPTION,order_rule=ORDER_RULE,
       model_sha256={k:v[2] for k,v in MODELS.items()},settings=dict(inference=INFERENCE,normal_rank=ranking,
            normal_regions=regions,ordering=ORDER_SETTINGS),environment=dict(python=sys.version.split()[0],onnxruntime=ort.__version__,
            pillow=pillow_version,numpy=np.__version__),scores={k:{m:s['outlines'] for m,s in report[k]['methods'].items()} for k in ('six','blind')},
       artifacts=dict(cases='cases.json',scores='scores.json',selection='selection.json',review='comparison.jpg',
                      regions='review-regions.json',visual_feedback='feedback.jsonl',predictions='predictions'))
    save(out/'manifest.json',manifest)
    print('Recommended outline method:',adopted,flush=True)
    print(out,flush=True)


if __name__=='__main__':
    main()
