"""Grade supplemental masks against the strict returns plus scoped-texture combined default."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from benchmark.inset_mask_chooser_train import ROOT, OUT
from benchmark import facade_inset_round as suite
from benchmark.inset_mask_chooser import refine
from benchmark.texture_camera_eval import checked,digest,read,save
from benchmark.wall_texture_evaluate import metrics
from engine import inset_returns,surface_model,texture_guided_count
from engine.pillar_seams import native_mask


@lru_cache(maxsize=1)
def shipped():
    return {r['id']:r for r in read(ROOT/'benchmark/reviews/facade-inset-round-01/geometry-and-texture.json')['rows']}


def count(image, parts, atlas_enabled):
    atlas=None
    if atlas_enabled:
        from engine.atlas_count import load_atlas
        atlas=load_atlas(suite.ATLAS,'26.3')
    with TemporaryDirectory() as temp:
        rows=surface_model._count_regions(image,parts,'26.3',texture_guided_count.METHOD,atlas,Path(temp))['surfaces']
    return {r['id']:r['dimensions']['blocks'] for r in rows}


def inspect(job):
    learned,count_guard,collar,job=job
    split,frame,_,_=job
    image,original,counts,camera=suite.ordinary_inputs(*job)
    from engine.facade_inset_phase import supplemental
    base,strict=inset_returns.refine(image,original,camera) if inset_returns.possible(image.size,original) else (original,[])
    before,texture=supplemental(image,base,camera,strict) if strict else (base,[])
    gold=shipped()[frame['id']]
    assert len(strict)==gold['returns'] and len(texture)==gold['texture'] and frame['files']['rgb.png']==gold['rgb_sha256']
    after,events=refine(image,original,camera,before,strict,learned=learned)
    if count_guard and events:
        from benchmark.inset_outline_count_guard import apply
        atlas=None
        if 'atlas_phase' in counts:
            from engine.atlas_count import load_atlas
            atlas=load_atlas(suite.ATLAS,'26.3')
        after,events=apply(image,before,after,events,camera,atlas,collar=collar)
    result=dict(id=frame['id'],split=split,rgb=frame['files']['rgb.png'],accepted=len(events))
    if not events:
        return result
    labels,truth=suite.visible_map(split,frame,metadata=True)
    sizes=np.bincount(truth.ravel())
    faces={f['surface_id']:f for f in labels['faces']}
    pre=metrics(before,truth,image.size)
    post=metrics(after,truth,image.size)
    a={p['id']:p for p in pre['parts']}
    b={p['id']:p for p in post['parts']}
    gained=set(post['matched'])-set(pre['matched'])
    lost=set(pre['matched'])-set(post['matched'])
    def best_iou(data,sid):
        return max((z['intersections'].get(sid,0)/max(1,z['pixels']+int(sizes[sid])-z['intersections'].get(sid,0))
                    for z in data['parts']),default=0.)
    falls={str(sid):[round(best_iou(pre,sid),6),round(best_iou(post,sid),6)] for sid in pre['matched']
           if best_iou(pre,sid)>best_iou(post,sid)+1.e-9}
    before_ids={e['host'] for e in events}
    after_ids=before_ids|{e['output'] for e in events}
    before_dims=count(image,[p for p in before if p['id'] in before_ids],'atlas_phase' in counts)
    after_dims=count(image,[p for p in after if p['id'] in after_ids],'atlas_phase' in counts)
    for e in events:
        p=b[e['output']]
        e['pixels']=p['pixels']
        sid=p['dominant']
        n=p['correct_if_dominant']
        host_sid=a[e['host']]['dominant']
        face,host=faces[sid],faces[host_sid]
        parallel=face['normal_world']==host['normal_world']
        depth=int(np.dot(host['normal_world'],np.subtract(host['xyz_world'],face['xyz_world']))) if parallel else None
        axis=np.flatnonzero(face['normal_world'])[0]
        xyz=np.asarray([f['xyz_world'] for f in labels['faces'] if f['surface_id']==sid])
        hx=np.asarray([f['xyz_world'] for f in labels['faces'] if f['surface_id']==host_sid])
        tangent=2-axis if axis!=1 else 0
        extent=(np.ptp(xyz,axis=0)+1)[[tangent,1]].astype(int).tolist()
        h_extent=(np.ptp(hx,axis=0)+1)[[tangent,1]].astype(int).tolist()
        recovered=sid in gained and parallel and depth>0 and n/p['pixels']>=.80 and n/(p['pixels']+sizes[sid]-n)>=.5
        e.update(reference=sid,host_reference=host_sid,purity=n/p['pixels'],iou=float(n/(p['pixels']+sizes[sid]-n)),
                 verified_novel=bool(recovered),true_depth=depth,shape_extent=extent,host_extent=h_extent,
                 before_host_dims=before_dims[e['host']],after_host_dims=after_dims[e['host']],back_dims=after_dims[e['output']])
        e['guess_has_correct_depth']=bool(e['evidence']['depth']==depth)
        e['new_foreground_px_wrong']=p['pixels']-n
        e['previously_correct_host_px_reassigned']=p['intersections'].get(host_sid,0) if sid!=host_sid else 0
        e['background_px_reassigned']=p['background']
    union0=np.logical_or.reduce([native_mask(p,image.size) for p in before])
    union1=np.logical_or.reduce([native_mask(p,image.size) for p in after])
    assert np.array_equal(union0,union1)
    result.update(events=events,gained=sorted(gained),lost=sorted(lost),previous_match_iou_declined=falls,unmatched=post['unmatched_predictions']-pre['unmatched_predictions'],
                  paired=post['paired_pixels']-pre['paired_pixels'],background=post['background_pixels']-pre['background_pixels'])
    return result


def main(workers,learned,count_guard,collar):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(inspect,((learned,count_guard,collar,j) for j in suite.jobs([*suite.framed.SIZES,'shelter']))))
    groups={'development':['train'],'fixed':suite.framed.EVALUATION,'alternate':['extra'],'new':['new'],'shelter':['shelter']}
    summary={}
    for name,splits in groups.items():
        cases=[r for r in rows if r['split'] in splits]
        changed=[r for r in cases if r['accepted']]
        events=[e for r in changed for e in r['events']]
        summary[name]=dict(views=len(cases),changed=len(changed),accepted=len(events),verified=sum(e['verified_novel'] for e in events),
                           wrong_depth=sum(not e['guess_has_correct_depth'] for e in events),unmatched=sum(r['unmatched'] for r in changed),
                           paired=sum(r['paired'] for r in changed),background=sum(r['background'] for r in changed),
                           gained=sum(len(r['gained']) for r in changed),lost=sum(len(r['lost']) for r in changed))
    filename='collar.json' if collar else 'count-guard.json' if count_guard else 'results.json' if learned else 'no-tree.json'
    save(OUT/filename,dict(baseline_sha256=digest(ROOT/'benchmark/reviews/facade-inset-round-01/geometry-and-texture.json'),
            model_sha256=digest(OUT/'model.json'),groups=summary,frames=rows))
    print(summary)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=5)
    p.add_argument('--no-tree',action='store_true')
    p.add_argument('--count-guard',action='store_true')
    p.add_argument('--collar',action='store_true')
    args=p.parse_args()
    main(args.workers,not args.no_tree,args.count_guard or args.collar,args.collar)
