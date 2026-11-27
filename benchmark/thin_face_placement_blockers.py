"""Can an already placed cube with two visible faces identify a wrongly occluding block?"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from benchmark.thin_face_placement_run import current_inputs,ref_inputs,registered,read,save,HERE,suite,COHORTS
from benchmark.thin_face_placement_score import placement_scores,read_view,horizontal_slivers,framed,REPORT
from engine.pillar_seams import native_mask,orientation
from engine.visible_placement import _frontmost_cells,_sample_face
from engine.visible_raster import SAMPLE_STRIDE


def votes(im,parts,camera,cells,rows):
    anchors=[r for r in rows if r['existing'] and r['count_axes']==2 and r['shared']>=.5
             and (r['host_visible']<r['host_fraction'] or r['side_visible']<r['side_fraction'])]
    if not anchors:return []
    P,_=orientation(camera,im.size)
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    prior=sorted(cells)
    lookup={c:i for i,c in enumerate(prior)}
    shape=(im.height//SAMPLE_STRIDE,im.width//SAMPLE_STRIDE)
    owner,axes=_frontmost_cells(prior,P,eye,shape,face_axes=True)
    footprint=np.bincount(owner[owner>=0],minlength=len(prior))
    keys={r[k] for r in anchors for k in ('host','side')}
    masks={p['id']:native_mask(p,im.size)[2::4,2::4] for p in parts if p['id'] in keys}
    results=[]
    for r in anchors:
        cell=tuple(r['cell'])
        rivals=defaultdict(list)
        for face in ('host','side'):
            axis=r[face+'_axis']
            sign=1 if eye[axis]>cell[axis]+1 else -1
            hit=_sample_face(P,cell,axis,sign,shape)
            if hit is None:continue
            box,poly,*_=hit
            y,x=np.nonzero(poly & masks[r[face]][box])
            y+=box[0].start;x+=box[1].start
            ids=owner[y,x]
            for idx in np.unique(ids):
                if idx<0 or idx==lookup[cell] or max(abs(a-b) for a,b in zip(cell,prior[idx]))>2:continue
                active=ids==idx
                rivals[idx].append((face,int(active.sum()),int((axes[y[active],x[active]]!=axis).sum())))
        for idx,faces in rivals.items():
            n=sum(f[1] for f in faces); wrong=sum(f[2] for f in faces)
            if n<4:continue
            results.append(dict(anchor=cell,blocker=prior[idx],rays=n,fraction=n/footprint[idx],wrong=wrong/n,
                faces=len(faces),host_atlas=r['host_atlas'],side_atlas=r['side_atlas'],shared=r['shared']))
    return results


def case(job):
    split,f,prior,ctl=job
    raw=read(HERE/'runs/thin-final'/(f['id']+'.json'))
    im,parts,_,cam,base,original,pack=current_inputs(job)
    proposed=votes(im,parts,cam,base,raw['rows'])
    result=dict(id=f['id'],split=split,rgb=raw['rgb'],pack=pack,pairs=[],methods={})
    if not proposed:return result
    consistency=consistent_blockers(im,parts,cam,base,proposed)
    net=consistent_blockers(im,parts,cam,base,proposed,net=True)
    truth,occupied,T,_=ref_inputs(job)
    shift=np.array(original.get('translation',ctl['translation']))
    optional=set()
    methods={m:set() for m in ('broad','protected','reprojected','net_reprojected','oracle')}
    methods['reprojected']={tuple(r['cell']) for r in consistency if r['accepted']}
    methods['net_reprojected']={tuple(r['cell']) for r in net if r['accepted']}
    result['reprojection']=consistency
    result['net_reprojection']=net
    valid=lambda c:tuple(next(iter(registered([c],T)))+shift) in truth
    for r in proposed:
        # broad and protected selection depend exclusively on source-derived features.
        c=r['blocker']
        if r['fraction']>=.40 and r['wrong']>=.50:
            methods['broad'].add(c)
        if r['fraction']>=.85 and r['wrong']>=.80 and r['shared']>=.67 and (r['host_atlas'] or r['side_atlas']):
            methods['protected'].add(c)
        anchor_good=valid(r['anchor'])
        blocker_good=valid(c)
        if anchor_good and not blocker_good:
            methods['oracle'].add(c)
        result['pairs'].append(dict(r,anchor_correct=anchor_good,blocker_correct=blocker_good))
    if split in framed.EVALUATION and any(methods.values()):
        _,_,_,_,_,labels,ids,_,_,_,_=read_view(suite.CONTROL,split,f,prior,ctl)
        optional=horizontal_slivers(labels,ids)
    for k,remove in methods.items():
        if not remove:continue
        before=registered(base,T);after=registered(base-remove,T)
        score=placement_scores(after,before,dict.fromkeys(truth),occupied)
        old={tuple(np.add(c,score['baseline_translation'])) for c in before}
        new={tuple(np.add(c,score['translation'])) for c in after}
        r=score['relative_geometry']
        result['methods'][k]=dict(gained=r['gained'],lost=r['lost'],extra_removed=r['extra_removed'],
            extra_added=r['extra_added'],hidden_removed=r['hidden_removed'],
            required_lost=len(((old-new)&truth)-optional),required_gain=len(((new-old)&truth)-optional))
    return result


def consistent_blockers(im,parts,camera,cells,proposals,*,net=False):
    """Re-render all of each disputed block. Preserve coverage and correctly oriented visible rays."""
    P,rot=orientation(camera,im.size)
    eye=np.linalg.solve(P[:,:3],-P[:,3])
    shape=(im.height//SAMPLE_STRIDE,im.width//SAMPLE_STRIDE)
    axes=np.zeros((3,*shape),bool)
    for part in parts:
        if part.get('normal_camera') is None or not part.get('support'):continue
        direction=rot.T@part['normal_camera']
        ax=int(abs(direction).argmax())
        if abs(direction[ax])>=.7:
            axes[ax]|=native_mask(part,im.size)[2::4,2::4]
    source=axes.any(0)
    candidates=sorted({r['blocker'] for r in proposals if r['wrong']>=.8 and r['shared']>=.50
                       and (r['host_atlas'] or r['side_atlas'])})
    result=[]
    kept=set(cells)
    def render(c):
        owner,axis=_frontmost_cells(sorted(c),P,eye,shape,face_axes=True)
        return owner>=0,(owner>=0) & np.take_along_axis(axes,axis.clip(min=0)[None],axis=0)[0]
    covered,correct=render(kept)
    for c in candidates:
        next_covered,next_correct=render(kept-{c})
        uncovered=int((source & covered & ~next_covered).sum())
        gain=int((next_correct&~correct).sum())
        loss=int((correct&~next_correct).sum())
        accept=uncovered==0 and gain>=4 and (gain>=4*loss if net else loss==0)
        result.append(dict(cell=c,lost_coverage=uncovered,lost_correct_axis=loss,gained_correct_axis=gain,accepted=accept))
        if accept:
            kept.remove(c)
            covered,correct=next_covered,next_correct
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits',nargs='+',default=['train'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(case,suite.jobs(args.splits)))
    save(HERE/'runs'/('blocker-'+args.splits[0]+'.json'),rows)
    for name,splits in COHORTS.items():
        group=[r for r in rows if r['split'] in splits]
        if not group:continue
        print(name,len(group),sum(len(r['pairs']) for r in group),
             {k:dict(sum((Counter(r['methods'].get(k,{})) for r in group),Counter()))
              for k in ('broad','protected','reprojected','net_reprojected','oracle')})
