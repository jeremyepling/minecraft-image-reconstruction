"""Run occlusion-aware edge votes on the published inset default and grade native mask ownership."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from benchmark import facade_inset_round as suite
from benchmark.edge_vote_outline import adjustments, revised_parts
from benchmark.inset_default_verification import local_release
from benchmark.mask_blocks import decoded_pack, registered
from benchmark.relative_geometry import placement_scores
from benchmark.texture_camera_eval import checked,digest,read,save
from benchmark.wall_texture_evaluate import metrics
from engine import inset_returns,surface_model,texture_guided_count,visible_placement
from engine.plane_consensus import row_runs
from engine.pillar_seams import native_mask
from engine.surface_perimeter import with_outline
from engine.surface_boundary_refine import _retain_topology
from engine.visible_inset_cells import inset_edits

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'benchmark/reviews/facade-local-placement-01'
OUT=ROOT/'benchmark/reviews/edge-vote-outlines-01'


def baseline_pack(job):
    split,frame,prior,control=job
    key=frame['id']
    rows,z=local_release()
    row=rows[key]
    assert frame['files']['rgb.png']==row['rgb_sha256'] and control['pack_sha256']==row['control_pack_sha256']
    if row['methods']:
        ref=row['methods']['clear_front']
        expected=ref['pack_sha256']
        data=z.read(f'clear_front/{split}/{key}/reconstruction.zip')
    elif split=='shelter':
        expected=row['control_pack_sha256']
        with ZipFile(suite.ARCHIVE) as archive:
            data=archive.read('production/transfer/'+key+'/reconstruction.zip')
    else:
        expected=row['control_pack_sha256']
        data=checked(suite.CONTROL/split/key/'reconstruction.zip',expected).read_bytes()
    assert sha256(data).hexdigest()==expected
    return expected,set(decoded_pack(BytesIO(data)))


def recount(image,parts,atlas=None):
    with TemporaryDirectory() as out:
        return surface_model._count_regions(image,parts,'26.3',texture_guided_count.METHOD,atlas,Path(out))


def changes(parts,updated,truth,ids):
    before=metrics(parts,truth,truth.shape[::-1])
    after=metrics(updated,truth,truth.shape[::-1])
    oldby={p['id']:p for p in before['parts']}
    newby={p['id']:p for p in after['parts']}
    areas=np.bincount(truth.ravel())
    def iou(data,sid):
        inter=data['intersections'].get(sid,0)
        return inter/max(1,areas[sid]+data['pixels']-inter)
    host={i+1:oldby[p['id']]['dominant'] for i,p in enumerate(parts)}
    old,new,flat=ids
    targets=truth.ravel()[flat]
    oldref=np.array([host[k] for k in old.ravel()[flat]])
    newref=np.array([host[k] for k in new.ravel()[flat]])
    improved=(oldref!=targets)&(newref==targets)
    worse=(oldref==targets)&(newref!=targets)
    ious=[iou(newby[k],p['dominant'])-iou(p,p['dominant']) for k,p in oldby.items()]
    return dict(gained_surfaces=sorted(set(after['matched'])-set(before['matched'])),
                lost_surfaces=sorted(set(before['matched'])-set(after['matched'])),
                unmatched=after['unmatched_predictions']-before['unmatched_predictions'],
                background=after['background_pixels']-before['background_pixels'],
                paired=after['paired_pixels']-before['paired_pixels'], improved=int(improved.sum()),
                harmed=int(worse.sum()), other=int(len(flat)-improved.sum()-worse.sum()),
                higher_iou=int(sum(v>1.e-8 for v in ious)),lower_iou=int(sum(v< -1.e-8 for v in ious)),
                mean_iou_delta=sum(ious)/len(ious))


def geometry(job,image,original,parts,changed,counts,camera,cells,strict,extra):
    split,frame,prior,control=job
    def place(final):
        atlas=None
        if 'atlas_phase' in counts:
            from engine.atlas_count import load_atlas
            atlas=load_atlas(suite.ATLAS,'26.3')
        cc=recount(image,final,atlas)
        if strict or extra:
            # Inset hosts still use the original global layout. Replay it with edge adjustments on surviving original parts.
            pairs={a['id']:(a,p) for a,p in zip(parts,final) if p is not a}
            original_base=[]
            for original_part in original:
                key=original_part['id']
                if key in pairs:
                    before,after=(native_mask(q,image.size) for q in pairs[key])
                    mask=native_mask(original_part,image.size)
                    mask[before!=after]=after[before!=after]
                    original_part=dict(original_part,support=dict(size=list(image.size),runs=row_runs(mask)),support_area=int(mask.sum()))
                original_base.append(with_outline(original_part))
            cc=recount(image,original_base,atlas)
            pp=original_base
        else:
            pp=final
        result,_=visible_placement.place(image,'26.3',pp,counts=cc,camera=camera,atlas_path=suite.ATLAS)
        if strict:
            add,sub=inset_edits(image.size,camera,final,strict,result)
            result=(set(result)|add)-sub
        return set(result)
    # Same-code unedited replay is required before attributing a coordinate delta to the outline.
    base=place(parts)
    now=place(changed)
    common=dict(control_equal=base==cells, replay_control_delta=len(base^cells), added=len(now-base),removed=len(base-now))
    if now==base or base!=cells:
        return common
    root=suite.SHELTER if split=='shelter' else suite.framed.NEW_INPUTS if split=='new' else suite.framed.transfer.BUILDS
    inputs=suite.SHELTER if split=='shelter' else suite.framed.NATIVE_INPUTS.get(split)
    ref='old' if split in ('extra','new','shelter') else split
    truth,occupied,tr,_=suite.framed.references(ref,frame,inputs,suite.framed.TRAINING_DATA,suite.framed.DEPTH,
        dict(native=suite.framed.native.BUILDS,old=root,fresh=root))
    scores=placement_scores(registered(now,tr),registered(base,tr),dict.fromkeys(truth),occupied)
    score=scores['relative_geometry']
    common['relative']={k:score[k] for k in ('correct','missed','extras','gained','lost','extra_added','extra_removed','hidden_added','hidden_removed')}
    if split in suite.framed.EVALUATION:
        from benchmark.roof_span_worst_stages import read_view
        from benchmark.strip_visibility_audit import horizontal_slivers
        _,_,_,_,_,labels,ids,tgt,*_=read_view(suite.CONTROL,split,frame,prior,control)
        optional=horizontal_slivers(labels,ids)
        placed=suite.framed.spacing.shift(registered(now,tr),scores['translation'])
        previous=suite.framed.spacing.shift(registered(base,tr),scores['baseline_translation'])
        gain,loss=placed-previous,previous-placed
        common['priority_delta']=dict(required_gained=len((gain&tgt)-optional), required_lost=len((loss&tgt)-optional),
            extra_added=len(gain-tgt),extra_removed=len(loss-tgt))
    return common


def inspect(work):
    job,physical=work
    split,frame,prior,control=job
    image,orig,counts,camera=suite.ordinary_inputs(*job)
    parts,strict,phase,extra,_=inset_returns.detect(image,orig,'26.3',camera=camera)
    atlas=None
    if 'atlas_phase' in counts:
        from engine.atlas_count import load_atlas
        atlas=load_atlas(suite.ATLAS,'26.3')
    if parts is not orig:
        parts=[with_outline(p) for p in parts]
        counts=recount(image,parts,atlas)
        surface_model._inset_layout_counts(image,'26.3',[with_outline(p) for p in orig],counts,
            {event['host'] for event in strict+extra},texture_guided_count.METHOD,atlas)
    digest_pack,cells=baseline_pack(job)
    owner,runs,options=adjustments(image,parts,counts,camera,cells)
    row=dict(id=frame['id'],split=split,rgb=frame['files']['rgb.png'],pack=digest_pack,
             runs=len(runs),run_blocks=sum(len(c) for _,c in runs),options=len(options),methods={})
    if not options:return row
    # Predict all variants before looking up the reference. Overlapping proposals must agree pixel-by-pixel.
    variants={}
    for mode in ('two','three','strong','interior','protected'):
        selected=[e for e in options if mode=='two' or (e['cells']>=3 and
                  (mode!='strong' or (e['cells']>=4 and e['margin']>=4 and e['strength']>=10)))]
        if not selected:continue
        new=owner.copy()
        veto=np.zeros(owner.shape,bool)
        for e in selected:
            if mode=='protected' and (e['cells']<4 or e['margin']<4 or e['strength']<10 or abs(e['offset'])>1):
                continue
            keep=e['clear'] if mode in ('interior','protected') else slice(None)
            flat=e['flat'][keep]
            target=e['after'][keep]
            old=new.flat[flat]
            veto.flat[flat]|=(old!=owner.flat[flat])&(old!=target)
            new.flat[flat]=target
        new[veto]=owner[veto]
        if mode=='protected':
            new=_retain_topology(owner,new,dict(radius=3))
        flat=np.flatnonzero(new!=owner)
        if not len(flat):continue
        updated=revised_parts(parts,owner,new)
        variants[mode]=(updated,new,flat,selected)
    truth=suite.visible_map(split,frame)
    for mode,(updated,new,flat,selected) in variants.items():
        result=changes(parts,updated,truth,(owner,new,flat))
        changedidx=[i for i,p in enumerate(parts) if updated[i] is not p]
        ids={parts[i]['id'] for i in changedidx}
        olddims={r['id']:r['dimensions']['blocks'] for r in recount(image,[p for p in parts if p['id'] in ids],atlas)['surfaces']}
        newdims={r['id']:r['dimensions']['blocks'] for r in recount(image,[updated[i] for i in changedidx],atlas)['surfaces']}
        result['count_changes']={k:[v,newdims[k]] for k,v in olddims.items() if v!=newdims[k]}
        result['operations']=[{k:v for k,v in e.items() if k not in ('flat','after','clear','members','host_axis','side_axis')} for e in selected]
        result['pixels']=len(flat)
        if physical and mode==physical:
            result['geometry']=geometry(job,image,orig,parts,updated,counts,camera,cells,strict,extra)
        row['methods'][mode]=result
    return row


def record(work):
    job,physical,out=work
    row=inspect((job,physical))
    save(out/(row['id']+'.json'),row)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',nargs='+',default=list(suite.framed.SIZES)+['shelter'])
    p.add_argument('--out',type=Path,default=OUT/'cases.json')
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--geometry',choices=['two','three','strong','interior','protected'])
    a=p.parse_args()
    jobs=list(suite.jobs(a.splits))
    records=a.out.with_suffix('')
    records.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(record,[(j,a.geometry,records) for j in jobs]))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    save(a.out,rows)
    print('frames',len(rows),'runs',sum(r['runs'] for r in rows),'options',sum(r['options'] for r in rows),flush=True)
    for m in ('two','three','strong','interior','protected'):
        tally=Counter()
        for r in rows:
            if m in r['methods']:
                row=r['methods'][m]; tally.update({k:v for k,v in row.items() if isinstance(v,int)})
                tally['changed']+=1
        print(m,dict(tally))
