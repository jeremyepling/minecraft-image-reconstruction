"""Trace the original 41 candidate-mask opportunities after final prediction and scoring."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from benchmark.camera_round import ROOT, digest, read, save
from benchmark.mask_metrics import PROTOCOL, footprint, measure
from benchmark.normal_round import groups
from benchmark.recovery_overview import checked
from benchmark.single_wall_sheet import METHODS, REVIEWS, load_gzip
from benchmark.surface_reference import ARCHIVE, recorded
from runtime.mode import image_only


def overlap(one,two):
    both=np.count_nonzero(one & two)
    return float(both/(one.sum()+two.sum()-both))


def omitted_parent(target, raw, masks, parents, supports, settings):
    """Identify the best target-shaped cut across every parent when no vote was offered."""
    best=None
    for pid,parent in supports.items():
        for sid in raw:
            cut=masks[sid]&parent
            iou=overlap(cut,target)
            if best is not None and iou<=best['clipped_iou']:
                continue
            area=int(parent.sum())
            inside=int(cut.sum())
            minimum=max(settings['min_side_area'],settings['min_parent_share']*area)
            if iou<PROTOCOL['iou']:
                reason='no single parent retains half overlap'
            elif abs(parents[pid]['normal_camera'][1])>settings['max_vertical_normal_y']:
                reason='parent orientation excluded'
            elif inside<minimum:
                reason='proposed piece too small'
            elif area-inside<minimum or inside>settings['max_parent_share']*area:
                reason='remainder too small'
            elif inside<settings['min_containment']*masks[sid].sum():
                reason='insufficient containment'
            else:
                reason='insufficient depth sample pairs'
            best=dict(parent=pid,proposal=sid,clipped_iou=iou,reason=reason)
    return best


def trace(case, row, source, hashes, model_files, settings):
    _,ref=recorded(case)
    targets=row['opportunities']
    stem='proposals/'+row['sha256']
    directory=REVIEWS/'mobile-regions-01'
    for ext in ('.npz','.json.gz'):
        checked(directory/(stem+ext),model_files[stem+ext])
    with gzip.open(directory/(stem+'.json.gz'),'rt') as stream:
        selected=json.load(stream)['selected']
    with np.load(directory/(stem+'.npz'),allow_pickle=False) as data:
        masks=np.unpackbits(data['packed'][selected],axis=2,count=int(data['shape'][2])).astype(bool)
    mask_by_id=dict(zip(selected,masks,strict=True))
    regions=load_gzip(source,row['predictions'],hashes)
    evidence=load_gzip(source,row['boundary_evidence'],hashes)
    assigned={}
    for method in METHODS:
        _,pairs=measure(regions[method],ref,use_mask=True)
        assigned[method]={rid:regions[method][i] for i,rid in pairs}
        if set(assigned[method])!=set(row['scopes'][method]['matched']):
            raise ValueError('Surface assignment changed')
    parents={p['id']:p for p in regions['current']}
    labels=ref['id_map']
    size=ref['image_size']
    supports={pid:footprint(p,labels.shape,size,use_mask=True) for pid,p in parents.items()}
    results=[]
    for rid in targets:
        target=labels==rid
        if rid in assigned['current']:
            raise ValueError('Opportunity already matched in the baseline')
        raw={i:float(j) for i,m in mask_by_id.items() if (j:=overlap(m,target))>=PROTOCOL['iou']}
        if not raw:
            raise ValueError('Earlier model screen lost a candidate')
        offers=[]
        for i,candidate in enumerate(evidence['inventory']):
            pid,sid=candidate['parent'],candidate['proposal']
            if sid not in raw:
                continue
            clipped=overlap(mask_by_id[sid]&supports[pid],target)
            accepted={}
            for method in METHODS[1:]:
                event=evidence['events'][method][i]
                if (event['parent'],event['proposal'])!=(pid,sid):
                    raise ValueError('Depth decision and original proposal differ')
                accepted[method]=event['accepted']
            offers.append(dict(parent=pid,proposal=sid,clipped_iou=clipped,accepted=accepted))
        matches={}
        for method in METHODS[1:]:
            p=assigned[method].get(rid)
            matches[method]=(None if p is None else dict(id=p['id'],parent=p.get('parent_id'),
                proposal=p.get('single_wall_proposal'),side=p.get('single_wall_side')))
        omitted=omitted_parent(target,raw,mask_by_id,parents,supports,settings) if not offers else None
        results.append(dict(reference=rid,raw_ious=raw,offered=offers,matched=matches,best_excluded_cut=omitted))
    return results


def summarize(targets):
    viable=[[o for o in t['offered'] if o['clipped_iou']>=PROTOCOL['iou']] for t in targets]
    info=dict(targets=len(targets),any_matching_sam_offered=sum(bool(t['offered']) for t in targets),
              still_matches_after_parent_clip=sum(bool(opts) for opts in viable),
              best_excluded_cut=Counter(t['best_excluded_cut']['reason'] for t in targets if t['best_excluded_cut']),methods={})
    for method in METHODS[1:]:
        matched=sum(t['matched'][method] is not None for t in targets)
        accepted_before=sum(any(o['accepted'][method] for o in t['offered']) for t in targets)
        accepted_after=sum(any(o['accepted'][method] for o in opts) for opts in viable)
        direct=sum(1 for t in targets if (part:=t['matched'][method]) and part['side']=='inside'
                   and part['proposal'] in t['raw_ious'])
        remainder=sum(1 for t in targets if (part:=t['matched'][method]) and part['side']=='remainder')
        info['methods'][method]=dict(any_matching_sam_accepted=accepted_before,
            clipped_matching_sam_accepted=accepted_after,found=matched,found_as_matching_sam_piece=direct,
            found_as_remainder=remainder,found_other=matched-direct-remainder)
    return info


@image_only()
def write_diagnostic(source,out_file):
    source,out_file=Path(source),Path(out_file)
    manifest=read(source/'manifest.json')
    hashes=manifest['artifacts']
    checked(source/'cases.json',hashes['cases.json'])
    checked(ARCHIVE,manifest['reference_sha256'])
    rows=read(source/'cases.json')
    prior=REVIEWS/'recovery-summary-01'
    prior_sha=read(prior/'manifest.json')['artifacts']['cases.json']
    checked(prior/'cases.json',prior_sha)
    old={r['key']:r for r in read(prior/'cases.json')}
    groups_by_split,_,fixtures=groups()
    lookup={f'{c.floor}/{c.id}':c for group in groups_by_split.values() for c in group}
    if fixtures!=manifest['fixtures'] or set(old)!=set(lookup) or set(lookup)!={r['key'] for r in rows}:
        raise ValueError('Frame IDs or fixture sets changed')
    sam=REVIEWS/'mobile-regions-01/manifest.json'
    model_files=read(sam)['artifacts']
    pictures=[]
    for row in rows:
        key=row['key']
        if old[key]['sha256']!=row['sha256'] or old[key]['remaining']['in_filtered_sam']!=row['opportunities']:
            raise ValueError('The fixed diagnostic targets changed')
        targets=trace(lookup[key],row,source,hashes,model_files,manifest['settings']) if row['opportunities'] else []
        pictures.append(dict(frame=key,sha256=row['sha256'],cohort=row['cohort'],targets=targets))
    results={group:summarize([t for r in pictures if group in ('all',r['cohort']) for t in r['targets']])
             for group in ('all','development','previously_reviewed_regression')}
    out_file.parent.mkdir(parents=True,exist_ok=True)
    save(out_file,dict(title='What happened to the matching MobileSAM outlines in the 41 baseline misses?',
        diagnostic='Recorded shapes identify candidates shared with each target at IoU >= 0.5. Per-target counts allow any matching filtered mask. A candidate must retain enough of the target when cut to a parent to qualify for the clipped count. A surface can also be recovered as a remainder of a different cut.',
        source=source.name,cases_sha256=hashes['cases.json'],control_cases_sha256=prior_sha,
        sam_manifest_sha256=digest(sam),reference_sha256=digest(ARCHIVE),summary=results,pictures=pictures))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=REVIEWS/'single-wall-depth-01')
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'opportunity-trace-{uuid4().hex[:12]}.json')
    args=parser.parse_args()
    print(write_diagnostic(args.source,args.out))
