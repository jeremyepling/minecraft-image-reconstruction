"""Grade the published Wall Separator's remaining misses, using recorded geometry only for evaluation."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from benchmark.camera_round import ROOT, read, save, digest
from benchmark.mask_metrics import measure, footprint
from benchmark.normal_round import groups
from benchmark.split_round import load_predictions
from benchmark.surface_reference import ARCHIVE, recorded
from runtime.mode import image_only


METHOD = 'wall-depth-anything'
DEFAULT_ARCHIVE = ROOT/'benchmark/reviews/separation-01'
HISTORICAL = ROOT/'benchmark/reviews/overhang-01'


def kind(patch):
    return {-1:'underside',0:'vertical',1:'top'}[int(patch['normal'][1])]


def verified(root,path,expected):
    if digest(root/path)!=expected:
        raise ValueError(f'Changed source: {root/path}')
    return read(root/path)


def summary(rows):
    return dict(n=len(rows),orientation=dict(Counter(r['orientation'] for r in rows)),
                causes=dict(Counter(r['category'] for r in rows)),
                absorbed=sum(bool(r['absorbed_by']) for r in rows),
                parallel_step=sum(any(n['parallel_at_other_depth'] for q in r['absorbed_by']
                                      for n in q['also_covers']) for r in rows),
                any_claim=sum(r['union_coverage']>0 for r in rows),
                half_claimed=sum(r['union_coverage']>=.5 for r in rows),
                one_half_cover=sum(r['largest_coverage']>=.5 for r in rows),
                fragmented=sum(r['pure_fragments']>=2 for r in rows),
                near_40_to_50=sum(.4<=r['best_iou']<.5 for r in rows),
                near_45_to_50=sum(.45<=r['best_iou']<.5 for r in rows),
                close_49_to_50=sum(.49<=r['best_iou']<.5 for r in rows),
                any_saved_method_found=sum(bool(r['found_by_other_method']) for r in rows),
                can_join_two_unmatched=sum(r['best_unmatched_pair_iou']>=.5 for r in rows),
                other_methods=dict(Counter(m for r in rows for m in r['found_by_other_method'])))


@image_only()
def run(out):
    source=DEFAULT_ARCHIVE
    manifest=read(source/'manifest.json')
    historical=read(HISTORICAL/'manifest.json')
    legacy=verified(HISTORICAL,'results.json',historical['artifacts']['results.json'])
    saved=verified(source,'cases.json',manifest['artifacts']['cases.json'])
    groups_by_split,_,fixtures=groups()
    lookup={f'{c.floor}/{c.id}':c for group in groups_by_split.values() for c in group}
    if (fixtures!=manifest['fixtures'] or fixtures!=historical['fixtures'] or
        digest(ARCHIVE)!=manifest['reference_sha256'] or digest(ARCHIVE)!=legacy['reference_sha256'] or
        set(lookup)!={r['key'] for r in saved} or set(lookup)!=set(legacy['views'])):
        raise ValueError('Fixture, reference or frame mismatch')
    totals=Counter()
    rows=[]
    cases=[]
    for r in saved:
        key=r['key']
        cap=lookup[key]
        old=legacy['views'][key]
        if not r['sha256']==old['sha256']==digest(cap.path('image')):
            raise ValueError(key)
        _,ref=recorded(cap)
        patches=ref['patches']
        oldrefs={o['id']:o for o in old['references']}
        path=r['predictions']
        parts=load_predictions(source,path,manifest['artifacts'][path])[METHOD]
        score,pairs=measure(parts,ref,use_mask=True)
        matched={rid:i for i,rid in pairs}
        if score!=r['scores'][METHOD] or set(matched)!=set(r['scopes'][METHOD]['matched']):
            raise ValueError(f'Saved Wall Separator scores or match IDs disagree for {key}')
        score_ids=ref['id_map']
        masks=[footprint(p,score_ids.shape,ref['image_size'],True) for p in parts]
        areas=np.bincount(score_ids.ravel(),minlength=max(patches)+1)
        hits=np.stack([np.bincount(score_ids[m],minlength=len(areas)) for m in masks])
        sizes=hits.sum(axis=1)
        covered=np.logical_or.reduce(masks)
        covered_counts=np.bincount(score_ids[covered],minlength=len(areas))
        frame_misses=[]
        for pid,patch in patches.items():
            if not patch['eligible']:
                continue
            totals[kind(patch)]+=1
            if pid in matched:
                continue
            intersections=hits[:,pid]
            area=areas[pid]
            ious=intersections/(sizes+area-intersections)
            cover=intersections/area
            purity=intersections/np.maximum(sizes,1)
            best=int(ious.argmax())
            eligible_fragments=np.flatnonzero((cover>=.1)&(purity>=.7))
            absorption=[]
            for i in np.flatnonzero(cover>=.5):
                also=[]
                for other,p in patches.items():
                    if other==pid or not p['eligible'] or hits[i,other]<areas[other]/2:
                        continue
                    also.append(dict(id=other,coverage=hits[i,other]/areas[other],
                                     parallel_at_other_depth=bool(patch['normal']==p['normal'] and patch['offset']!=p['offset']),
                                     orientation=kind(p),offset=p['offset']))
                if also:
                    absorption.append(dict(proposal=parts[i]['id'],coverage=cover[i],also_covers=also))
            pair_iou=0.
            pair_ids=[]
            unmatched=[i for i in np.flatnonzero(intersections) if i not in matched.values()]
            target=score_ids==pid
            for i,j in combinations(unmatched,2):
                together=masks[i]|masks[j]
                overlap=np.count_nonzero(together & target)
                iou=overlap/(together.sum()+area-overlap)
                if iou>pair_iou:
                    pair_iou=iou
                    pair_ids=[parts[i]['id'],parts[j]['id']]
            other_methods={m for m,v in r['scopes'].items() if m!=METHOD and pid in v['matched']}
            other_methods.update(m for m,v in oldrefs[pid]['methods'].items() if v['proposal'] is not None)
            fraction=float(covered_counts[pid]/area)
            if absorption:
                category='merged with another scored face'
            elif len(eligible_fragments)>=2:
                category='split into two or more mostly pure pieces'
            elif fraction==0:
                category='no predicted pixels'
            elif fraction<.5:
                category='less than half covered by all guesses'
            elif cover.max()>=.5:
                category='single overextended or partial guess'
            else:
                category='coverage distributed across mixed guesses'
            row=dict(key=key,layout=r['layout'],split=r['cohort'],id=pid,orientation=kind(patch),
                     normal=patch['normal'],offset=patch['offset'],samples=int(area),pixels=oldrefs[pid]['pixels'],
                     category=category,union_coverage=fraction,largest_coverage=float(cover.max()),
                     best_iou=float(ious[best]),best_proposal=parts[best]['id'] if intersections.any() else None,
                     best_purity=float(purity[best]),best_coverage=float(cover[best]),absorbed_by=absorption,
                     pure_fragments=len(eligible_fragments),best_unmatched_pair_iou=float(pair_iou),
                     best_unmatched_pair=pair_ids,found_by_other_method=sorted(other_methods))
            rows.append(row)
            frame_misses.append(row)
        cases.append(dict(key=key,sha256=r['sha256'],eligible=score['eligible'],matched_ids=sorted(matched),
                          prediction_archive_sha256=manifest['artifacts'][path],summary=summary(frame_misses)))
    info=dict(method=METHOD,version=manifest['version'],fixtures=fixtures,reference_sha256=digest(ARCHIVE),
              source_manifest_sha256=digest(source/'manifest.json'),legacy_manifest_sha256=digest(HISTORICAL/'manifest.json'),
              frames=len(cases),eligible=dict(totals),matched=sum(len(c['matched_ids']) for c in cases),all=summary(rows),
              layouts={layout:summary([r for r in rows if r['layout']==layout]) for layout in sorted({r['layout'] for r in rows})},
              splits={split:summary([r for r in rows if r['split']==split]) for split in sorted({r['split'] for r in rows})},
              scoring='Eligible surfaces at stride four. One-to-one IoU >= 0.50. Absorbed = a single guess covers >=50% of each of two scored surfaces. Fragmented = at least two guesses each cover >=10% of the target, with >=70% purity. Near-threshold and absorption can overlap. Pair diagnostic looks for a union of two guesses currently unmatched to any regular scored surface; small-surface matches may be affected. Older surface matches are read from their versioned results and are a hindsight ceiling.',
              cases=cases)
    out.mkdir(parents=True,exist_ok=False)
    save(out/'summary.json',info)
    save(out/'misses.json',rows)
    return info


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    print(run(parser.parse_args().out)['all'])
