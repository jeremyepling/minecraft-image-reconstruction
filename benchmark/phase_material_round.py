"""Measure exact block IDs from phase-only atlas registration on visible image-derived faces."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.block_seam_round import POOL
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_materials import true_faces
from benchmark.count_metrics import reference_planes
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.grid_placement_check import corner_error, CORNER_TOLERANCE
from benchmark.known_spacing_atlas import predict
from benchmark.mask_metrics import measure
from benchmark.normal_round import groups
from benchmark.phase_material_match import create_matcher, match_known, predict_ids
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.textures import match_cells
from benchmark.verify_default_strip import SIGNATURES, region_signature
from benchmark.wall_join_test import checked, read_gzip
from engine.masked_count import count
from engine.strip_neighbor import clean_surfaces
from engine.surface_sample import sample_surface
from runtime.mode import image_only, mark_image_only


REVIEW = ROOT/'benchmark/reviews/atlas-phase-materials-01'
METHODS = ('shape','atlas')
ARMS = ('ordinary','restricted')
RULE = ('Primary endpoint is correctly located visible faces with exact top-1 block ID from the production '
        'Texture matcher. Compare gained and lost faces and wrong IDs across development, regression, '
        'and separate families. Require fresh-family evidence before promoting a default.')


@lru_cache(maxsize=2)
def matcher(method):
    return create_matcher(method)


def predictions(image,parts,frozen):
    samples = [sample_surface(image,p) for p in parts]
    ordinary = [count(s) for s in samples]
    if frozen is not None:
        for p,s,o,f in zip(parts,samples,ordinary,frozen,strict=True):
            if (p['id']!=f['id'] or (s['quad'] if s else None)!=f['quad'] or
                    any(o[k]!=f['baseline'][k] for k in ('counts','phase'))):
                raise ValueError('Frozen rectification or counter changed')
    restricted = [predict(s,o) for s,o in zip(samples,ordinary)]
    changes = []
    for p,o,r in zip(parts,ordinary,restricted):
        if (r['counts']!=o['counts'] or any(a is not None and a!=b for a,b in zip(o['phase'],r['phase']))
                or (None in o['counts'] and r['phase']!=o['phase'])):
            raise ValueError('Atlas changed a measured quantity')
        if r['phase']!=o['phase']:
            changes.append(dict(proposal=p['id'],counts=o['counts'],before=o['phase'],after=r['phase']))
    results = {}
    for method in METHODS:
        results[method] = {arm:predict_ids(image,parts,samples,readings,matcher(method))
                           for arm,readings in zip(ARMS,(ordinary,restricted))}
    return results,changes


def link(rows,paired,valid,ref):
    indices = defaultdict(list)
    for i,p in enumerate(rows):
        indices[p['proposal']].append(i)
    associated = {}
    for proposal,rid in paired.items():
        eligible = {c for (p,c) in valid if p==rid}
        shapes = [rows[i]['quad'] for i in indices[proposal]]
        by_quad = {id(q):i for q,i in zip(shapes,indices[proposal])}
        inv = np.linalg.inv(ref['charts'][rid])
        for cell,(quad,iou,_) in match_cells(shapes,eligible,inv)[0].items():
            associated[by_quad[id(quad)]] = ((rid,cell),iou,corner_error(quad,cell,inv))
    return associated


def grade(rows,links,valid):
    scores = Counter(eligible=len(valid),proposed=len(rows),named=0,aligned=0,correct=0,wrong_id=0,
                     unresolved_aligned=0,unpaired=0,misaligned=0,unverified_named=0)
    graded = []
    for i,r in enumerate(rows):
        block_id = r['block_id']
        scores['named'] += int(block_id is not None)
        rec = dict(proposal=r['proposal'],cell=r['cell'],quad=r['quad'],guess=block_id,
                   group=r['block_index'],source_face=r['material_source_face'])
        if i in links:
            k,iou,error = links[i]
            f = valid[k]
            actual = 'minecraft:'+f['block_id'].removeprefix('minecraft:')
            rec.update(reference=[k[0],*k[1]],xyz=f['xyz'],direction=f['direction'],expected=actual,
                       iou=iou,max_corner_error=error,aligned=error<=CORNER_TOLERANCE)
            if rec['aligned']:
                scores['aligned'] += 1
                scores['unresolved_aligned' if block_id is None else 'correct' if block_id==actual else 'wrong_id'] += 1
            else:
                scores['misaligned'] += 1
                scores['unverified_named'] += int(block_id is not None)
        else:
            scores['unpaired'] += 1
            scores['unverified_named'] += int(block_id is not None)
        graded.append(rec)
    return dict(scores),graded


def evaluate(key,sha,layout,cohort,image,parts,load_reference,frozen=None):
    raw,changes = predictions(image,parts,frozen)
    ref,faces = load_reference()
    planes = reference_planes(ref,faces)
    valid = true_faces(planes,image.size)
    _,pairs = measure(parts,ref,use_mask=True)
    paired = {parts[i]['id']:rid for i,rid in pairs}
    # Independent of material labels and predictions. Also test exact corners for every paired face.
    def order(k):
        f = valid[k]
        return hashlib.sha256(str((sha,f['xyz'],f['direction'])).encode()).digest()
    audit = set(sorted(valid,key=order)[:8])
    links = {a:link(raw['shape'][a],paired,valid,ref) for a in ARMS}
    known_keys = audit.union(*(v[0] for lookup in links.values() for v in lookup.values()))
    results = {}
    for method,arms in raw.items():
        known = {}
        for k in sorted(known_keys):
            face = valid[k]
            known[k] = match_known(image,face['image_corners'],matcher(method))
        stats,records = {},{}
        for a,rows in arms.items():
            if [p['quad'] for p in rows]!=[p['quad'] for p in raw['shape'][a]]:
                raise ValueError('Material comparator changed the geometry')
            stats[a],records[a] = grade(rows,links[a],valid)
            for rec in records[a]:
                if 'reference' in rec:
                    rid,x,y = rec['reference']
                    rec['known_corner_id'] = known[(rid,(x,y))]
        def aligned_keys(a):
            return {tuple(p['reference']):p for p in records[a] if p.get('aligned')}
        before,after = map(aligned_keys,ARMS)
        correct = [{k for k,p in d.items() if p['guess']==p['expected']} for d in (before,after)]
        new = set(after)-set(before)
        comparison = dict(gained_correct=len(correct[1]-correct[0]),lost_correct=len(correct[0]-correct[1]),
            new_aligned=len(new),new_correct=sum(after[k]['guess']==after[k]['expected'] for k in new),
            new_known_correct=sum(after[k]['known_corner_id']==after[k]['expected'] for k in new),
            lost_aligned=len(set(before)-set(after)),known_sample=len(audit),
            known_sample_correct=sum(known[k]=='minecraft:'+valid[k]['block_id'].removeprefix('minecraft:') for k in audit))
        results[method] = dict(scores=stats,comparison=comparison,faces=records)
    return dict(key=key,sha256=sha,layout=layout,cohort=cohort,proposals=len(parts),matched=len(paired),
                outline_signature=region_signature(parts),eligible_faces=len(valid),phase_changes=changes,methods=results)


def fixed(job):
    cap,row,pred_sha,signature,old = job
    with Image.open(checked(cap.path('image'),row['sha256'])) as src:
        image = src.convert('RGB')
    upstream = read_gzip(checked(BASE/row['predictions'],pred_sha))['gap']
    parts,_ = clean_surfaces(image,upstream)
    if region_signature(parts)!=signature:
        raise ValueError('Different outlines')
    def reference():
        return recorded(cap)[1],read(cap.path('labels'))['faces']
    return evaluate(row['key'],row['sha256'],row['layout'],old['cohort'],image,parts,reference,old['surfaces'])


def fixed_cases(workers):
    source = read(checked(BASE/'manifest.json',BASE_SHA))
    pool = read(POOL/'manifest.json')
    checked(ARCHIVE,pool['reference_sha256'])
    sig = read(checked(SIGNATURES,pool['signatures_sha256']))['frames']
    cc,_,fixtures = groups()
    captures = {c.floor+'/'+c.id:c for group in cc.values() for c in group}
    old = read(checked(POOL/'candidates.json',pool['files']['candidates.json']))
    rows = read(checked(BASE/'cases.json',source['artifacts']['cases.json']))
    if pool['fixtures']!=fixtures or len(rows)!=56 or source['frames']!=pool['frames']:
        raise ValueError('Fixed suite changed')
    jobs = []
    for r,f in zip(rows,old,strict=True):
        if (r['key'],r['sha256'])!=(f['key'],f['sha256']):
            raise ValueError('Mismatched frame')
        jobs.append((captures[r['key']],r,source['artifacts'][r['predictions']],sig[r['key']],f))
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as executor:
        return list(executor.map(fixed,jobs))


def totals(frames):
    out = {}
    for m in METHODS:
        counters = {a:Counter() for a in (*ARMS,'comparison')}
        for f in frames:
            for a in ARMS:
                counters[a].update(f['methods'][m]['scores'][a])
            counters['comparison'].update(f['methods'][m]['comparison'])
        out[m] = counters
    return out


@image_only()
def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    if args.suite=='fixed':
        frames = fixed_cases(args.workers)
        inputs = dict(pool_manifest=digest(POOL/'manifest.json'),reference_sha256=digest(ARCHIVE),
                      outline_signatures_sha256=digest(SIGNATURES))
    else:
        from benchmark.phase_material_families import iter_family
        frames = [evaluate(k,sha,args.suite,args.suite,image,parts,ref)
                  for k,sha,image,parts,ref in iter_family(args.suite,args.shard,args.surfaces)]
        inputs = dict(shard_sha256=digest(args.shard),surfaces_sha256=digest(args.surfaces))
    save(out/'cases.json',frames)
    scopes = dict(all=frames)
    for f in frames:
        for scope in {f['cohort'],f['layout']}:
            scopes.setdefault(scope,[]).append(f)
    save(out/'scores.json',{s:totals(rows) for s,rows in scopes.items()})
    save(out/'manifest.json',dict(code=revision(),suite=args.suite,inputs=inputs,rule=RULE,
         frames=[{k:f[k] for k in ('key','sha256','cohort','layout','outline_signature')} for f in frames],
         matchers={m:matcher(m).provenance for m in METHODS},reference_sample_per_frame=8,
         artifacts={n:digest(out/n) for n in ('scores.json','cases.json')}))


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--suite',choices=['fixed','cloister1','twinspire1'],default='fixed')
    cli.add_argument('--shard',type=Path)
    cli.add_argument('--surfaces',type=Path)
    cli.add_argument('--workers',type=int,default=4)
    args = cli.parse_args()
    if args.suite!='fixed' and (not args.shard or not args.surfaces):
        cli.error('Family suites need --shard and --surfaces')
    run(args)
