"""Score native outline adjustments on the shipped Surface Finder's frozen proposals."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.crop_fit_round import BASE, BASE_SHA
from benchmark.ledge_join_round import score_dimensions, summary
from benchmark.mask_metrics import measure, PROTOCOL
from benchmark.normal_round import groups
from benchmark.split_round import score_regions
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.verify_default_strip import APPROVED, SIGNATURES, region_signature
from benchmark.wall_join_test import checked, read_gzip
from engine.strip_neighbor import clean_surfaces
from engine.surface_perimeter import _native_alpha
from runtime.mode import image_only, mark_image_only


def edge_points(mask):
    # Mask pixels with a 4-neighbor outside, including borders around interior holes.
    pad = np.pad(mask, 1)
    interior = mask & pad[:-2,1:-1] & pad[2:,1:-1] & pad[1:-1,:-2] & pad[1:-1,2:]
    return np.argwhere(mask & ~interior)


def distance_sum(source, target):
    """Exact nearest Euclidean distance sum, discarding distant targets by triangle inequality."""
    tiles = source // 24
    keys = tiles[:,0] * (tiles[:,1].max()+1) + tiles[:,1]
    order = keys.argsort()
    total = 0.
    for group in np.split(order, np.flatnonzero(np.diff(keys[order]))+1):
        points = source[group]
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = np.linalg.norm(points-center, axis=1).max()
        dist = np.linalg.norm(target-center, axis=1)
        near = target[dist <= dist.min()+2*radius+1e-8]
        diff = points[:,None,:] - near
        total += np.sqrt(np.einsum('ijk,ijk->ij',diff,diff).min(axis=1)).sum()
    return float(total)


def support(part, size):
    canvas = np.zeros(size[::-1],bool)
    region = _native_alpha(part['support'], size)
    if region:
        alpha,(l,t,r,b) = region
        canvas[t:b,l:r] = np.asarray(alpha)!=0
    return canvas


def native_metrics(proposals, native, pairs, size):
    totals = {m:Counter() for m in proposals}
    by_surface = []
    for idx, rid in pairs:
        truth = native == rid
        base = support(proposals['current'][idx], size)
        reference_edges = edge_points(truth)
        original = None
        methods = {}
        for name, parts in proposals.items():
            mask = base if name=='current' else support(parts[idx],size)
            if original is not None and np.array_equal(mask,base):
                rec = dict(original)
            else:
                edges = edge_points(mask)
                if not len(edges):
                    raise ValueError('Detail refinement erased a matched surface')
                added, removed = mask & ~base, base & ~mask
                rec = dict(added_correct=int((added & truth).sum()), added_wrong=int((added & ~truth).sum()),
                           removed_correct=int((removed & truth).sum()), removed_wrong=int((removed & ~truth).sum()),
                           intersection=int((mask & truth).sum()), union=int((mask | truth).sum()),
                           edge_sum=distance_sum(edges,reference_edges)+distance_sum(reference_edges,edges),
                           edge_n=len(edges)+len(reference_edges))
            methods[name] = rec
            totals[name].update(rec)
            if name=='current':
                original = rec
        by_surface.append(dict(reference=rid, proposal=proposals['current'][idx]['id'],methods=methods))
    return by_surface,totals


def evaluate(job):
    case,row,expected,source_sha,signature,methods,out = job
    with Image.open(checked(case.path('image'),row['sha256'])) as src:
        photo = src.convert('RGB')
    raw = read_gzip(checked(BASE/row['predictions'],source_sha))['gap']
    started = perf_counter()
    base,_ = clean_surfaces(photo,raw)
    timing = {'current':perf_counter()-started}
    if region_signature(base)!=signature:
        raise ValueError('Shipped baseline changed')
    proposals = {'current':base}
    for name in methods:
        module,setting = name.split(':',1)
        predict = importlib.import_module('benchmark.'+module).predict
        started = perf_counter()
        refined = predict(photo,deepcopy(base),row['sha256'],setting)
        timing[name] = perf_counter()-started
        if [p['id'] for p in refined]!=[p['id'] for p in base]:
            raise ValueError('Refinements must retain all incoming proposal IDs in order')
        proposals[name] = refined
    path = 'predictions/'+row['key']+'.json.gz'
    dest = out/path
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(gzip.compress(json.dumps(proposals,separators=(',',':'),allow_nan=False).encode(),compresslevel=3,mtime=0))
    scores, scopes = score_regions(case,proposals,row['key'])
    rec = dict(key=row['key'],sha256=row['sha256'],cohort=row['cohort'],layout=row['layout'],predictions=path,
               scopes=scopes,scores=scores,cached_processing_s=timing)
    score_dimensions(case,expected,proposals,rec,'neighbor')
    for field in ('scores','sizes'):
        if rec[field]['current']!=expected[field]['neighbor']:
            raise ValueError('Shipped baseline mismatch: '+row['key'])
    pose,ref = recorded(case)
    native = _native_reference(pose,ref,read(case.path('labels'))['faces'])
    _,pairs = measure(base,ref,use_mask=True)
    rec['paired_surfaces'],rec['native'] = native_metrics(proposals,native,pairs,photo.size)
    return rec


def totals(rows,methods):
    keys = ['all',*sorted({r['cohort'] for r in rows}),*sorted({r['layout'] for r in rows})]
    table = {}
    for group in keys:
        subset = [r for r in rows if group in ('all',r['cohort'],r['layout'])]
        table[group] = summary(subset,methods)
        for m in methods:
            count = Counter()
            for row in subset:
                count.update(row['native'][m])
            sums = dict(count)
            sums.update(edge_mean_px=count['edge_sum']/count['edge_n'],pixel_iou=count['intersection']/count['union'])
            table[group][m]['native'] = sums
            table[group][m]['size_changes'] = {key:sum(len(r['size_changes'][m][key]) for r in subset)
                                               for key in ('gained','lost')}
    return table


def provenance(method, selected):
    module, setting = method.split(':', 1)
    candidate = importlib.import_module('benchmark.'+module)
    data = dict(candidate.provenance(setting)) if hasattr(candidate, 'provenance') else dict(settings=candidate.SETTINGS[setting])
    fields = {}
    used = {r['sha256'] for r in selected}
    for name in data.pop('field_manifests', []):
        path = ROOT / name
        evidence = read(path)
        times = evidence.get('timing', {})
        total = Counter()
        if isinstance(times, dict):
            for sha in used:
                if isinstance(times.get(sha), dict):
                    total.update(times[sha])
        fields[name] = dict(sha256=digest(path), fields_reused=len(used & times.keys()) if isinstance(times,dict) else None,
                            preprocessing_seconds=dict(total), details={k:evidence[k] for k in
                            ('release','checkpoint','upstream','moge','utils3d','device','settings','torch','segmentation','load_seconds') if k in evidence})
    return dict(**data, field_manifests=fields)


@image_only()
def run(out,workers,split,methods,sheet=False):
    source = read(checked(BASE/'manifest.json',BASE_SHA))
    gold = read(APPROVED/'manifest.json')
    for key in ('fixtures','frames','splits','protocol','reference_sha256','version'):
        if source[key]!=gold[key]:
            raise ValueError('Frozen upstream inputs differ: '+key)
    rows = read(checked(BASE/'cases.json',source['artifacts']['cases.json']))
    expected = {r['key']:r for r in read(checked(APPROVED/'cases.json',gold['artifacts']['cases.json']))}
    signatures = read(SIGNATURES)['frames']
    checked(ARCHIVE,source['reference_sha256'])
    cohorts,suite,fixtures = groups()
    captures = {c.floor+'/'+c.id:c for group in cohorts.values() for c in group}
    frame_hashes = {r['key']:r['sha256'] for r in source['frames']}
    if frame_hashes!={r['key']:r['sha256'] for r in rows} or frame_hashes!={k:r['sha256'] for k,r in expected.items()}:
        raise ValueError('Source frame list differs from its manifest')
    if fixtures!=source['fixtures'] or set(captures)!=set(expected) or len(rows)!=56 or source['protocol']!=PROTOCOL:
        raise ValueError('Frozen inputs changed')
    selected = [r for r in rows if split in ('all',r['cohort'])]
    if not selected:
        raise ValueError('Empty fixture split')
    out.mkdir(parents=True,exist_ok=False)
    jobs = [(captures[r['key']],r,expected[r['key']],source['artifacts'][r['predictions']],signatures[r['key']],methods,out)
            for r in selected]
    code = revision()
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        results = list(pool.map(evaluate,jobs))
    labels = ['current',*methods]
    scores = totals(results,labels)
    save(out/'cases.json',results)
    save(out/'scores.json',scores)
    if sheet:
        if 'edge_detail:native4-stable-counts' in methods:
            from benchmark.detail_count_sheet import render_count_review
            render_count_review(out)
        else:
            from benchmark.detail_sheet import render
            render(out)
    manifest = dict(run_id=out.name,code=code,frames=[dict(key=r['key'],sha256=r['sha256'],cohort=r['cohort'],layout=r['layout']) for r in selected],
        fixtures=fixtures,version=suite['version'],splits=source['splits'],selection=split,methods=labels,workers=workers,
        reference_sha256=source['reference_sha256'],capture=source['capture'],protocol=PROTOCOL,
        baseline_sources={'benchmark/reviews/sky-gap-01/manifest.json':BASE_SHA,
                          'benchmark/reviews/strip-neighbor-01/manifest.json':digest(APPROVED/'manifest.json')},
        settings={m:provenance(m, selected) for m in methods},
        candidate_sha256={m:digest(ROOT/'benchmark'/(m.split(':')[0]+'.py')) for m in methods},
        native_metric='Fixed shipped matched IDs, all native pixel centers along 4-connected inner boundary including holes; pooled bidirectional Euclidean distance. Add/remove assessed per proposal against its fixed reference; unmatched proposals excluded from paired metric.',
        timing='CPU wall seconds per picture. Current starts from saved sky-filtered proposals; candidates start from shipped baseline. Any precomputed candidate inference reported separately.',
        reproduce='python -m benchmark.detail_round --split '+split+' '+ ' '.join('--method '+m for m in methods)+' --out runs/detail-replay'+(' --sheet' if sheet else ''),
        artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()})
    save(out/'manifest.json',manifest)
    for m in labels:
        sample = scores['all'][m]
        print(m,sample['matched'],sample['sizes'],sample['native'],flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--split',choices=['all','development','previously_reviewed_regression'],default='all')
    cli.add_argument('--workers',type=int,default=4)
    cli.add_argument('--sheet',action='store_true')
    cli.add_argument('--method',action='append',default=[])
    args=cli.parse_args()
    run(args.out.resolve(),args.workers,args.split,args.method,args.sheet)
