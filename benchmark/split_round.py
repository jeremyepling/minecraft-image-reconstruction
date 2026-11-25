"""Isolate wall separation, underside recovery, and their evidence sources on saved frames."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_metrics import PROTOCOL, aggregate, footprint, measure
from benchmark.mask_round import load_lines
from benchmark.normal_fields import cached_fields, fields
from benchmark.normal_round import groups
from benchmark.overhang_matches import score_parts
from benchmark.split_depth import prepare as prepare_point_depth
from benchmark.split_tiny_lines import ARCHIVE as TINY_LINES
from benchmark.surface_detection_maps import _native_reference
from benchmark.surface_reference import ARCHIVE, recorded
from runtime.mode import image_only


BASE = ROOT/'benchmark/reviews/perimeters-01'
OLD = ROOT/'benchmark/reviews/mask-four-01'
LINES = ROOT/'benchmark/data/mask-lines-v1'
WALL = ('wall-depth-anything','wall-moge')
STRIP = ('strip-deeplsd','strip-scalelsd','strip-both','strip-mlsd')
METHODS = ('current',*WALL,*STRIP)
NAMES = dict(zip(METHODS,('Previous contrast-aligned finder','Wall Separator + Depth Anything V2',
    'Wall Separator + MoGe-2 depth','Underside Finder + DeepLSD',
    'Underside Finder + ScaleLSD v2','Underside Finder + DeepLSD and ScaleLSD','Underside Finder + M-LSD Tiny')))
RULE = ('Choose each input on development, preferring a higher layout-average F1 for walls, '
        'and more undersides for strips if F1 stays within 0.01 of the control. '
        'Ties prefer fewer unpaired proposals and the first listed input. For an ordinary '
        'surface default require more matches and higher F1 on both development and regression, '
        'no fewer undersides on either, opening fill increasing at most 0.003, and correctly '
        'assigned pixel fraction dropping at most 0.002 on either. Keep comparably scored '
        'experimental methods opt-in until their ordinary image entry point is verified.')


def load_predictions(directory, name, expected):
    path = directory/name
    if digest(path) != expected:
        raise ValueError(f'Archived prediction changed: {path}')
    with gzip.open(path,'rt',encoding='utf8') as stream:
        return json.load(stream)


def fragmented(parts, ref):
    """Two distinct, mostly pure guesses each claim at least 10% of one scored surface."""
    ids = ref['id_map']
    areas = np.bincount(ids.ravel(),minlength=max(ref['patches'])+1)
    pieces = np.zeros(len(areas),int)
    for p in parts:
        mask = footprint(p,ids.shape,ref['image_size'],True)
        hits = np.bincount(ids[mask],minlength=len(areas))
        pieces += (hits>0) & (hits>=.1*areas) & (hits>=.7*hits.sum())
    return [i for i,patch in ref['patches'].items() if patch['eligible'] and pieces[i]>=2]


def evaluate(job):
    from engine.surface_wall_split import refine as walls
    from engine.surface_strip_split import refine as strips
    case, row, old, checks, manifests, tiny, cache, points, out = job
    key, sha = row['key'],row['sha256']
    if digest(case.path('image'))!=sha or sha!=old['sha256']:
        raise ValueError(f'Mismatched source for {key}')
    with Image.open(case.path('image')) as src:
        image = src.convert('RGB')
    current = load_predictions(BASE,row['predictions'],checks[0])['color-edges']
    rectangles = load_predictions(OLD,old['predictions'],checks[1])['current-b']
    maps = cached_fields(sha,cache)
    if maps is None:
        private_cache = out.parent/(out.name+'-base-fields')
        maps = dict(zip(('normal','depth'),fields(case.path('image'),sha,private_cache)))
    if hashlib.sha256(maps['depth'].tobytes()).hexdigest()!=row['numerical_depth_sha256']:
        raise ValueError(f'Depth Anything field differs from archived prediction: {key}')
    lines = {m:load_lines(LINES,m,sha,image.size,declaration) for m,declaration in manifests.items()}
    if tiny['image_sha256']!=sha or tiny['image_size']!=list(image.size):
        raise ValueError(f'M-LSD input mismatch: {key}')
    lines['mlsd'] = tiny
    alternate = np.load(points/(sha+'.npy'),allow_pickle=False)
    predictions = {'current': current}
    timings = {}
    for method, depth in zip(WALL,(maps['depth'],alternate)):
        start = perf_counter()
        predictions[method] = walls(image,deepcopy(current),depth,rectangles)
        timings[method] = perf_counter()-start
    for method, names in zip(STRIP,(('deeplsd',),('scalelsd',),('deeplsd','scalelsd'),('mlsd',))):
        start = perf_counter()
        predictions[method] = strips(image,deepcopy(current),{n:lines[n] for n in names},maps['normal'],maps['depth'])
        timings[method] = perf_counter()-start
    return save_and_score(case,row,predictions,timings,out,'color-edges')


def save_and_score(case,row,predictions,timings,out,baseline_method):
    """Persist predictions before using the scorer, with 'current' as the shared control."""
    key = row['key']
    for method,parts in predictions.items():
        if len({p['id'] for p in parts})!=len(parts):
            raise ValueError(f'Duplicate region ID for {key}/{method}')
    rel = 'predictions/'+key+'.json.gz'
    path = out/rel
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(predictions,allow_nan=False,separators=(',',':')).encode(),mtime=0))
    # Scoring begins after every independent prediction is on disk.
    pose,ref = recorded(case)
    native = _native_reference(pose,ref,read(case.path('labels'))['faces'])
    pixels = np.bincount(native.ravel(),minlength=max(ref['patches'])+1)
    refs = [dict(id=i,normal=p['normal'],eligible=p['eligible'],samples=p['support_px'],
                 pixels=int(pixels[i]),methods={}) for i,p in ref['patches'].items()]
    scores, detail = {},{}
    for method,parts in predictions.items():
        scores[method], pairs = measure(parts,ref,use_mask=True)
        matched = score_parts(parts,method,ref,native,refs)
        if matched!={rid:parts[i]['id'] for i,rid in pairs}:
            raise ValueError(f'Independent region matching disagrees on {key}/{method}')
        eligible = set(matched)
        small = [r['id'] for r in refs if not r['eligible'] and r['methods'][method]['proposal'] is not None]
        down = {r['id'] for r in refs if r['eligible'] and r['normal'][1]==-1}
        vertical = {r['id'] for r in refs if r['eligible'] and r['normal'][1]==0}
        split = fragmented(parts,ref)
        base = set(detail['current']['matched']) if method!='current' else eligible
        base_split = set(detail['current']['fragmented']) if method!='current' else set(split)
        detail[method] = dict(matched=sorted(eligible),gained=sorted(eligible-base),lost=sorted(base-eligible),
            undersides=sorted(eligible & down),underside_total=len(down),vertical=sorted(eligible & vertical),
            vertical_total=len(vertical),small=small,unpaired=len(parts)-len(eligible)-len(small),
            fragmented=split,new_fragmentation=sorted((set(split)-base_split)&base))
    if scores['current']!=row['scores'][baseline_method]:
        raise ValueError(f'Frozen baseline mismatch for {key}')
    return dict(key=key,sha256=row['sha256'],layout=row['layout'],cohort=row['cohort'],predictions=rel,
                scopes=detail,scores=scores,cached_processing_s=timings)


def pooled(rows, method):
    result = aggregate([r['scores'][method] for r in rows])
    details = [r['scopes'][method] for r in rows]
    for name in ('gained','lost','undersides','vertical','small','fragmented','new_fragmentation'):
        result[name] = sum(len(d[name]) for d in details)
    for name in ('unpaired','underside_total','vertical_total'):
        result[name] = sum(d[name] for d in details)
    result['layout_mean_f1'] = float(np.mean([aggregate([r['scores'][method] for r in rows if r['layout']==k])['f1']
                                              for k in sorted({r['layout'] for r in rows})]))
    return result


def choose(dev):
    def quality(method):
        x = dev[method]
        return x['layout_mean_f1'], -x['unpaired']
    def underside(method):
        x = dev[method]
        return x['layout_mean_f1']>=dev['current']['layout_mean_f1']-.01, x['undersides'], quality(method)
    return dict(wall=max(WALL,key=quality),strip=max(STRIP,key=underside))


def improves(table, method):
    x, base = table[method],table['current']
    return (x['matched']>base['matched'] and x['f1']>base['f1'] and x['undersides']>=base['undersides']
            and x['opening_fill']<=base['opening_fill']+.003 and x['assigned_fraction']>=base['assigned_fraction']-.002)


@image_only()
def run(args):
    from engine.surface_wall_split import SETTINGS as wall_settings
    from engine.surface_strip_split import SETTINGS as strip_settings
    from benchmark.split_sheet import render
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    stamp = revision()
    manifests = {m:read(LINES/m/'manifest.json') for m in ('deeplsd','scalelsd')}
    with gzip.open(TINY_LINES,'rt',encoding='utf8') as stream:
        tiny = json.load(stream)
    cohorts, suite, fixtures = groups()
    lookup = {f'{c.floor}/{c.id}':c for group in cohorts.values() for c in group}
    baseline, old = read(BASE/'manifest.json'),read(OLD/'manifest.json')
    if not (fixtures==baseline['fixtures']==old['fixtures']==tiny['fixtures'] and
            digest(ARCHIVE)==baseline['reference_sha256']==old['reference_sha256']):
        raise ValueError('Baseline fixtures or reference geometry differ')
    for directory, hashes in ((BASE,baseline['artifacts']),(OLD,old['artifact_sha256'])):
        if digest(directory/'cases.json')!=hashes['cases.json']:
            raise ValueError(f'Changed case roster for {directory}')
    rows = read(BASE/'cases.json')
    prior = {r['key']:r for r in read(OLD/'cases.json')}
    if len(rows)!=56 or len(lookup)!=56 or set(prior)!=set(lookup) or {r['key'] for r in rows}!=set(lookup):
        raise ValueError('Expected the frozen 56 frames')
    depth_index = prepare_point_depth(args.point_cache)
    settings = dict(walls=wall_settings,strips=strip_settings,scorer=PROTOCOL,rule=RULE)
    save(out/'protocol.json',settings)
    results,summary = [],{}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for name, is_dev in (('development',True),('regression',False)):
            jobs = []
            for r in rows:
                if (r['cohort']=='development')!=is_dev:
                    continue
                key = r['key']
                original = prior[key]
                checks = baseline['artifacts'][r['predictions']],old['artifact_sha256'][original['predictions']]
                jobs.append((lookup[key],r,original,checks,manifests,tiny['frames'][r['sha256']],args.fields,args.point_cache,out))
            split = list(executor.map(evaluate,jobs))
            results.extend(split)
            table = {m:pooled(split,m) for m in METHODS}
            summary[name] = table
            if is_dev:
                selected = choose(table)
                save(out/'development-selection.json',dict(**selected,rule=RULE,scores=table))
            print(f'{name}: {len(split)} frames',flush=True)
    summary['all'] = {m:pooled(results,m) for m in METHODS}
    summary['buildings'] = {k:{m:pooled([r for r in results if r['layout']==k],m) for m in METHODS}
                            for k in sorted({r['layout'] for r in results})}
    candidates = [m for m in selected.values() if all(improves(summary[s],m) for s in ('development','regression'))]
    selected['recommended'] = max(candidates,key=lambda m:summary['development'][m]['layout_mean_f1'],default='current')
    save(out/'cases.json',results)
    save(out/'scores.json',summary)
    render(out,results,lookup,selected)
    lines = {m:{k:v for k,v in declaration.items() if k not in ('files','timings','runtime_seconds')}
             for m,declaration in manifests.items()}
    lines['mlsd'] = {k:v for k,v in tiny.items() if k!='frames'}
    manifest = dict(run_id=out.name,code=stamp,version=suite['version'],fixtures=fixtures,
        source_manifests={d.name:digest(d/'manifest.json') for d in (BASE,OLD)},reference_sha256=digest(ARCHIVE),
        splits='20 development pictures of Shopfront and Wing; 36 regression pictures of six previously reviewed layouts.',
        capture=baseline['capture'],frames=[{k:r[k] for k in ('key','layout','cohort','sha256')} for r in results],
        models=baseline['models'],line_models=lines,line_input_sha256=dict(mlsd=digest(TINY_LINES),
            **{m:digest(LINES/m/'manifest.json') for m in manifests}),
        alternate_depth=depth_index,methods=NAMES,settings=settings,selected=selected,
        runtime=dict(cached_processing_cpu_wall_seconds={m:sum(r['cached_processing_s'][m] for r in results) for m in METHODS[1:]},
        point_head_inference_and_conversion_seconds=sum(r['inference_and_conversion_s'] for r in depth_index['fields'].values()),
        note='Per-method time summed across pictures under concurrent load. Original raw line, normal and Depth Anything inference was reused.',
        workers=args.workers), scores='scores.json',per_picture='cases.json',sheet='comparison.jpg',
        artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()},
        reproduce=r'.\Invoke-SurfaceRound.ps1 -Round separation')
    save(out/'manifest.json',manifest)
    from benchmark.split_count_check import run as check_counts
    check_counts(out,out/'counts')
    print(out,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'separation-{uuid4().hex[:12]}')
    parser.add_argument('--fields',type=Path,default=ROOT.parent/'normal-round/runs/normal-fields')
    parser.add_argument('--point-cache',type=Path,default=ROOT/'runs/moge-point-depth')
    parser.add_argument('--workers',type=int,default=3)
    run(parser.parse_args())
