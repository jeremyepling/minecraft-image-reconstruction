"""Try one additional surface-finding step on the frozen Wall Separator output."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_metrics import PROTOCOL
from benchmark.mask_round import load_lines
from benchmark.normal_fields import cached_fields, fields
from benchmark.normal_round import groups
from benchmark.split_round import LINES, improves, load_predictions, pooled, save_and_score
from benchmark.split_tiny_lines import ARCHIVE as TINY
from benchmark.surface_reference import ARCHIVE
from runtime.mode import image_only


PREVIOUS = ROOT/'benchmark/reviews/separation-01'
DEFAULT = 'wall-depth-anything'
JUNCTION = 'junction-cuts'
COMBINED = 'wall-and-underside'
METHODS = ('current', JUNCTION, COMBINED)
DISPLAY = {'current':'Wall Separator',JUNCTION:'Junction cuts',COMBINED:'Wall + underside'}
RULE = ('Compare each addition independently to the frozen Wall Separator. Choose the highest '
        'layout-average F1 among development improvements, then confirm on regression. An '
        'improvement needs more true matches and a higher pooled F1 without losing underside count, '
        'with at most 0.003 additional opening fill and 0.002 lower assigned pixel fraction. '
        'Report each trade-off, including border error, unmatched proposals and downstream counts.')


def predict_and_score(job):
    from engine.surface_junction_split import refine as junctions
    from engine.surface_strip_split import refine as strips
    case,row,checks,tiny,line_manifest,cache,out = job
    sha = row['sha256']
    if digest(case.path('image'))!=sha or sha!=tiny['image_sha256']:
        raise ValueError(f"Screenshot mismatch for {row['key']}")
    with Image.open(case.path('image')) as source:
        image = source.convert('RGB')
    if tiny['image_size']!=list(image.size):
        raise ValueError('M-LSD segments do not use the source coordinate system')
    walls = load_predictions(PREVIOUS,row['predictions'],checks['previous'])[DEFAULT]
    maps = cached_fields(sha,cache)
    if maps is None:
        normal,depth = fields(case.path('image'),sha,out.parent/(out.name+'-base-fields'))
    else:
        normal,depth = maps['normal'],maps['depth']
    if hashlib.sha256(depth.tobytes()).hexdigest()!=checks['depth']:
        raise ValueError('Depth field differs from the frozen wall experiment')
    lines = {'scalelsd':load_lines(LINES,'scalelsd',sha,image.size,line_manifest)}
    start = perf_counter()
    split = junctions(image,deepcopy(walls),lines,normal,depth)
    wall_time = perf_counter()-start
    start = perf_counter()
    combined = strips(image,deepcopy(walls),{'mlsd':tiny},normal,depth)
    timings = {JUNCTION:wall_time,COMBINED:perf_counter()-start}
    predictions = {'current':walls,JUNCTION:split,COMBINED:combined}
    result = save_and_score(case,row,predictions,timings,out,DEFAULT)
    if result['scopes']['current']['matched']!=row['scopes'][DEFAULT]['matched']:
        raise ValueError(f"Frozen surface identities changed for {row['key']}")
    return result


@image_only()
def run(out,cache,workers):
    from benchmark.remaining_surface_audit import run as audit_misses
    from benchmark.split_count_check import run as check_counts
    from benchmark.split_sheet import render
    from engine.surface_junction_split import SETTINGS as junction_settings
    from engine.surface_strip_split import SETTINGS as strip_settings
    previous = read(PREVIOUS/'manifest.json')
    if digest(PREVIOUS/'cases.json')!=previous['artifacts']['cases.json']:
        raise ValueError('Frozen separation results changed')
    cohorts,suite,fixtures = groups()
    if fixtures!=previous['fixtures'] or digest(ARCHIVE)!=previous['reference_sha256']:
        raise ValueError('Fixture or evaluation geometry differs from the previous run')
    lookup = {f'{c.floor}/{c.id}':c for cohort in cohorts.values() for c in cohort}
    rows = read(PREVIOUS/'cases.json')
    if len(rows)!=56 or set(r['key'] for r in rows)!=set(lookup):
        raise ValueError('Expected all 56 frozen views')
    with gzip.open(TINY,'rt',encoding='utf8') as stream:
        tiny=json.load(stream)
    line_manifest = read(LINES/'scalelsd/manifest.json')
    if digest(TINY)!=previous['line_input_sha256']['mlsd'] or digest(LINES/'scalelsd/manifest.json')!=previous['line_input_sha256']['scalelsd']:
        raise ValueError('Saved line inputs have changed')
    perimeter = ROOT/'benchmark/reviews/perimeters-01'
    if digest(perimeter/'manifest.json')!=previous['source_manifests']['perimeters-01']:
        raise ValueError('Depth source manifest has changed')
    if digest(perimeter/'cases.json')!=read(perimeter/'manifest.json')['artifacts']['cases.json']:
        raise ValueError('Depth field identifiers have changed')
    depths = {r['key']:r['numerical_depth_sha256'] for r in read(perimeter/'cases.json')}
    out.mkdir(parents=True,exist_ok=False)
    settings = dict(junction=junction_settings,underside=strip_settings,score=PROTOCOL,rule=RULE)
    save(out/'protocol.json',settings)
    stamp = revision()
    all_results,summary = [],{}
    selected = {'wall':JUNCTION,'strip':COMBINED,'display':DISPLAY}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for group in ('development','regression'):
            jobs=[]
            for row in rows:
                if (row['cohort']=='development')!=(group=='development'):
                    continue
                key = row['key']
                checks = {'previous':previous['artifacts'][row['predictions']], 'depth':depths[key]}
                jobs.append((lookup[key],row,checks,tiny['frames'][row['sha256']],line_manifest,cache,out))
            results=list(pool.map(predict_and_score,jobs))
            table={m:pooled(results,m) for m in METHODS}
            summary[group]=table
            all_results.extend(results)
            if group=='development':
                passing = [m for m in METHODS[1:] if improves(table,m)]
                candidate = max(passing,key=lambda m:table[m]['layout_mean_f1'], default='current')
                save(out/'development-selection.json',dict(**selected,choice=candidate,rule=RULE,scores=table))
            print(f'{group}: {len(results)} pictures',flush=True)
    selected['recommended'] = candidate if improves(summary['regression'],candidate) else 'current'
    summary['all']={m:pooled(all_results,m) for m in METHODS}
    summary['buildings']={k:{m:pooled([r for r in all_results if r['layout']==k],m) for m in METHODS}
                          for k in sorted({r['layout'] for r in all_results})}
    save(out/'cases.json',all_results)
    save(out/'scores.json',summary)
    audit_misses(out/'remaining')
    render(out,all_results,lookup,selected)
    save(out/'manifest.json',dict(run_id=out.name,code=stamp,version=suite['version'],fixtures=fixtures,
        source_manifests={PREVIOUS.name:digest(PREVIOUS/'manifest.json')},reference_sha256=digest(ARCHIVE),
        capture=previous['capture'],splits=previous['splits'],models=previous['models'],
        line_models={n:previous['line_models'][n] for n in ('mlsd','scalelsd')},line_input_sha256=previous['line_input_sha256'],
        frames=[{k:r[k] for k in ('key','layout','cohort','sha256')} for r in all_results],
        methods=DISPLAY,control=DEFAULT,selected=selected,settings=settings,per_picture='cases.json',scores='scores.json',
        sheet='comparison.jpg',count_check='counts/manifest.json',remaining_misses='remaining/summary.json',
        runtime=dict(workers=workers,inference='frozen fields and line maps when available',
            cached_refinement_seconds={m:sum(r['cached_processing_s'][m] for r in all_results) for m in METHODS[1:]}),
        artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()},
        reproduce=r'.\Invoke-SurfaceRound.ps1 -Round recovery'))
    check_counts(out,out/'counts')
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'recovery-{uuid4().hex[:12]}')
    parser.add_argument('--fields',type=Path,default=ROOT.parent/'normal-round/runs/normal-fields')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    run(args.out.resolve(),args.fields,args.workers)
