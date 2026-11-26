"""Compare three block counters on the identical frozen Surface Finder masks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.count_metrics import chart_spans, count_totals, reference_planes, select
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE, recorded
from engine.surface_dimensions import dimensions
from runtime.mode import image_only


FINDER=ROOT/'benchmark/reviews/mask-four-01'
METHODS=('legacy-repeat','masked-repeat','atlas-repeat')


def cached_regions(row, frozen):
    path=FINDER/row['predictions']
    if digest(path)!=frozen['artifact_sha256'][row['predictions']]:
        raise ValueError(f'Frozen predictions changed for {row["key"]}')
    with gzip.open(path,'rt',encoding='utf8') as stream:
        return json.load(stream)['mask-first']


def measure(image,region,atlas,times):
    from engine import atlas_count, legacy_count, masked_count
    from engine.surface_sample import sample_surface
    sample=sample_surface(image,region)
    quad=sample['quad'] if sample is not None else None
    results={}
    for name,implementation in zip(METHODS,(legacy_count,masked_count,atlas_count)):
        t=perf_counter()
        if sample is None:
            result=dict(counts=[None,None],phase=[None,None],reason='no sampling chart')
        elif name=='atlas-repeat':
            result=implementation.count(sample,atlas,version='26.3')
        else:
            result=implementation.count(sample,version='26.3')
        times[name]+=perf_counter()-t
        result=dict(result)
        result['dimensions']=dimensions(region,quad,image.size,result['counts'])
        results[name]=result
    return dict(id=region['id'],quad=np.asarray(quad).tolist() if quad is not None else None,
                chart_source=('finder' if region.get('quad') is not None else 'derived from mask') if quad is not None else None,
                measures=results)


def evaluate(case,row,atlas,frozen):
    with Image.open(case.path('image')) as src:
        image=Image.fromarray(np.asarray(src.convert('RGB')))
    if digest(case.path('image'))!=row['sha256']:
        raise ValueError('Image changed')
    regions=cached_regions(row,frozen)
    times=Counter()
    preds=[measure(image,region,atlas,times) for region in regions]
    # Predict every region before reading its reference ID, material or pose.
    _,ref=recorded(case)
    faces=read(case.path('labels'))['faces']
    planes=reference_planes(ref,faces)
    matches={m['proposal']:m['recorded_surface'] for m in row['matched_regions']['mask-first']}
    for rec in preds:
        pid=matches.get(rec['id'])
        rec.update(recorded_surface=pid,expected=planes[pid]['size'] if pid is not None else None,
                   expected_chart=chart_spans(ref['charts'][pid],rec['quad']) if pid is not None else None)
    diag=[]
    for rectangle in ref['rectangles']:
        rec=measure(image,dict(id=rectangle['patch_id'],quad=rectangle['quad']),atlas,Counter())
        target=chart_spans(ref['charts'][rectangle['patch_id']],rec['quad'] or rectangle['quad'])
        rec.update(expected=target,expected_chart=target)
        diag.append(rec)
    return dict(key=row['key'],id=case.id,layout=row['layout'],cohort=row['cohort'],floor=case.floor,
                sha256=row['sha256'],eligible=row['scores']['mask-first']['eligible'],surfaces=preds,
                known_rectangles=diag,method_seconds=dict(times))


def summary(cases):
    return {m:dict(surface=count_totals(cases,m),known_rectangles=count_totals(cases,m,True)) for m in METHODS}


def run(args):
    from benchmark.count_materials import evaluate_materials
    from benchmark.count_review import review
    from benchmark.count_extent_diagnostic import evaluate as check_reference_extents
    from engine import atlas_count, masked_count, surface_pitch, surface_sample
    from engine.atlas_count import load_atlas
    atlas=load_atlas(args.atlas,version='26.3')
    frozen=read(FINDER/'manifest.json')
    cohorts,suite,fixtures=groups()
    if fixtures!=frozen['fixtures'] or digest(ARCHIVE)!=frozen['reference_sha256'] or atlas.provenance['client_sha256']!=suite['client_sha256']:
        raise ValueError('Input fixtures differ from the published Surface Finder experiment')
    original=read(FINDER/'cases.json')
    if digest(FINDER/'cases.json')!=frozen['artifact_sha256']['cases.json']:
        raise ValueError('Frozen surface assignments changed')
    lookup={f'{c.floor}/{c.id}':c for group in cohorts.values() for c in group}
    rows=[r for r in original if not args.dev_only or r['cohort']=='development']
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    stamp=revision()
    results=[]
    for cohort in dict.fromkeys(r['cohort'] for r in rows):
        for row in [r for r in rows if r['cohort']==cohort]:
            results.append(evaluate(lookup[row['key']],row,atlas,frozen))
            if len(results)%8==0:
                print(f"{len(results)}/{len(rows)} pictures measured",flush=True)
        if cohort=='development':
            decision=select(results,METHODS)
            save(out/'selection.json',decision)
        print(cohort, len(results),'pictures processed',flush=True)
    save(out/'cases.json',results)
    stats={'all':summary(results)}
    for key in ('cohort','layout','floor'):
        stats[key]={v:summary([r for r in results if r[key]==v]) for v in sorted({r[key] for r in results})}
    save(out/'scores.json',stats)
    if not args.skip_material_check:
        evaluate_materials(results,rows,lookup,atlas,frozen,decision['selected'],out)
    if not args.dev_only:
        from benchmark.count_export import export_counts
        export_counts(results,rows,lookup,out,decision['selected'])
        check_reference_extents(lookup.values(),out/'extent-diagnostic.json')
    review(out,results,rows,lookup,atlas,frozen,decision['selected'])
    code_files=['engine/surface_sample.py','engine/legacy_count.py','engine/masked_count.py','engine/atlas_count.py',
                'engine/surface_pitch.py','engine/surface_dimensions.py','engine/surface_tiles.py','benchmark/count_metrics.py',
                'benchmark/count_materials.py','benchmark/count_export.py','benchmark/count_extent_diagnostic.py','benchmark/count_round.py']
    manifest=dict(run_id=out.name,code=stamp,version=suite['version'],surface_finder='mask-first',
        frozen_finder_manifest_sha256=digest(FINDER/'manifest.json'),shared_models=frozen['shared_models'],
        client_sha256=frozen['client_sha256'],capture=frozen['capture'],fixtures=fixtures,reference_sha256=digest(ARCHIVE),
        atlas=atlas.provenance,methods=list(METHODS),selection=decision,methods_source_sha256={f:digest(ROOT/f) for f in code_files},
        stage_names=['Surface Finder','Surface Straightener','Block Counter','Material Matcher'],
        settings={'straightener':surface_sample.SETTINGS,'legacy-repeat':surface_pitch.SETTINGS,
                  'masked-repeat':masked_count.SETTINGS,'atlas-repeat':atlas_count.SETTINGS,
                  'material_matcher':'rank_tile, vanilla 26.3 atlas, exact ID and top five',
                  'models_this_replay':'frozen Surface Finder predictions; zero model inference calls',
                  'counter_timing':'wall seconds on predicted sampling charts only; excludes common straightening and evaluation',
                  'reference_surface_match':'one-to-one, mask intersection over union >= 0.5',
                  'material_sampling':'phase on both axes, full mask support, at least 4 native pixels per edge'},
        reproduce=r'.\Invoke-BlockCountRound.ps1',
        scoring='whole visible mask span rounded to a block, compared to integer extents of recorded visible cells; exact in both axes. Front-on diagnostic on true planar rectangles uses half-block tolerance.',
        split='20 development images (shopfront and wing); 36 previously reviewed regression views from six other buildings',
        frames=[{k:r[k] for k in ('key','sha256','cohort','layout')} for r in rows],
        method_seconds={m:sum(r['method_seconds'][m] for r in results) for m in METHODS},
        artifacts={p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()})
    save(out/'manifest.json',manifest)
    print(out,flush=True)


@image_only()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/('count-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:6]))
    parser.add_argument('--dev-only',action='store_true')
    parser.add_argument('--skip-material-check',action='store_true',help='For development smoke tests')
    args=parser.parse_args()
    if args.skip_material_check and not args.dev_only:
        parser.error('--skip-material-check is available only on --dev-only runs')
    run(args)


if __name__=='__main__':
    main()
