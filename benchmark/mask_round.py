"""Compare the four region methods on frozen images. No game runtime is used."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import __version__ as pillow_version

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_metrics import PROTOCOL, aggregate, measure
from benchmark.normal_fields import cached_fields, fields as generate_fields
from benchmark.normal_assets import SOURCES
from benchmark.normal_round import groups
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.surface_round import artifact_path, image_of
from engine.surface_fields import MODELS, SETTINGS as FIELD_SETTINGS
from runtime.mode import image_only


REFERENCE = ROOT/'benchmark/reviews/normals-01'
METHODS = ('current-b','mask-first','scalelsd','deeplsd')
RULE = ('Use development layouts equally. Promote mask-first over B for this surface comparison if '
        'mean layout F1 or correctly assigned eligible pixel fraction improves by at least 2 points, '
        'the other falls by no more than 2 points, and opening fill increases by at most 1 point. '
        'Each line alternative must meet the same rule against mask-first. Among passing methods '
        'take highest mean layout F1 then pixel accuracy; prefer mask-first on ties. '
        'Ordinary builds also require demonstrated correct metric block placement.')


def summarize(rows):
    return {m:aggregate([r['scores'][m] for r in rows]) for m in rows[0]['scores']}


def select(dev):
    layouts = sorted({r['layout'] for r in dev})
    by_layout = [summarize([r for r in dev if r['layout']==layout]) for layout in layouts]
    ratings = {m:{k:float(np.mean([table[m][k] for table in by_layout if table[m][k] is not None]))
                  for k in ('f1','assigned_fraction','opening_fill')} for m in by_layout[0]}
    def qualifies(m, control):
        a,b = ratings[m],ratings[control]
        df,dp = a['f1']-b['f1'],a['assigned_fraction']-b['assigned_fraction']
        return max(df,dp)>=.02 and min(df,dp)>=-.02 and a['opening_fill']<=b['opening_fill']+.01
    candidates = ['current-b']
    if qualifies('mask-first','current-b'):
        candidates.append('mask-first')
    for m in METHODS[2:]:
        if m in ratings and qualifies(m,'mask-first') and qualifies(m,'current-b'):
            candidates.append(m)
    return dict(rule=RULE,ratings=ratings,selected=max(candidates,key=lambda m:(ratings[m]['f1'],ratings[m]['assigned_fraction'])))


def get_fields(image_path, sha, cache, generate):
    maps=cached_fields(sha,cache)
    if maps is not None:
        return maps['normal'],maps['depth'],'cached'
    if not generate:
        raise ValueError(f'Missing compatible numerical maps for {sha} in {cache}; use --generate-fields to regenerate')
    a,b=generate_fields(image_path,sha,cache)
    return a,b,'fresh'


def load_lines(directory, method, sha, size, declaration):
    path = directory/method/(sha+'.json.gz')
    if digest(path)!=declaration['files'].get(sha):
        raise ValueError(f'Line evidence digest mismatch for {method}/{sha}')
    with gzip.open(path,'rt',encoding='utf8') as file:
        evidence = json.load(file)
    if evidence['image_sha256']!=sha or list(size)!=evidence['image_size']:
        raise ValueError(f'Line data belongs to a different frame: {path}')
    return evidence


def predict(case, row, args, model_manifests):
    from engine.mask_first import regions
    from engine.surface_model import proposals_for
    source=case.path('image')
    sha=case.frame['image_sha256']
    if digest(source)!=sha:
        raise ValueError('Source image hash differs')
    image=image_of(case)
    start=perf_counter()
    normals,depth,field_source=get_fields(source,sha,args.fields,args.generate_fields)
    timing={'model_fields_'+field_source:perf_counter()-start}
    with gzip.open(REFERENCE/row['prediction'],'rt',encoding='utf8') as f:
        b=json.load(f)['normal-regions']['model']['planes']
    start=perf_counter()
    replay=proposals_for(image,'26.3','normal-regions',normals)
    timing['current-b']=perf_counter()-start
    same_b=len(b)==len(replay) and all(np.allclose(x['quad'],y['quad'],atol=.01) and x['support']==y['support']
                                       for x,y in zip(b,replay))
    if not same_b:
        raise ValueError(f'Current B replay differs from the frozen prediction for {case.id}')
    start=perf_counter()
    parts=regions(image,normals,depth,version='26.3')
    timing['mask-first']=perf_counter()-start
    predictions={'current-b':b,'mask-first':parts}
    if args.phase=='full':
        from engine.mask_line_refine import refine
        for method in METHODS[2:]:
            evidence=load_lines(args.lines,method,sha,image.size,model_manifests[method])
            start=perf_counter()
            predictions[method]=refine(image,parts,normals,depth,evidence)
            timing[method+'_refine']=perf_counter()-start
            timing[method+'_inference_recorded']=evidence['inference_seconds']
    return image,predictions,timing,same_b


def exported_blocks(out, data, cohorts, selected):
    from benchmark.mask_blocks import evaluate_case
    cases = {f'{case.floor}/{case.id}':case for group in cohorts.values() for case in group}
    with ZipFile(out/'block-packs.zip','x',compression=ZIP_DEFLATED) as archive:
        for row in data:
            case = cases[row['key']]
            with gzip.open(out/row['predictions'],'rt',encoding='utf8') as f:
                predictions = json.load(f)
            with TemporaryDirectory(dir=out) as temp:
                directory = Path(temp)
                report = evaluate_case(case,image_of(case),predictions['current-b'],predictions[selected],
                                       directory,selected_uses_masks=selected!='current-b')
                for file in sorted(directory.iterdir()):
                    archive.write(file,row['key']+'/'+file.name)
            row['camera_scale_status'] = report['camera_status']
            row['blocks'] = {}
            for name in ('baseline','selected','selected_with_B_charts'):
                method = report[name]
                row['blocks'][name] = dict(score=method['score'],regions=method['regions'])
                if 'known_camera_and_plane_diagnostic' in method:
                    row['blocks'][name]['diagnostic'] = method['known_camera_and_plane_diagnostic']['score']
    def combine(rows):
        totals = {}
        for name,fields in rows[0]['blocks'].items():
            totals[name] = {}
            for field in fields:
                values = Counter()
                for row in rows:
                    values.update(row['blocks'][name][field])
                totals[name][field] = dict(values)
        return totals
    save(out/'blocks.json',dict(selected=selected,all=combine(data),
         development=combine([r for r in data if r['cohort']=='development']),
         previously_reviewed_regression=combine([r for r in data if r['cohort']!='development']),
         camera_scale_status=dict(Counter(r['camera_scale_status'] for r in data)),
         note='Correct visible coordinates after decoding exported pack NBT. In-game placement was not run.'))


def run(args):
    from engine.mask_first import SETTINGS as region_settings
    cohorts,suite,fixture_hashes=groups()
    if len(set(c.frame['image_sha256'] for group in cohorts.values() for c in group))!=52 or sum(map(len,cohorts.values()))!=56:
        raise ValueError('Expected 56 pinned frames with 52 distinct PNG digests')
    model_manifests={m:read(args.lines/m/'manifest.json') for m in METHODS[2:]} if args.phase=='full' else {}
    prior={r['key']:r for r in read(REFERENCE/'cases.json')}
    code=revision()
    out=args.out.resolve()
    out.mkdir(exist_ok=False,parents=True)
    data=[]
    for cohort,cases in cohorts.items():
        if args.phase=='first' and cohort!='development':
            continue
        for case in cases:
            key=f'{case.floor}/{case.id}'
            row=prior[key]
            image,predictions,timing,same_b=predict(case,row,args,model_manifests)
            # Persist every picture's predictions before using any recorded surface as a scoring target.
            file=out/'predictions'/(key+'.json.gz')
            file.parent.mkdir(parents=True,exist_ok=True)
            file.write_bytes(gzip.compress(json.dumps(predictions,separators=(',',':'),allow_nan=False).encode(),mtime=0))
            _,reference=recorded(case)
            scores={}
            for method,parts in predictions.items():
                scores[method],_=measure(parts,reference,use_mask=method!='current-b')
            for k in ('matched','eligible','proposed'):
                if scores['current-b'][k]!=row['methods']['normal-regions']['outlines'][k]:
                    raise ValueError(f'Published B counts differ: {key} {k}')
            data.append(dict(key=key,id=case.id,sha256=row['sha256'],layout=row['layout'],floor=case.floor,
                             cohort='development' if cohort=='development' else 'previously_reviewed_regression',
                             historical_cohort=cohort,source=artifact_path(case.path('image'),out),
                             predictions=artifact_path(file,out),timing_seconds=timing,baseline_replay_agrees=same_b,
                             line_changed_region_pixels={m:sum(p.get('line_changed_pixels',0) for p in predictions[m])//2
                                                         for m in model_manifests},scores=scores))
        print(f'{cohort}: {len(cases)} images',flush=True)
        if cohort=='development':
            decision=select(data)
            save(out/'selection.json',decision)
    if args.phase=='full':
        exported_blocks(out,data,cohorts,decision['selected'])
    save(out/'cases.json',data)
    grouped={k:summarize([r for r in data if r['layout']==k]) for k in sorted({r['layout'] for r in data})}
    summary=dict(all=summarize(data),layouts=grouped,development=summarize([r for r in data if r['cohort']=='development']))
    rest=[r for r in data if r['cohort']!='development']
    if rest:
        summary['previously_reviewed_regression']=summarize(rest)
    save(out/'scores.json',summary)
    methods=list(data[0]['scores'])
    baseline=read(REFERENCE/'manifest.json')
    manifest=dict(run_id=out.name,phase=args.phase,code=code,version=suite['version'],fixtures=fixture_hashes,
                  client_sha256=baseline['client_sha256'],capture=baseline['capture'],baseline_revision=baseline['code'],
                  reference_sha256=digest(ARCHIVE),previous_b_run_sha256=digest(REFERENCE/'manifest.json'),
                  shared_models={k:dict(sha256=v[2],pinned_source=SOURCES[k][1]) for k,v in MODELS.items()},
                  settings=dict(mask_first=region_settings,metrics=PROTOCOL,shared_inference=FIELD_SETTINGS),
                  split={k:sorted({r['layout'] for r in data if r['cohort']==k}) for k in {r['cohort'] for r in data}},
                  split_history='Only shopfront and wing used for tuning. All six other layouts have appeared in earlier reports.',
                  methods=methods,selection=decision,baseline_replays_matching=sum(r['baseline_replay_agrees'] for r in data),
                  timings_seconds={k:sum(r['timing_seconds'].get(k,0.) for r in data) for k in set().union(*(r['timing_seconds'] for r in data))},
                  frames=[dict(key=r['key'],sha256=r['sha256']) for r in data],
                  environment=dict(numpy=np.__version__,pillow=pillow_version,python=platform.python_version(),platform=platform.platform()),
                  artifacts=dict(scores='scores.json',per_picture='cases.json',predictions='predictions/',selection='selection.json',
                                 sheet='comparison.jpg'))
    if args.phase=='full':
        from benchmark.mask_blocks import SETTINGS as block_settings
        from engine.mask_line_refine import SETTINGS as line_settings
        manifest['settings']['line_refinement']=line_settings
        manifest['settings']['block_export']=block_settings
        manifest['line_models']=model_manifests
        unique={r['sha256']:r for r in data}.values()
        manifest['unique_line_inference_seconds']={m:sum(r['timing_seconds'][m+'_inference_recorded'] for r in unique)
                                                   for m in model_manifests}
        subprocess.run([sys.executable,'-m','benchmark.mask_loss_trace','--fields',str(args.fields.resolve()),
                        '--out',str(out/'loss-trace')],cwd=ROOT,check=True)
        manifest['artifacts'].update(report='report.html',block_scores='blocks.json',exported_packs='block-packs.zip',
                                     loss_trace='loss-trace/manifest.json',loss_sheet='loss-trace/stages.jpg')
    from benchmark.mask_sheet import render
    render(out,data,methods)
    if args.phase=='full':
        from benchmark.mask_report import render_report
        render_report(out)
    manifest['numerical_cache_sha256']={r['sha256']:digest(args.fields/(r['sha256']+'.npz'))
                                         for r in {v['sha256']:v for v in data}.values()}
    manifest['artifact_sha256']={f.relative_to(out).as_posix():digest(f) for f in sorted(out.rglob('*')) if f.is_file()}
    save(out/'manifest.json',manifest)
    print(out,flush=True)


@image_only()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['first','full'],default='full')
    parser.add_argument('--fields',type=Path,default=ROOT/'runs/normal-fields')
    parser.add_argument('--generate-fields',action='store_true')
    parser.add_argument('--lines',type=Path,default=ROOT/'benchmark/data/mask-lines-v1')
    parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args())


if __name__=='__main__':
    main()
