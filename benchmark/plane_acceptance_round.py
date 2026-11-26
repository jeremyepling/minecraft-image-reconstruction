"""Compare two ways to accept additional plane outlines on the frozen single-image suite."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from benchmark.afterpass_round import COHORTS, FIELDS, picture_table, validate_fields
from benchmark.afterpass_sheet import plane_overlay
from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.ledge_join_round import score_dimensions, summary
from benchmark.ledge_join_sheet import render_comparison
from benchmark.mask_metrics import PROTOCOL, aggregate, measure
from benchmark.normal_round import groups
from benchmark.planarity_afterpass import raw_diagnostic
from benchmark.split_round import save_and_score
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.wall_join_test import SAM, checked, read_gzip
from engine import plane_inset, plane_consensus
from runtime.mode import image_only, mark_image_only


BASE = ROOT/'benchmark/reviews/afterpass-models-01'
PLANES = FIELDS/'planarity'
SAM_ROOT = BASE.parent/SAM
NAMES = dict(current='513-surface baseline', inset='Single inset', consensus='Two-model agreement')
METHODS = tuple(NAMES)
RULE = ('On development, prefer settings that preserve surface F1 and opening fill, and retain correct '
        'surface pixel assignment within 0.2 percentage points. Rank eligible settings by surfaces found, '
        'then F1, then correct pixels. When no setting qualifies, report the best available result as experimental.')


def capture_inputs(row, capture, hashes):
    with Image.open(checked(capture.path('image'),row['sha256'])) as photo:
        image = photo.convert('RGB')
    parts = read_gzip(checked(BASE/row['predictions'], hashes['previous']))['planarity']
    sha = row['sha256']
    with np.load(checked(PLANES/'fields'/(sha+'.npz'), hashes['planes']), allow_pickle=False) as data:
        fields = dict(data)
    prefix = SAM_ROOT/'proposals'/sha
    with np.load(checked(prefix.with_suffix('.npz'), hashes['sam_masks']), allow_pickle=False) as data:
        masks = np.unpackbits(data['packed'],axis=2,count=int(data['shape'][2])).astype(bool)
    records = read_gzip(checked(prefix.with_suffix('.json.gz'),hashes['sam_records']))['candidates']
    if (tuple(fields['image_size']) != image.size or masks.shape[1:] != fields['relaxed'].shape or
            len(records) != len(masks)):
        raise ValueError('Misaligned model outputs')
    return image, parts, fields, masks, records


def predict(name, preset, inputs):
    image, parts, fields, masks, records = inputs
    if name == 'inset':
        return plane_inset.predict(image,parts,fields,preset)
    return plane_consensus.predict(image,parts,fields,preset,sam_masks=masks,sam_records=records)


def development(job):
    row, capture, hashes = job
    inputs = capture_inputs(row,capture,hashes)
    scores = {}
    _, ref = recorded(capture)
    for name, module in (('inset',plane_inset),('consensus',plane_consensus)):
        scores[name] = {}
        for preset in module.PRESETS:
            parts, _ = predict(name,preset,inputs)
            scores[name][preset], _ = measure(parts,ref,use_mask=True)
    return scores


def qualifies(s, baseline):
    return (s['f1'] >= baseline['f1'] and s['opening_fill'] <= baseline['opening_fill']
            and s['assigned_fraction'] >= baseline['assigned_fraction']-.002)


def freeze(dev_scores, dev_rows):
    baseline = aggregate([r['scores']['planarity'] for r in dev_rows])
    selected, comparisons = {}, {}
    for name in METHODS[1:]:
        results = {k:aggregate([d[name][k] for d in dev_scores]) for k in dev_scores[0][name]}
        def rank(k):
            s = results[k]
            return qualifies(s,baseline), s['matched'], s['f1'], s['correct_pixels']
        selected[name] = max(results,key=rank)
        comparisons[name] = results
    return dict(rule=RULE,baseline=baseline,selected=selected,development=comparisons,
                presets={'inset':plane_inset.PRESETS,'consensus':plane_consensus.PRESETS},
                frames=[r['key'] for r in dev_rows])


def evaluate(job):
    row, capture, hashes, selected, out = job
    inputs = capture_inputs(row,capture,hashes)
    predictions, elapsed, events = {'current':inputs[1]}, {'current':0.}, {}
    for name,preset in selected.items():
        start = perf_counter()
        predictions[name], events[name] = predict(name,preset,inputs)
        elapsed[name] = perf_counter()-start
    result = save_and_score(capture,row,predictions,elapsed,out,'planarity')
    score_dimensions(capture,row,predictions,result,baseline_method='planarity')
    result['events'] = events
    return result


def sheet(out, rows, captures, table, source=None):
    def raw(put, photo, row, box, prefix, focus=None):
        field = PLANES/'fields'/(row['sha256']+'.npz')
        with np.load(field,allow_pickle=False) as data:
            labels = data['relaxed']
        caption = 'Each color marks one model region'
        if focus is not None:
            _, ref = recorded(captures[row['key']])
            best, = raw_diagnostic(labels,ref,[focus])
            caption = f"Best raw outline for R{focus}: {best['best_iou']:.0%} overlap"
        put(prefix+'planes',plane_overlay(photo,labels).crop(box),'Raw Planarity regions',caption)
        return {'planes_sha256':digest(field),'channel':'relaxed'}
    return render_comparison(out,rows,captures,table,NAMES,'Which missing surfaces can we recover from the existing outlines?',
        raw_stages=[('planes','Raw plane proposals','Model regions before either acceptance check.')],
        raw_panels=raw,source=source)


@image_only()
def run(out, workers):
    base = read(BASE/'manifest.json')
    rows = read(checked(BASE/'cases.json',base['artifacts']['cases.json']))
    checked(ARCHIVE,base['reference_sha256'])
    cohorts, suite, fixtures = groups()
    captures = {f'{c.floor}/{c.id}':c for group in cohorts.values() for c in group}
    roster = {r['key']:r['sha256'] for r in rows}
    if len(rows)!=56 or set(roster)!=set(captures) or fixtures!=base['fixtures']:
        raise ValueError('Frozen fixture mismatch')
    checked(PLANES/'manifest.json', base['candidates']['planarity']['fields_manifest_sha256'])
    masks_meta = validate_fields(PLANES, fixtures, roster)
    sam_meta = read(SAM_ROOT/'manifest.json')
    if sam_meta['fixtures']!=fixtures or sam_meta['reference_sha256']!=base['reference_sha256']:
        raise ValueError('MobileSAM and baseline fixtures differ')
    jobs = []
    for row in rows:
        sha = row['sha256']
        hashes = dict(previous=base['artifacts'][row['predictions']],
                      planes=masks_meta['artifacts'][f'fields/{sha}.npz'],
                      sam_masks=sam_meta['artifacts'][f'proposals/{sha}.npz'],
                      sam_records=sam_meta['artifacts'][f'proposals/{sha}.json.gz'])
        jobs.append((row,captures[row['key']],hashes))
    dev_jobs = [j for j in jobs if j[0]['cohort']=='development']
    if len(dev_jobs)!=20 or {j[0]['layout'] for j in dev_jobs}!={'wing','shopfront'}:
        raise ValueError('Unexpected development split')
    out.mkdir(parents=True,exist_ok=False)
    stamp = revision()
    with ProcessPoolExecutor(max_workers=workers,initializer=mark_image_only) as pool:
        selection = freeze(list(pool.map(development,dev_jobs)),[j[0] for j in dev_jobs])
        save(out/'development-freeze.json',selection)
        print('Frozen development selections',selection['selected'],flush=True)
        results = list(pool.map(evaluate,[(*j,selection['selected'],out) for j in jobs]))
    table = {'all':summary(results,METHODS)}
    for scope in (*COHORTS,*sorted({r['layout'] for r in results})):
        table[scope] = summary([r for r in results if scope in (r['cohort'],r['layout'])],METHODS)
    save(out/'scores.json',table)
    save(out/'cases.json',results)
    (out/'pictures.md').write_text(picture_table(results,METHODS,NAMES),encoding='utf8')
    sheet(out,results,captures,table)
    save(out/'manifest.json',dict(run_id=out.name,code=stamp,protocol=PROTOCOL,version=suite['version'],
        capture=base['capture'],fixtures=fixtures,reference_sha256=digest(ARCHIVE),splits=base['splits'],
        frames=base['frames'],methods=NAMES,baseline='plane-recovery',selected=selection['selected'],
        stages={'source':'single screenshot','outlines':'Regions First + Wall Recovery + Pixel-wise Planarity',
                'challengers':'two independent refinements of the same final masks','counter':'masked-repeat'},
        upstream_manifests={p.name:digest(p/'manifest.json') for p in (BASE,PLANES,SAM_ROOT)},
        new_inference_s=0,score_file='scores.json',reproduce=r'.\Invoke-SurfaceRound.ps1 -Round acceptance',
        artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}))
    print([(k,v['matched'],v['unpaired'],v['sizes']['correct']) for k,v in table['all'].items()],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs'/f'plane-acceptance-{uuid4().hex[:12]}')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    run(args.out.resolve(), args.workers)
