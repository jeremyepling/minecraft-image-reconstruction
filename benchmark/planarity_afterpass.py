"""Select a Pixel-wise Planarity after-pass on development images, then grade the frozen choice."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_metrics import aggregate, measure
from benchmark.planarity_fields import MODEL_SHA
from benchmark.surface_reference import recorded
from benchmark.wall_join_test import BASE, SOURCE_METHOD, capture_index, checked, read_gzip
from engine.planarity_fusion import PRESETS, SETTINGS, predict
from runtime.mode import image_only


RULE = ('Consider only settings within 0.5 percentage points of baseline development opening fill. '
        'Maximize matched surfaces, then surface F1, then correctly assigned pixels; prefer the earlier '
        'simpler setting on ties. If every candidate exceeds the opening budget, select least opening '
        'fill for the measured challenger and flag it as ineligible for promotion.')


def inputs(row, captures, baseline, base, folder, cache):
    sha = row['sha256']
    with Image.open(checked(captures[row['key']].path('image'),sha)) as src:
        image = Image.fromarray(np.asarray(src.convert('RGB')))
    name = row['predictions']
    parts = read_gzip(checked(baseline/name, base['artifacts'][name]))[SOURCE_METHOD]
    field_name = f'fields/{sha}.npz'
    with np.load(checked(folder/field_name,cache['artifacts'][field_name]),allow_pickle=False) as file:
        fields = dict(file)
    if tuple(fields['image_size'])!=image.size:
        raise ValueError(f'Fields disagree with source size for {sha}')
    return image,parts,fields


def development(rows, captures, baseline, base, folder, cache, archive):
    dev = [r for r in rows if r['cohort']=='development']
    if len(dev)!=20 or {r['layout'] for r in dev}!={'shopfront','wing'}:
        raise ValueError('Changed development split')
    totals = {k:Counter() for k in ['baseline',*PRESETS]}
    metrics = ('matched','eligible','proposed','correct_pixels','opening_filled','opening_pixels')
    for row in dev:
        image,parts,fields = inputs(row,captures,baseline,base,folder,cache)
        _,ref = recorded(captures[row['key']],archive)
        totals['baseline'].update({k:row['scores'][SOURCE_METHOD][k] for k in metrics})
        for name in PRESETS:
            regions,_ = predict(image,parts,fields,name)
            score,_ = measure(regions,ref,use_mask=True)
            totals[name].update({k:score[k] for k in metrics})
    for t in totals.values():
        t['f1'] = 2*t['matched']/(t['eligible']+t['proposed'])
        t['opening_fill'] = t['opening_filled']/max(1,t['opening_pixels'])
    allowed = [k for k in PRESETS if totals[k]['opening_fill']<=totals['baseline']['opening_fill']+.005]
    contenders = allowed or list(PRESETS)
    def priority(k):
        s = totals[k]
        return (0 if allowed else -s['opening_fill'], s['matched'], s['f1'], s['correct_pixels'])
    selected = max(contenders,key=priority)
    return dict(selected=selected,settings=SETTINGS,presets=PRESETS,scores=totals,
                within_opening_budget=selected in allowed,rule=RULE,
                development_frames=[r['key'] for r in dev])


def raw_diagnostic(labels, ref, missed):
    """After prediction, count a raw instance even when fusion rejected its shape or position."""
    ids = ref['id_map']
    if labels.shape!=ids.shape:
        raise ValueError('Wrong diagnostic lattice')
    areas = np.bincount(labels.ravel())
    evidence = []
    for patch in sorted(missed):
        mask = ids==patch
        hit = np.bincount(labels[mask],minlength=len(areas))
        ious = hit/(areas+int(mask.sum())-hit)
        ious[0] = 0.
        label = int(np.argmax(ious))
        evidence.append(dict(patch=patch, label=label or None, best_iou=float(ious[label])))
    return evidence


@image_only()
def run(source,fields,out):
    baseline = source/'benchmark/reviews'/BASE
    data = source/'benchmark/data'
    base = read(baseline/'manifest.json')
    rows = read(checked(baseline/'cases.json',base['artifacts']['cases.json']))
    captures = capture_index(data)
    cache = read(fields/'manifest.json')
    archive = checked(data/'visible-geometry-v1.zip',base['reference_sha256'])
    if (len(rows)!=56 or set(captures)!={r['key'] for r in rows} or cache['fixtures']!=base['fixtures']
            or cache['checkpoint_sha256']!=MODEL_SHA
            or set(cache['artifacts'])!={f"fields/{r['sha256']}.npz" for r in rows}):
        raise ValueError('Wrong baseline, fields or screenshot roster')
    out.mkdir(exist_ok=False,parents=True)
    freeze = development(rows,captures,baseline,base,fields,cache,archive)
    save(out/'development-freeze.json',freeze)
    preset = freeze['selected']
    labels_key = PRESETS[preset][0]
    print('Development freeze:',preset,flush=True)
    for k,s in freeze['scores'].items():
        print(k,s['matched'],s['proposed'],round(s['f1'],4),round(s['opening_fill'],4),flush=True)
    cases,artifacts = [],{}
    for row in rows:
        image,parts,evidence = inputs(row,captures,baseline,base,fields,cache)
        start = perf_counter()
        regions,events = predict(image,parts,evidence,preset)
        elapsed = perf_counter()-start
        name = f"predictions/{row['key']}.json.gz"
        dest = out/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(gzip.compress(json.dumps(regions,separators=(',',':'),allow_nan=False).encode(),mtime=0))
        artifacts[name] = digest(dest)
        # Evaluation opens geometry only after writing the full automatic prediction.
        _,ref = recorded(captures[row['key']],archive)
        score,pairs = measure(regions,ref,use_mask=True)
        matched = {patch for _,patch in pairs}
        previous = set(row['scopes'][SOURCE_METHOD]['matched'])
        eligible = {i for i,p in ref['patches'].items() if p['eligible']}
        raw = raw_diagnostic(evidence[labels_key],ref,eligible-previous)
        cases.append({**{k:row[k] for k in ('key','sha256','layout','cohort')},'predictions':name,
            'prediction_sha256':artifacts[name],'cached_processing_s':elapsed,'score':score,'events':events,
            'matched':sorted(matched),'gained':sorted(matched-previous),'lost':sorted(previous-matched),
            'raw_before_fusion_on_baseline_misses':raw})
    save(out/'cases.json',cases)
    for name in ('cases.json','development-freeze.json'):
        artifacts[name] = digest(out/name)
    raw = [r for c in cases for r in c['raw_before_fusion_on_baseline_misses']]
    summary = {name:aggregate([c['score'] for c in cases if cohort is None or c['cohort']==cohort])
               for name,cohort in [('all',None),('development','development'),
                                   ('regression','previously_reviewed_regression')]}
    save(out/'manifest.json',dict(run_id=out.name,method='Pixel-wise Planarity after Surface Finder',
        selected=preset, sheet_labels_key=labels_key, code=revision(), fixtures=base['fixtures'],
        reference_sha256=base['reference_sha256'],version=base['version'],capture=base['capture'],splits=base['splits'],
        frames=[{k:r[k] for k in ('key','sha256','layout','cohort')} for r in rows],
        baseline=dict(run=BASE,method=SOURCE_METHOD,manifest_sha256=digest(baseline/'manifest.json')),
        checkpoint_sha256=MODEL_SHA,model_revision=cache['model_revision'],upstream=cache['upstream'],
        model_settings=cache['settings'],settings=SETTINGS,fields=os.path.relpath(fields,out).replace(os.sep,'/'),
        fields_manifest_sha256=digest(fields/'manifest.json'),artifacts=artifacts,
        freeze='development-freeze.json',scores=summary,
        raw_headroom=dict(baseline_misses=len(raw),at_half=sum(r['best_iou']>=.5 for r in raw),
                          at_quarter=sum(r['best_iou']>=.25 for r in raw),
                          gained=sum(len(c['gained']) for c in cases),lost=sum(len(c['lost']) for c in cases)),
        inference_s=sum(r['inference_s'] for r in cache['timing'].values()),load_s=cache['load_seconds'],
        segmentation_s=sum(r['segmentation_s'] for r in cache['timing'].values()),
        cached_processing_s=sum(c['cached_processing_s'] for c in cases),
        reproduce='python -m benchmark.planarity_afterpass --source-root . --fields <numeric-cache> --out runs/planarity-replay'))
    print(f"Completed {len(cases)} pictures: {summary['all']['matched']}/{summary['all']['eligible']}",flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--fields',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    run(args.source_root.resolve(),args.fields.resolve(),args.out.resolve())
