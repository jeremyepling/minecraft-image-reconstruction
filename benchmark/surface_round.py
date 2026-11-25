"""Evaluate exterior outlines, plane-local texture pitch and visible 3D on native captures."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, __version__ as pillow_version

from benchmark.camera_round import ROOT, fixture_digest, read, revision, save
from benchmark.camera_metrics import truth_camera
from benchmark.exterior_set import Capture, check_capture_conditions, load_suite
from benchmark.surface_fixture import FROZEN, FROZEN_SHA256
from benchmark.surface_metrics import score_planes, score_pitch
from benchmark.surface_reference import recorded
from engine.surface_model import assemble, intrinsic, pitch, proposals_for, SETTINGS
from runtime.mode import image_only, mark_image_only


OUTLINES = ('contact', 'edges', 'edges+shade')
PITCH = ('texel','repeat')
RULE = dict(outlines='Maximize mean development-layout plane F1 at IoU 0.5. Tie: registry order.',
            pitch='Maximize layout-mean net correct count rate on diagnostic charts (+1 both spans within half a block, -1 wrong, 0 abstain). Tie: more accurate registered grids, then registry order.')


def image_of(case):
    with Image.open(case.path('image')) as src:
        return Image.fromarray(np.asarray(src.convert('RGB')))


def evaluate_outlines(job):
    case, folder = job
    image = image_of(case)
    predictions = {m:proposals_for(image,'26.3',m) for m in OUTLINES}
    _, t = recorded(case)
    result = {m:dict(proposals=p, scores=score_planes(t,p)) for m,p in predictions.items()}
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'outlines.json',result)


def lift_scores(model, matches):
    result = []
    for match in matches:
        p = model['planes'][match['proposal_index']]
        item = dict(patch_id=match['patch_id'], id=p['id'], units=p.get('units'),
                    truth_depth=match['depth_at_center'])
        if 'normal_camera' in p:
            cos = abs(np.dot(p['normal_camera'],match['normal_camera']))
            item['normal_error_deg'] = float(np.degrees(np.arccos(np.clip(cos,0,1))))
            if p.get('units') == 'blocks' and match['depth_at_center'] is not None:
                item['depth'] = float(np.mean(np.asarray(p['corners_camera'])[:,2]))
                item['depth_error_blocks'] = abs(item['depth'] - match['depth_at_center'])
        result.append(item)
    return result


def evaluate_scale(job):
    case, folder, outlines = job
    image = image_of(case)
    auto = read(folder/'outlines.json')[outlines]
    parts = auto['proposals']
    pose,t = recorded(case)
    # Oracle charts are explicitly diagnostic. Only this loop receives truth-derived footprints.
    rectangles = t['rectangles']
    report = {}
    for method in PITCH:
        oracle = []
        for rect in rectangles:
            q = rect['quad']
            pred = pitch(image,q,'26.3',method)
            score = score_pitch(t,rect['patch_id'],q,pred)
            oracle.append(dict(patch_id=rect['patch_id'], quad=q, prediction=pred, score=score))
        predictions = [pitch(image,p['quad'],'26.3',method) for p in parts]
        model = assemble(parts,predictions,image.size)
        matches = auto['scores']['matches']
        automatic = []
        for row in matches:
            idx, patch = row['proposal_index'],row['patch_id']
            automatic.append(dict(id=parts[idx]['id'],patch_id=patch,
                                  score=score_pitch(t,patch,parts[idx]['quad'],predictions[idx])))
        # Recorded focal length is used only in the paired lifting diagnostic.
        P = truth_camera(pose)
        z = P[2,:3] / np.linalg.norm(P[2,:3])
        f = np.linalg.norm(P[0,:3] - (P[0,:3] @ z)*z) / np.linalg.norm(P[2,:3])
        known = assemble(parts,predictions,image.size,intrinsic(image.size,f))
        report[method] = dict(oracle=oracle,auto=automatic,model=model,
                              estimated_lift=lift_scores(model,matches),known_lift=lift_scores(known,matches))
    save(folder/'surfaces.json',report)


def frac(a,b):
    return a / b if b else 0.


def artifact_path(path, run):
    return Path(os.path.relpath(path, run)).as_posix()


def outline_summary(rows):
    n = sum(r['eligible'] for r in rows)
    p = sum(r['proposed'] for r in rows)
    m = sum(r['matched'] for r in rows)
    pixels = sum(r['proposed_pixels'] for r in rows)
    return dict(matched=m,eligible=n,proposed=p,precision=m/p if p else None,recall=frac(m,n),
                f1=frac(2*m,p+n),foreground_spill=sum(r['spill_pixels'] for r in rows)/pixels if pixels else None)


def pitch_summary(rows):
    accurate = sum(bool(r.get('half_block_both')) for r in rows)
    supported = sum(r.get('counts') is not None for r in rows)
    registered = sum(bool(r.get('half_block_both')) and (r.get('grid_f1') or 0.)>=.5 for r in rows)
    return dict(charts=len(rows),counts_returned=supported,within_half_block=accurate,
                charts_with_grid_points=sum(r['predicted_intersections']>0 for r in rows),
                accurate_registered=registered,registered_rate=frac(registered,len(rows)),
                accurate_rate=frac(accurate,len(rows)),net_correct_rate=frac(2*accurate-supported,len(rows)),
                grid_precision=frac(sum(r['predicted_within_3px'] for r in rows),sum(r['predicted_intersections'] for r in rows)),
                grid_recall=frac(sum(r['truth_within_3px'] for r in rows),sum(r['truth_intersections'] for r in rows)))


def depth_summary(rows):
    angles = [r['normal_error_deg'] for r in rows if 'normal_error_deg' in r]
    depths = [r['depth_error_blocks'] for r in rows if 'depth_error_blocks' in r]
    return dict(matched=len(rows),normals=len(angles),within_15_degrees=sum(a<=15 for a in angles),
                median_normal_error_degrees=float(np.median(angles)) if angles else None,
                metric_depths=len(depths),within_one_block=sum(d<=1 for d in depths),
                median_depth_error_blocks=float(np.median(depths)) if depths else None)


def summaries(cases, out, with_scale):
    summary = {}
    for group in ['all', *sorted({c.frame['layout'] for c,_ in cases})]:
        cohort = [(c,path) for c,path in cases if group=='all' or c.frame['layout']==group]
        data = [read(out/path/'outlines.json') for _,path in cohort]
        item = dict(frames=len(data),outlines={m:outline_summary([d[m]['scores'] for d in data]) for m in OUTLINES})
        if with_scale:
            data = [read(out/path/'surfaces.json') for _,path in cohort]
            item['pitch'] = {m:{mode:pitch_summary([r['score'] for d in data for r in d[m][mode]]) for mode in ('oracle','auto')} for m in PITCH}
            item['lift'] = {m:{mode:depth_summary([r for d in data for r in d[m][mode]]) for mode in ('estimated_lift','known_lift')} for m in PITCH}
            item['connections'] = {m:sum(len(d[m]['model']['shared_edges']) for d in data) for m in PITCH}
        summary[group] = item
    return summary


@image_only()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--workers',type=int,default=3)
    args = ap.parse_args()
    run_id = 'surface-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    out = (args.out or ROOT/'runs'/run_id).resolve()
    out.mkdir(parents=True,exist_ok=False)
    suite, pairs = load_suite()
    groups = defaultdict(list)
    for variants in pairs.values():
        for case in variants.values():
            group = 'dev' if case.frame['split']=='dev' else 'regression'
            groups[group].append((case,Path('cases')/case.floor/case.id))
    from benchmark.surface_world import OUTPUT as world_dir, FILENAME as archive
    from benchmark.camera_round import digest
    world = read(world_dir/'world-index.json')['worlds'][0]
    if digest(world_dir/archive) != world['sha256']:
        raise ValueError('Frozen confirmation world differs from its index.')
    frozen = read(FROZEN/'index.json')
    if fixture_digest(FROZEN,frozen) != FROZEN_SHA256 or frozen['client_sha256'] != suite['client_sha256']:
        raise ValueError('Confirmation release digest or client changed')
    seen = {case.frame['layout'] for cases in groups.values() for case,_ in cases}
    for f in frozen['frames']:
        if f['layout'] in seen or f['split']!='holdout':
            raise ValueError('Confirmation layout overlaps development or regression')
        case = Capture(f['floor'],FROZEN,f)
        check_capture_conditions(read(case.path('pose')),suite['capture_conditions'],case.id)
        groups['confirmation'].append((case,Path('cases')/case.floor/case.id))
    stages = {}
    summary = {}
    code = revision()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=mark_image_only) as pool:
        for cohort in ('dev','regression','confirmation'):
            cases = groups[cohort]
            print(f'{cohort}: scoring three outline methods on {len(cases)} images',flush=True)
            list(pool.map(evaluate_outlines,[(c,out/p) for c,p in cases]))
            if cohort=='dev':
                partial = summaries(cases,out,False)
                stages['outlines'] = max(OUTLINES,key=lambda m: np.mean([v['outlines'][m]['f1'] for k,v in partial.items() if k!='all']))
            print(f'{cohort}: testing both pitch methods using {stages["outlines"]} outlines',flush=True)
            list(pool.map(evaluate_scale,[(c,out/p,stages['outlines']) for c,p in cases]))
            summary[cohort] = summaries(cases,out,True)
            if cohort=='dev':
                by_layout = [s for layout,s in summary[cohort].items() if layout!='all']
                stages['pitch'] = max(PITCH,key=lambda m: tuple(np.mean([s['pitch'][m]['oracle'][stat] for s in by_layout])
                                                        for stat in ('net_correct_rate','registered_rate')))
                save(out/'selection.json',dict(rule=RULE,selected=stages,development=summary['dev']))
                print(f'Development selection frozen: {stages}',flush=True)
    records = [dict(id=c.id,floor=c.floor,layout=c.frame['layout'],palette=c.frame['palette'],cohort=k,
                    source=artifact_path(c.path('image'),out),sha256=c.frame['image_sha256'],folder=p.as_posix())
               for k,cases in groups.items() for c,p in cases]
    save(out/'cases.json',records)
    save(out/'scores.json',summary)
    floors = {cohort:{floor:summaries([(c,p) for c,p in cases if c.floor==floor],out,True)['all']
                      for floor in sorted({c.floor for c,p in cases})} for cohort,cases in groups.items()}
    save(out/'floor-scores.json',floors)
    from benchmark.surface_topology import evaluate_run
    topology = evaluate_run(out)
    (out/'feedback.jsonl').touch()
    from benchmark.surface_sheet import render
    render(out,groups,stages)
    from engine.surface_edges import SETTINGS as edges_settings
    from engine.surface_pitch import SETTINGS as pitch_settings
    manifest = dict(run_id=out.name,code=code,version='26.3',client_sha256=suite['client_sha256'],
                    environment=dict(python=sys.version.split()[0],numpy=np.__version__,pillow=pillow_version),
                    capture=suite['capture_conditions'], split={k:sorted({c.frame['layout'] for c,_ in v}) for k,v in groups.items()},
                    releases={**{f:d['sha256'] for f,d in suite['conditions'].items()},'surface-confirm-v1':FROZEN_SHA256},
                    confirmation_world=dict(archive=artifact_path(world_dir/archive,out),sha256=world['sha256']),
                    stages=stages,methods=dict(outlines=list(OUTLINES),pitch=list(PITCH),lift='shared-edges-v1'),
                    settings=dict(outlines=edges_settings,shade_weight=.22,pitch=pitch_settings,lift=SETTINGS,
                                  reference_methods=['engine.camera_grid:seam-fit/contact_planes','engine.texel_grid:texel_grid']),
                    selection_rule=RULE, confirmation=summary['confirmation']['all'],
                    confirmation_topology=topology['cohorts']['confirmation'],
                    artifacts=dict(review='comparison.jpg',gallery='fixtures.jpg',scores='scores.json',selection='selection.json',
                                   floors='floor-scores.json',topology='topology.json',review_regions='review-regions.json',
                                   cases='cases.json',per_case=['outlines.json','surfaces.json'],visual_feedback='feedback.jsonl'))
    save(out/'manifest.json',manifest)
    print(out,flush=True)


if __name__=='__main__':
    main()
