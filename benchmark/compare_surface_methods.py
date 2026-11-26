"""Run one fresh surface candidate offline, or compare up to three completed candidate jobs."""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
from importlib import import_module
import json
from pathlib import Path
from time import perf_counter
from types import CodeType

import numpy as np
from PIL import __version__ as pillow_version

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.normal_fields import cached_fields
from benchmark.normal_round import groups, relation_score, support_gap
from benchmark.score_saved_planes import ROUND
from benchmark.surface_metrics import score_pitch, score_planes
from benchmark.surface_reference import ARCHIVE, recorded
from benchmark.surface_round import artifact_path, depth_summary, image_of, lift_scores, outline_summary, pitch_summary
from benchmark.surface_topology import opening_coverage, opening_mask
from engine import normal_rank, normal_regions, surface_edges, surface_model, surface_order, surface_pitch
from engine.surface_fields import MODELS, SETTINGS as FIELD_SETTINGS, infer


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def code_recipe(*roots):
    """Hash the selected local functions, their global helpers and their numerical constants."""
    def names(code):
        yield from code.co_names
        for value in code.co_consts:
            if isinstance(value, CodeType):
                yield from names(value)

    recipe = {}
    remaining = list(roots)
    while remaining:
        func = inspect.unwrap(remaining.pop())
        file = Path(inspect.getsourcefile(func)).resolve().relative_to(ROOT).as_posix()
        key = file + ':' + func.__qualname__
        if key in recipe:
            continue
        recipe[key] = inspect.getsource(func)
        for name in set(names(func.__code__)):
            value = func.__globals__.get(name)
            original = inspect.unwrap(value)
            if inspect.isfunction(original) and original.__module__.startswith(('engine.', 'benchmark.')):
                remaining.append(original)
            elif name.isupper() and isinstance(value, (bool, int, float, str, tuple, list, dict, np.ndarray)):
                recipe[file + ':' + name] = value.tolist() if isinstance(value, np.ndarray) else value
    return fingerprint(recipe)


def stage_implementations(methods):
    # These roots account for the lazy imports in proposals_for, pitch, infer and order.
    from benchmark import depth_probe
    from engine import camera_grid, camera_seam, cube_axes, geometry, structural_lines, surface_fields, texel_grid

    outlines = methods['outlines']
    backends = {'contact':camera_grid.predict, 'edges':surface_edges.detect_planes, 'edges+shade':surface_edges.detect_planes,
                'normal-rank':normal_rank.detect, 'normal-regions':normal_regions.detect}
    edges = [surface_model.proposals_for]
    if outlines in backends:
        edges.append(backends[outlines])
    else:
        # For a new worker-defined dispatch branch, conservatively include its imported
        # local backends. A known method above fingerprints only its own active backend.
        for node in ast.walk(ast.parse(inspect.getsource(surface_model.proposals_for))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(('engine.', 'benchmark.')):
                module = import_module(node.module)
                for alias in node.names:
                    backend = getattr(module, alias.name)
                    if inspect.isfunction(backend):
                        edges.append(backend)
    if outlines == 'contact':
        edges.extend([camera_seam.predict, cube_axes.fit_axes, structural_lines.detect])
    ordering = [surface_order.order, geometry.project, geometry.transform]
    if outlines.startswith('normal-'):
        edges.extend([surface_fields.observed_normal, infer])
    if methods['order'] == 'depth-anything':
        ordering.extend([infer, depth_probe.preprocess])
    spacing = surface_pitch.fit if methods['pitch'] == 'repeat' else texel_grid.texel_grid
    return dict(outlines=code_recipe(*edges), pitch=code_recipe(surface_model.pitch, spacing),
                lift=code_recipe(surface_model.assemble), order=code_recipe(*ordering))


def evaluator_signature():
    return code_recipe(evaluate, summary, grouped_summary)


def stage_settings(methods):
    outlines = methods['outlines']
    settings = {stage:{} for stage in methods}
    if outlines in ('edges', 'edges+shade', 'normal-rank', 'normal-regions'):
        settings['outlines']['edges'] = surface_edges.SETTINGS
    if outlines.startswith('normal-'):
        settings['outlines'].update(inference=FIELD_SETTINGS, model=MODELS['normal'][2])
    if outlines == 'normal-rank':
        settings['outlines']['ranking'] = normal_rank.SETTINGS
    if outlines == 'normal-regions':
        settings['outlines']['regions'] = normal_regions.SETTINGS
    if methods['pitch'] == 'repeat':
        settings['pitch'] = surface_pitch.SETTINGS
    settings['order'] = dict(surface_order.SETTINGS)
    if methods['order'] == 'depth-anything':
        settings['order'].update(inference=FIELD_SETTINGS, model=MODELS['depth'][2])
    return settings


def predict(image, version, methods, model_fields):
    """Only image-derived inputs cross this boundary."""
    parts = surface_model.proposals_for(image, version, methods['outlines'], model_fields.get('normal'))
    pitches = [surface_model.pitch(image, part['quad'], version, methods['pitch']) for part in parts]
    model = surface_model.assemble(parts, pitches, image.size)
    model['relative_depth_order'] = surface_order.order(model['planes'], image.size, model_fields.get('depth'))
    return model


def evaluate(model, reference, truth_pairs):
    parts = model['planes']
    outlines = score_planes(reference, parts)
    matches = outlines['matches']
    pitches = []
    for match in matches:
        part = parts[match['proposal_index']]
        pitches.append(score_pitch(reference, match['patch_id'], part['quad'], part['pitch']))
    gaps = opening_mask(reference['id_map'])
    filled = opening_coverage(gaps, parts)
    eligible = {i for i, p in reference['patches'].items() if p['eligible']}
    relations = relation_score(model['relative_depth_order'], parts, matches, truth_pairs, eligible)
    return dict(outlines=outlines, pitch=pitches, lift=lift_scores(model,matches), order=relations, gaps=filled,
                support_gaps=support_gap(gaps, parts, filled, reference['image_size']))


def summary(rows):
    data = [r['scores'] for r in rows]
    result = dict(images=len(rows), outlines=outline_summary([s['outlines'] for s in data]),
                  pitch=pitch_summary([p for s in data for p in s['pitch']]),
                  lift=depth_summary([p for s in data for p in s['lift']]))
    for stage, keys in [('gaps', ('filled_pixels','opening_pixels')), ('support_gaps', ('filled_pixels','opening_pixels')),
                        ('order', ('correct', 'eligible', 'proposed', 'reversed', 'other'))]:
        result[stage] = {k:sum(s[stage][k] for s in data) for k in keys}
    return result


def grouped_summary(rows):
    by_layout = defaultdict(list)
    for row in rows:
        by_layout[row['layout']].append(row)
    layouts = {layout:summary(subset) for layout,subset in sorted(by_layout.items())}
    scores = summary(rows)
    scores['mean_layout_f1'] = sum(s['outlines']['f1'] for s in layouts.values()) / len(layouts)
    return layouts, scores


def run(args):
    started = perf_counter()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = (args.out or ROOT / 'runs' / 'surface-experiments' / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    code = revision()
    # This experiment compares historical four-corner candidates with a quad-based scorer.
    methods = dict(outlines=args.outlines or 'edges+shade', pitch=args.pitch or 'repeat', order=args.order or 'geometry')
    if methods['outlines'] == 'mask-first':
        raise ValueError('Use Invoke-MaskRound.ps1 to compare region masks against the frozen suite.')
    kinds = (['normal'] if methods['outlines'].startswith('normal-') else []) + (
        ['depth'] if methods['order'] == 'depth-anything' else [])
    cohorts, suite, fixture_hashes = groups()
    cases = [c for split, group in cohorts.items() if args.split in ('all', split) for c in group]
    # normals-01 stores occlusion pairs from measured camera rays. Only their true plane IDs enter scoring.
    label_file = ROUND / 'cases.json'
    labels = {row['key']: row for row in read(label_file)}
    rows, inputs = [], []
    for case in cases:
        key = f'{case.floor}/{case.id}'
        sha = case.frame['image_sha256']
        if digest(case.path('image')) != sha:
            raise ValueError(f'Screenshot changed: {key}')
        image = image_of(case)
        if args.fields and kinds:
            fields = cached_fields(sha, args.fields, kinds)
            if fields is None:
                raise ValueError(f'No compatible saved model field for {key} in {args.fields}')
        else:
            fields = {kind: infer(image,kind) for kind in kinds}
        model = predict(image, suite['version'], methods, fields)
        target = out / 'predictions' / case.floor / (case.id + '.json.gz')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(json.dumps(model,separators=(',', ':'),allow_nan=False).encode(),mtime=0))
        _, reference = recorded(case)
        if labels[key]['sha256'] != sha:
            raise ValueError(f'Ordering reference belongs to another screenshot: {key}')
        scores = evaluate(model, reference, labels[key]['depth_diagnostic'])
        rows.append(dict(key=key, layout=case.frame['layout'], sha256=sha,
                         field_sha256={k:hashlib.sha256(np.ascontiguousarray(v)).hexdigest() for k,v in fields.items()},
                         prediction=artifact_path(target,out), scores=scores))
        inputs.append((key,sha))
    layouts, scores = grouped_summary(rows)
    save(out/'cases.json',rows)
    save(out/'layouts.json',layouts)
    pair_inputs = {key:sorted((p['near_patch'],p['far_patch']) for p in labels[key]['depth_diagnostic']['pairs'])
                   for key, _ in inputs}
    comparison = dict(version=suite['version'],split=args.split,screenshots=fingerprint(sorted(inputs)),source_reader=code_recipe(image_of),
                      fixtures=fixture_hashes, geometry_sha256=digest(ARCHIVE),
                      occlusion_labels_sha256=fingerprint(pair_inputs), evaluator=evaluator_signature())
    field_source = ('verified saved image maps' if args.fields else 'local inference') if kinds else 'unused'
    manifest = dict(schema=2,run_id=out.name,name=args.name,comparison=comparison,code=code,
                    prediction_source='fresh from screenshot',model_fields=field_source,
                    field_cache=str(args.fields.resolve()) if args.fields and kinds else None,
                    methods=methods,settings=stage_settings(methods),lift=dict(method='shared-edges-v1',settings=surface_model.SETTINGS),
                    implementations=stage_implementations(methods),
                    model_sha256={k:MODELS[k][2] for k in kinds},
                    environment=dict(numpy=np.__version__,pillow=pillow_version),scores=scores,
                    elapsed_seconds=perf_counter()-started,artifacts=dict(cases='cases.json',layouts='layouts.json',
                    predictions='predictions/'))
    save(out/'manifest.json',manifest)
    print_summary(manifest)
    print(out/'manifest.json')


def print_summary(manifest):
    s = manifest['scores']
    o, g, p, d = (s[k] for k in ('outlines','gaps','pitch','order'))
    print(f"{manifest['name']}  |  {s['images']} images  |  {o['matched']}/{o['eligible']} surfaces"
          f"  |  {o['proposed']-o['matched']} extra  |  {g['filled_pixels']}/{g['opening_pixels']} opening samples filled"
          f"  |  {p['within_half_block']}/{p['charts']} accurate spans  |  {d['correct']}/{d['proposed']} depth orders verified")


def validate_comparison(manifests):
    if not 2 <= len(manifests) <= 3:
        raise ValueError('Select the baseline and one or two challengers.')
    first = manifests[0]
    if any(m.get('schema') != 2 for m in manifests):
        raise ValueError('Run each candidate with the current experiment format to compare its stage implementations.')
    changed = set()
    for other in manifests[1:]:
        if first['comparison'] != other['comparison']:
            raise ValueError('Compare identical splits, screenshots, recorded geometry and scoring code.')
        if first['lift'] != other['lift']:
            changed.add('lift')
        changed.update(stage for stage in first['methods'] if first['methods'][stage] != other['methods'][stage]
                       or first['settings'][stage] != other['settings'][stage])
        changed.update(stage for stage in first['implementations']
                       if first['implementations'][stage] != other['implementations'][stage])
    if len(changed) > 1:
        raise ValueError(f"Compare one configured stage at a time; these runs change {', '.join(sorted(changed))}.")


def main():
    from runtime.mode import image_only

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    job = sub.add_parser('run', help='Generate and score a fresh candidate in its own folder')
    job.add_argument('--name', required=True)
    job.add_argument('--out', type=Path)
    job.add_argument('--outlines', help='Outline method registered in engine.surface_model')
    job.add_argument('--pitch', choices=('repeat','texel'))
    job.add_argument('--order', choices=('geometry','depth-anything'))
    job.add_argument('--split', choices=('development','regression','blind','all'),default='development')
    job.add_argument('--fields', type=Path, help='Read verified image-only model maps without updating this folder')
    view = sub.add_parser('compare', help='Compare the baseline result and one or two challengers')
    view.add_argument('runs', type=Path, nargs='+', help='Completed candidate run directories, baseline first')
    args = parser.parse_args()
    try:
        with image_only():
            if args.action == 'run':
                run(args)
            else:
                manifests = [read(path/'manifest.json') for path in args.runs]
                validate_comparison(manifests)
                for manifest in manifests:
                    print_summary(manifest)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
