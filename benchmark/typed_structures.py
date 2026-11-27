"""Score exact block IDs on the locked visible-geometry suite, using its geometry-only alignment."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

# Each image is a separate process. Keep small matrix operations single-threaded.
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MCIR_IMAGE_ONLY='1')

import numpy as np
from PIL import Image

from benchmark import current_placement_viewer as viewer
from benchmark import pavement_opening_eval as pavement
from benchmark import smooth_joint_placement as old
from benchmark.mask_blocks import decoded_pack
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.texture_camera_eval import checked, digest, read, save
from benchmark.thin_face_placement_run import ref_inputs
from engine.export import JAVA_26_3_FORMAT, write_pack

SUITE = viewer.suite
GEOMETRY_RELEASES = [viewer.ROOT/name for name in (
    'pavement-local-01', 'smooth-local-placement-01', 'smooth-joint-placement-01',
    'window-wall-01', 'counted-ground-corner-01', 'workshop-foreground-01')]
BASE = GEOMETRY_RELEASES[-1]
REVIEW = viewer.ROOT / 'typed-structures-01'
CACHE = SUITE.ROOT / 'runs/typed-structures'
GEOMETRY_MATERIAL = 'minecraft:smooth_stone'


def compact(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(data, separators=(',', ':'), allow_nan=False).encode('utf-8'))


def overlap(tp, fn, fp):
    denom = 2 * tp + fn + fp
    return dict(correct=tp, missed=fn, extras=fp, f1=2 * tp / denom if denom else None)


def score(predicted, truth, optional, assigned):
    """Predictions/truth are world-coordinate -> namespaced ID after the geometry-selected translation."""
    if not optional <= truth.keys() or not assigned <= predicted.keys():
        raise ValueError('Optional/assigned position outside its block set')
    at_position = predicted.keys() & truth.keys()
    same = {p for p in at_position if predicted[p] == truth[p]}
    required = truth.keys() - optional
    required_placed = at_position & required
    required_same = same & required
    extras = len(predicted.keys() - truth.keys())
    # Optional visible slivers carry no ID or occupancy penalty at their true location.
    strict = overlap(len(same), len(truth) - len(same), extras + len(at_position - same))
    facade = overlap(len(required_same), len(required) - len(required_same), extras + len(required_placed - required_same))
    material = dict(correct=len(same), positions=len(at_position),
                    accuracy=len(same) / len(at_position) if at_position else None)
    return dict(strict=strict, facade=facade, material=material, assigned=len(assigned),
                fallback=len(predicted)-len(assigned), assigned_correct=len(assigned & same),
                assigned_positions=len(assigned & at_position))


def geometry_score(predicted, truth, optional):
    positions = predicted.keys() & truth.keys()
    extras = len(predicted.keys() - truth.keys())
    required = truth.keys() - optional
    return dict(strict=overlap(len(positions), len(truth.keys() - positions), extras),
                facade=overlap(len(positions & required), len(required - positions), extras))


def aligned_sites(cells, transform, shift):
    if not cells:
        return []
    centers = np.array(cells) + .5
    world = np.floor(centers @ transform[:3, :3].T + transform[:3, 3]).astype(int) + shift
    result = [tuple(p) for p in world.tolist()]
    if len(set(result)) != len(cells):
        raise ValueError('Geometry registration collapsed distinct positions')
    return result


@lru_cache(maxsize=1)
def releases():
    summary, _, payloads = viewer.read_releases(GEOMETRY_RELEASES)
    cases = read(checked(BASE / 'cases.json', summary['cases_sha256']))
    return {row['id']: row for row in cases}, payloads


def source_path(split, frame):
    f = SUITE.framed
    if split in ('train', 'test', 'validation', 'regression'):
        return f.TRAINING_DATA / 'data' / split / frame['directory'] / 'rgb.png'
    if split == 'native':
        return f.native.rgb_path(frame, f.native.BUILDS)
    root = {'shelter': SUITE.SHELTER, 'new': f.NEW_INPUTS}.get(split, f.transfer.BUILDS)
    return root / frame['directory'] / 'captures' / (frame['source_id'] + '.png')


def prepare(task):
    job, cache = task
    split, frame, _, _ = job
    key = frame['id']
    cases, payloads = releases()
    case = cases[key]
    pack_hash = case.get('pack_sha256', case['previous_pack_sha256'])
    if key in payloads:
        if sha256(payloads[key]).hexdigest() != pack_hash:
            raise ValueError(f'Changed geometry export for {key}')
        cells = sorted(decoded_pack(BytesIO(payloads[key])))
    else:
        actual, positions = old.old_pack(job)
        if actual != pack_hash:
            raise ValueError(f'Archived geometry export mismatch for {key}')
        cells = sorted(positions)
    source_hash = frame['files']['rgb.png']
    rgb = checked(source_path(split, frame), source_hash)
    if source_hash != case['source_hash']:
        raise ValueError(f'Mismatched source image for {key}')
    _, _, _, camera = SUITE.ordinary_inputs(*job)
    # Inference reads only this input. Alignment and visible labels live in a separate directory.
    compact(cache / 'inputs' / (key + '.json'), dict(rgb=str(rgb), source_hash=source_hash,
            cells=cells, camera=camera, geometry_pack_sha256=pack_hash))
    coords, _, transform, _ = ref_inputs(job)
    labels, ids = pavement.visible_faces(split, frame)
    truth = {tuple(b['xyz_world']):b['block_id'] for b in labels['blocks']}
    optional = horizontal_slivers(labels, ids)
    if truth.keys() != coords:
        raise ValueError(f'Visible material labels differ from geometry truth for {key}')
    translation = case['result']['translation']
    aligned = aligned_sites(cells, transform, translation)
    baseline = dict.fromkeys(aligned, GEOMETRY_MATERIAL)
    geom = geometry_score(baseline, truth, optional)
    for metric in geom:
        if any(geom[metric][field] != case['result'][metric][field] for field in ('correct','missed','extras')):
            raise ValueError(f'Geometry reference mismatch for {key} in {metric}')
    compact(cache / 'reference' / (key + '.json'), dict(split=split, translation=translation,
            geometry=geom, aligned=aligned, truth=[[*p, name] for p,name in sorted(truth.items())],
            optional=sorted(optional), baseline=score(baseline, truth, optional, set())))
    return key


def layouts():
    ranking = read(viewer.ROOT / 'facade-placement-01/worst/ranking.json')
    return {key:(item['name'], item['shape_sha256']) for item in ranking['all_layouts'] for key in item['frame_ids']}


def pooled_scores(scores):
    result = {}
    for metric in ('strict', 'facade'):
        counts = (sum(s[metric][k] for s in scores) for k in ('correct', 'missed', 'extras'))
        result[metric] = overlap(*counts)
    if scores and 'material' in scores[0]:
        correct = sum(s['material']['correct'] for s in scores)
        positions = sum(s['material']['positions'] for s in scores)
        result['material'] = dict(correct=correct, positions=positions, accuracy=correct/positions if positions else None)
        for k in ('assigned', 'fallback', 'assigned_correct', 'assigned_positions'):
            result[k] = sum(s[k] for s in scores)
    return result


def bundle_file(bundle, member, payload):
    # Freeze ZIP timestamps. Structure template contents still come from the ordinary writer.
    info = ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    bundle.writestr(info, payload)


def typed_pack(ids):
    raw = BytesIO()
    write_pack(raw, ids, JAVA_26_3_FORMAT)
    stable = BytesIO()
    with ZipFile(raw) as source, ZipFile(stable, 'w') as dest:
        for member in source.namelist():
            bundle_file(dest, member, source.read(member))
    payload = stable.getvalue()
    decoded = decoded_pack(BytesIO(payload))
    if decoded != {p:'minecraft:'+v for p,v in ids.items()}:
        raise ValueError('Decoded export changed predicted IDs or position')
    return payload, decoded


def grade_view(inputs, ref, estimates):
    cells = list(map(tuple, inputs['cells']))
    coords = list(map(tuple, ref['aligned']))
    if len(estimates) != len(cells) or len(coords) != len(cells):
        raise ValueError('Site count changed before material scoring')
    ids = {p:v[0] for p,v in zip(cells, estimates, strict=True)}
    data, decoded = typed_pack(ids)
    evidence = {p:v[1] for p,v in zip(cells, estimates, strict=True)}
    if any(evidence[p] == 'fallback' and name != GEOMETRY_MATERIAL for p,name in decoded.items()):
        raise ValueError('Fallback exported a different material from the geometry baseline')
    selected = {w:decoded[p] for p,w in zip(cells, coords, strict=True)}
    assigned = {w for p,w in zip(cells, coords, strict=True) if evidence[p] != 'fallback'}
    truth = {tuple(row[:3]):row[3] for row in ref['truth']}
    optional = set(map(tuple, ref['optional']))
    scored = score(selected, truth, optional, assigned)
    if geometry_score(selected, truth, optional) != ref['geometry']:
        raise ValueError('Export changed frozen geometry')
    confusion = Counter((truth[p], selected[p]) for p in truth.keys() & selected.keys())
    sites = [[*p, decoded[p], evidence[p]] for p in cells]
    return scored, data, sites, confusion


def predictions_for(path, inputs, run):
    data = read(path)
    if 'methods' in data:  # Predictions are bound to the exact source, geometry and matcher.
        for k in ('source_hash', 'geometry_pack_sha256'):
            if data[k] != inputs[k]:
                raise ValueError(f'{k} mismatch in {path}')
        if data['predictor_sha256'] != run['predictor_sha256']:
            raise ValueError(f'Predictor mismatch in {path}')
        return data['methods']
    raise ValueError(f'Unsigned predictions in {path}')


def development(cache, predictions):
    """Pooled exact-ID scores on training only, for choosing the next candidate."""
    run = read(predictions/'run.json')
    keys = run['keys']
    if len(set(keys)) != 72 or any(not k.startswith('train-') for k in keys):
        raise ValueError('Development comparison needs the 72 training images')
    matrix = defaultdict(Counter)
    results = defaultdict(list)
    for key in keys:
        inputs = read(cache/'inputs'/(key+'.json'))
        ref = read(cache/'reference'/(key+'.json'))
        if ref['split'] != 'train':
            raise ValueError('Non-training reference supplied to development comparison')
        for name, estimates in predictions_for(predictions/(key+'.json'), inputs, run).items():
            scores, _, _, confusions = grade_view(inputs, ref, estimates)
            results[name].append(scores)
            matrix[name].update(confusions)
    return dict(views=len(keys), predictor_sha256=run['predictor_sha256'], atlas_loader_sha256=run['atlas_loader_sha256'],
                scores={name:pooled_scores(scores) for name,scores in results.items()},
                largest_confusions={name:[[t,p,n] for (t,p),n in sorted(c.items(), key=lambda v:(-v[1],v[0]))
                                          if t!=p][:15] for name,c in matrix.items()})


def error_audit(cases, groups, matrices, atlas):
    from engine.atlas_count import CALIBRATED_INDEX
    choices = json.loads(gzip.decompress(CALIBRATED_INDEX.read_bytes()))['choices']
    fingerprints = {label:sha256(tile.tobytes()).hexdigest() for label,tile in zip(atlas.labels,atlas.tiles)}
    art = defaultdict(set)
    for facing, textures in choices.items():
        for texture, names in textures.items():
            if texture in fingerprints:
                for name in names:
                    art['minecraft:'+name].add((facing, fingerprints[texture]))
    audits = {}
    for cohort in ('fixed','other','other_unseen','train'):
        if cohort == 'fixed':
            rows = [r for r in cases if r['shape_sha256']]
            counts = matrices['fixed']
        elif cohort == 'other':
            rows = [r for r in cases if not r['shape_sha256']]
            counts = matrices['other']
        elif cohort == 'other_unseen':
            rows = [r for r in cases if not r['shape_sha256'] and r['split'] != 'train']
            counts = matrices['other'] - matrices['split-train']
        else:
            rows = [r for r in cases if r['split']=='train']
            counts = matrices['split-train']
        targets, errors = Counter(), Counter()
        identical = outside = 0
        for (truth,predicted), n in counts.items():
            targets[truth] += n
            if truth == predicted:
                continue
            errors[truth] += n
            if truth not in art or predicted not in art:
                outside += n
            elif art[truth] & art[predicted]:
                identical += n
        totals = pooled_scores([r['result'] for r in rows])
        mat = totals['material']
        fallback_positions = mat['positions'] - totals['assigned_positions']
        fallback_correct = mat['correct'] - totals['assigned_correct']
        by_view = sorted(((r['id'], r['result']['material']['positions']-r['result']['material']['correct'])
                          for r in rows), key=lambda pair:(-pair[1],pair[0]))
        comparison = {}
        for metric, value in [('strict','f1'),('facade','f1'),('material','accuracy')]:
            cmp = Counter()
            for r in rows:
                a,b = r['result'][metric][value],r['baseline'][metric][value]
                cmp['tie' if a==b else 'win' if a>b else 'loss'] += 1
            comparison[metric] = {k:cmp[k] for k in ('win','tie','loss')}
        audits[cohort] = dict(by_view=by_view, view_comparison=comparison, wrong=sum(errors.values()),
            true_ids={t:dict(positions=targets[t], wrong=errors[t]) for t in sorted(targets)},
            perfect_material_ceiling=pooled_scores([r['geometry'] for r in rows]),
            wrong_with_identical_tile_on_any_common_facing=identical, wrong_with_id_outside_atlas=outside,
            fallback_correct=fallback_correct, fallback_wrong=fallback_positions-fallback_correct)
    audits['fixed']['by_shape'] = sorted(
        ([h,d['layout'],d['result']['material']['positions']-d['result']['material']['correct']]
         for h,d in groups.items()), key=lambda row:(-row[2],row[0]))
    return audits


def release(cache, predictions, out, method, development_runs=()):
    run = read(predictions / 'run.json')
    keys = read(cache / 'manifest.json')['frames']
    if run['keys'] != keys or len(set(keys)) != 204:
        raise ValueError('A full release requires all 204 frozen images in order')
    if method not in run['methods'] or digest(predictions/'predictor.py') != run['predictor_sha256']:
        raise ValueError('Missing selected method or stale predictor')
    fixed = layouts()
    if len(fixed) != 104 or not fixed.keys() <= set(keys):
        raise ValueError('Fixed layout cohort changed')
    cases, sites, variant_cases = [], {}, defaultdict(dict)
    matrix = defaultdict(Counter)
    out.mkdir(parents=True, exist_ok=True)
    with ZipFile(out/'typed-exports.zip', 'w') as bundle:
        for key in keys:
            inputs = read(cache/'inputs'/(key+'.json'))
            ref = read(cache/'reference'/(key+'.json'))
            smooth = [[GEOMETRY_MATERIAL.split(':',1)[1], 'fallback'] for _ in ref['aligned']]
            baseline, _, _, _ = grade_view(inputs, ref, smooth)
            label, shape = fixed.get(key, (None, None))
            row = dict(id=key, split=ref['split'], layout=label, shape_sha256=shape, source_hash=inputs['source_hash'],
                       geometry_pack_sha256=inputs['geometry_pack_sha256'], translation=ref['translation'],
                       geometry=ref['geometry'], baseline=baseline)
            proposals = predictions_for(predictions/(key+'.json'), inputs, run)
            if set(proposals) != set(run['methods']):
                raise ValueError(f'Missing variant in {key}')
            for name, estimates in proposals.items():
                result, data, local, confusion = grade_view(inputs, ref, estimates)
                variant_cases[name][key] = result
                if name == method:
                    row.update(pack_sha256=sha256(data).hexdigest(), result=result)
                    sites[key] = {k:row[k] for k in ('source_hash', 'geometry_pack_sha256', 'pack_sha256')}
                    sites[key]['rows'] = local
                    bundle_file(bundle, key+'/reconstruction.zip', data)
                    matrix['fixed' if key in fixed else 'other'].update(confusion)
                    matrix['split-'+ref['split']].update(confusion)
            cases.append(row)

    def detail(group):
        return {k:pooled_scores([r[k] for r in group]) for k in ('geometry', 'baseline', 'result')} | {'views':len(group)}

    def grouped(classify):
        grouped_rows = defaultdict(list)
        for row in cases:
            name = classify(row)
            if name is not None:
                grouped_rows[name].append(row)
        return {k:detail(rows) for k,rows in sorted(grouped_rows.items())}

    compact(out/'cases.json', cases)
    site_payload = json.dumps(sites, separators=(',',':'), allow_nan=False).encode()
    (out/'sites.json.gz').write_bytes(gzip.compress(site_payload, mtime=0))
    compact(out/'confusions.json', {name:[[t,p,n] for (t,p),n in sorted(c.items(), key=lambda v:(-v[1],v[0]))]
                                    for name,c in sorted(matrix.items())})
    def variant_summary(data):
        return {name:pooled_scores([data[k] for k in members]) for name,members in {
            'fixed':sorted(fixed), 'other':sorted(set(keys)-fixed.keys()),
            **{'split-'+split:[r['id'] for r in cases if r['split']==split] for split in sorted({r['split'] for r in cases})}
        }.items()}
    compact(out/'variants.json', {name:variant_summary(data) for name,data in variant_cases.items()})
    summary = dict(schema=1, method=method, predictor_sha256=run['predictor_sha256'],
                   atlas_loader_sha256=run['atlas_loader_sha256'],
                   projected_appearance_sha256=run['projected_appearance_sha256'],
                   geometry_release=BASE.name, geometry_cases_sha256=digest(BASE/'cases.json'),
                   atlas=viewer.calibrated_atlas().provenance,
                   cases_sha256=digest(out/'cases.json'), sites_sha256=digest(out/'sites.json.gz'),
                   exports_sha256=digest(out/'typed-exports.zip'), confusions_sha256=digest(out/'confusions.json'),
                   variants_sha256=digest(out/'variants.json'), all=detail(cases),
                   cohorts=grouped(lambda r:'fixed' if r['id'] in fixed else 'other'),
                   other_unseen=detail([r for r in cases if r['id'] not in fixed and r['split'] != 'train']),
                   layouts=grouped(lambda r:r['shape_sha256']), splits=grouped(lambda r:r['split']))
    for shape, details in summary['layouts'].items():
        group = [r for r in cases if r['shape_sha256'] == shape]
        details.update(layout=group[0]['layout'], shape_sha256=shape, frames=[r['id'] for r in group])
    compact(out/'errors.json', error_audit(cases, summary['layouts'], matrix, viewer.calibrated_atlas()))
    summary['errors_sha256'] = digest(out/'errors.json')
    if development_runs:
        trials = []
        with ZipFile(out/'development-matchers.zip', 'w') as bundle:
            for directory in development_runs:
                trial = read(directory/'development.json')
                snapshot = checked(directory/'predictor.py', trial['predictor_sha256'])
                name = f'round-{len(trials)+1}/placed_materials.py'
                bundle_file(bundle, name, snapshot.read_bytes())
                trials.append(dict(snapshot=name, **trial))
        compact(out/'development.json', trials)
        summary.update(development_sha256=digest(out/'development.json'),
                       development_matchers_sha256=digest(out/'development-matchers.zip'))
    compact(out/'summary.json', summary)
    print('Published exact ID and geometry evidence for', len(cases), 'views')


def verify(cache, out):
    """Decode every released pack independently and compare its scores to the published rows."""
    summary = read(out/'summary.json')
    named = {'cases_sha256':'cases.json', 'confusions_sha256':'confusions.json', 'exports_sha256':'typed-exports.zip',
             'sites_sha256':'sites.json.gz', 'variants_sha256':'variants.json', 'errors_sha256':'errors.json'}
    if 'development_sha256' in summary:
        named.update(development_sha256='development.json', development_matchers_sha256='development-matchers.zip')
    for field, name in named.items():
        checked(out/name, summary[field])
    rows = read(out/'cases.json')
    sites = json.loads(gzip.decompress((out/'sites.json.gz').read_bytes()))
    manifest = read(cache/'manifest.json')
    if [r['id'] for r in rows] != manifest['frames'] or set(sites) != set(manifest['frames']):
        raise ValueError('Different frame inventories in material release')
    with ZipFile(out/'typed-exports.zip') as bundle:
        if sorted(bundle.namelist()) != sorted(k+'/reconstruction.zip' for k in sites):
            raise ValueError('Archive inventory differs from scored frames')
        for row in rows:
            key = row['id']
            source = read(cache/'inputs'/(key+'.json'))
            ref = read(cache/'reference'/(key+'.json'))
            saved = sites[key]
            checked(Path(source['rgb']), row['source_hash'])
            for field in ('source_hash', 'geometry_pack_sha256'):
                if row[field] != source[field] or saved[field] != row[field]:
                    raise ValueError(f'{key} metadata mismatch for {field}')
            payload = bundle.read(key+'/reconstruction.zip')
            if sha256(payload).hexdigest() != row['pack_sha256'] or saved['pack_sha256'] != row['pack_sha256']:
                raise ValueError(f'{key} typed pack hash mismatch')
            decoded = decoded_pack(BytesIO(payload))
            evidence = {tuple(v[:3]):v[4] for v in saved['rows']}
            if decoded != {tuple(v[:3]):v[3] for v in saved['rows']} or len(saved['rows']) != len(decoded):
                raise ValueError(f'{key} overlay differs from decoded ID and coordinates')
            cells = list(map(tuple, source['cells']))
            if decoded.keys() != set(cells):
                raise ValueError(f'{key} placement geometry changed')
            world = list(map(tuple, ref['aligned']))
            predicted = {w:decoded[c] for c,w in zip(cells,world,strict=True)}
            assigned = {w for c,w in zip(cells,world,strict=True) if evidence[c] != 'fallback'}
            truth = {tuple(v[:3]):v[3] for v in ref['truth']}
            optional = set(map(tuple, ref['optional']))
            measured = score(predicted, truth, optional, assigned)
            baseline = score(dict.fromkeys(world, GEOMETRY_MATERIAL), truth, optional, set())
            if row['geometry'] != ref['geometry'] or baseline != row['baseline']:
                raise ValueError(f'{key} baseline changed')
            if measured != row['result'] or row['translation'] != ref['translation']:
                raise ValueError(f'{key} released scores differ from independently decoded template')
    for group in ('geometry','baseline','result'):
        if pooled_scores([r[group] for r in rows]) != summary['all'][group]:
            raise ValueError(f'Published aggregate differs for {group}')
    print('Verified every typed pack, site label, source binding and placement score on', len(rows), 'views')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'development', 'release', 'verify'])
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--development-run', type=Path, action='append', default=[],
                        help='Attach one measured training-only trial and its predictor source to a release')
    parser.add_argument('--method', default='auto')
    parser.add_argument('--out', type=Path, default=REVIEW)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.action == 'verify':
        verify(args.cache, args.out)
        return
    if args.action == 'development':
        if args.predictions is None:
            parser.error('development needs --predictions')
        result = development(args.cache, args.predictions)
        compact(args.predictions/'development.json', result)
        print({k:v['material'] for k,v in result['scores'].items()})
        return
    if args.action == 'release':
        if args.predictions is None:
            parser.error('release needs --predictions')
        release(args.cache, args.predictions, args.out, args.method, args.development_run)
        return
    jobs = list(SUITE.jobs([*SUITE.framed.SIZES, 'shelter']))
    with ProcessPoolExecutor(args.workers) as pool:
        keys = list(pool.map(prepare, [(j,args.cache) for j in jobs]))
    if len(set(keys)) != 204:
        raise ValueError('Incomplete 204-view suite')
    save(args.cache / 'manifest.json', dict(frames=keys, geometry_cases_sha256=digest(BASE/'cases.json')))
    print('Prepared',len(keys),'source images and independent scoring references')


if __name__ == '__main__':
    main()
