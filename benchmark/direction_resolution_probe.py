"""Change only MoGe's token count in the final underside pass, on frozen wall masks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, __version__ as pillow_version

from benchmark.camera_round import ROOT, digest, read, revision, save
from benchmark.mask_metrics import aggregate, measure
from benchmark.normal_fields import cached_fields
from benchmark.normal_round import groups
from benchmark.split_tiny_lines import ARCHIVE as LINES
from benchmark.surface_reference import ARCHIVE as REFERENCE, recorded
from engine.geometry import bilinear
from engine.surface_fields import MODELS, SETTINGS, session
from engine.surface_strip_split import SETTINGS as STRIPS, refine


TOKENS = (1200, 2400)
SPLITS = ('development', 'previously_reviewed_regression')
WALL_RUN = ROOT / 'benchmark/reviews/separation-01'
SHARED = ROOT.parent / 'normal-round'


def infer_pair(job):
    """Receive the screenshot and immutable model assets, with no scoring labels."""
    image_path, sha, weights, fields, work = job
    if digest(image_path) != sha:
        raise ValueError(f'Screenshot mismatch: {image_path}')
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    image.thumbnail((SETTINGS['normal_max_side'],) * 2, Image.Resampling.LANCZOS)
    rgb = (np.asarray(image, dtype=np.float32) / 255.).transpose(2, 0, 1)[None]
    net = session('normal', weights)
    times, hashes = {}, {}
    for tokens in TOKENS:
        start = perf_counter()
        result, = net.run(['normal'], dict(image=rgb, num_tokens=np.array(tokens, dtype=np.int64)))
        times[tokens] = perf_counter() - start
        normal = result[0]
        if normal.shape != (image.height, image.width, 3) or not np.isfinite(normal).all():
            raise ValueError(f'Invalid normal map for {sha} at {tokens} tokens')
        hashes[tokens] = sha256(normal.tobytes()).hexdigest()
        if tokens == TOKENS[0]:
            cached = cached_fields(sha, fields, ('normal',))
            if cached is None or not np.array_equal(normal, cached['normal']):
                raise ValueError(f'Control differs from archived field: {sha}')
        else:
            np.save(work / (sha + '.npy'), normal, allow_pickle=False)
    return sha, dict(inference_seconds=times, array_sha256=hashes)


def underside_diagnostic(ref, normals, matches):
    """Scoring only. Sample each eligible underside's direction error in degrees."""
    data = []
    for rid, patch in ref['patches'].items():
        if not patch['eligible'] or patch['normal'][1] != -1:
            continue
        yy, xx = np.where(ref['id_map'] == rid)
        xy = (np.stack((xx, yy), axis=1) + .5) * 4
        entry = dict(id=rid, samples=len(xy), methods={})
        for tokens, field in normals.items():
            vectors = bilinear(field, xy * np.array(field.shape[1::-1]) / ref['image_size'])
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1.e-9)
            degrees = np.degrees(np.arccos(np.clip(vectors @ patch['normal_camera'], -1, 1)))
            entry['methods'][tokens] = dict(found=rid in matches[tokens],
                median_error_degrees=round(float(np.median(degrees)), 3),
                within_30_degrees=round(float(np.mean(degrees <= 30)), 4))
        data.append(entry)
    return data


def score_frame(row, case, source_manifest, lines, fields, work):
    sha = row['sha256']
    wall_path = WALL_RUN / row['predictions']
    if digest(wall_path) != source_manifest['artifacts'][row['predictions']]:
        raise ValueError(f'Wall mask changed: {wall_path}')
    if sha != lines[sha]['image_sha256']:
        raise ValueError('Line evidence mismatch')
    with gzip.open(wall_path, 'rt') as file:
        walls = json.load(file)['wall-depth-anything']
    maps = cached_fields(sha, fields)
    normals = {1200: maps['normal'], 2400: np.load(work / (sha + '.npy'), allow_pickle=False)}
    with Image.open(case.path('image')) as source:
        rgb = source.convert('RGB')
    predictions, timings = {}, {}
    for tokens, normal in normals.items():
        start = perf_counter()
        predictions[tokens] = refine(rgb, walls, {'mlsd': lines[sha]}, normal, maps['depth'])
        timings[tokens] = perf_counter() - start
    _, ref = recorded(case)
    scores, matches = {}, {}
    for tokens, parts in predictions.items():
        scores[tokens], pairs = measure(parts, ref, use_mask=True)
        matches[tokens] = {rid for _, rid in pairs}
    return dict(key=row['key'], layout=row['layout'], split=row['cohort'], sha256=sha,
        wall_archive_sha256=digest(wall_path), scores=scores, matched={t:sorted(m) for t,m in matches.items()},
        gained=sorted(matches[2400] - matches[1200]), lost=sorted(matches[1200] - matches[2400]),
        undersides=underside_diagnostic(ref, normals, matches), refinement_seconds=timings)


def summarize(rows):
    results = {}
    for tokens in TOKENS:
        summary = aggregate([r['scores'][tokens] for r in rows])
        summary['underside_found'] = sum(u['methods'][tokens]['found'] for r in rows for u in r['undersides'])
        summary['underside_eligible'] = sum(len(r['undersides']) for r in rows)
        summary['refinement_seconds'] = sum(r['refinement_seconds'][tokens] for r in rows)
        results[tokens] = summary
    return dict(frames=len(rows), methods=results, gained=sum(len(r['gained']) for r in rows),
                lost=sum(len(r['lost']) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=SHARED / '.deps' / MODELS['normal'][1])
    parser.add_argument('--fields', type=Path, default=SHARED / 'runs/normal-fields')
    parser.add_argument('--work', type=Path, default=ROOT / 'runs/direction-resolution-01')
    parser.add_argument('--out', type=Path, default=ROOT / 'benchmark/reviews/direction-resolution-01')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    import onnxruntime as ort
    code = revision()
    source = read(WALL_RUN / 'manifest.json')
    groups_by_cohort, _, fixtures = groups()
    if fixtures != source['fixtures'] or digest(REFERENCE) != source['reference_sha256']:
        raise ValueError('Frozen suite changed')
    if digest(LINES) != source['line_input_sha256']['mlsd']:
        raise ValueError('M-LSD evidence changed')
    if digest(WALL_RUN / 'cases.json') != source['artifacts']['cases.json']:
        raise ValueError('Frozen wall run changed')
    with gzip.open(LINES, 'rt') as f:
        lines = json.load(f)['frames']
    cases = {f'{c.floor}/{c.id}':c for cohort in groups_by_cohort.values() for c in cohort}
    records = read(WALL_RUN / 'cases.json')
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    fields, results = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split in SPLITS:
            cohort = [r for r in records if r['cohort'] == split]
            jobs = {}
            for row in cohort:
                sha = row['sha256']
                image = cases[row['key']].path('image')
                if digest(image) != sha:
                    raise ValueError(f'Fixture image differs: {image}')
                if sha not in fields:
                    jobs[sha] = (image, sha, args.weights, args.fields, args.work)
            fields.update(pool.map(infer_pair, jobs.values()))
            subset = [score_frame(r, cases[r['key']], source, lines, args.fields, args.work) for r in cohort]
            results.extend(subset)
            summary = summarize(subset)
            print(split, {t:s['matched'] for t,s in summary['methods'].items()}, flush=True)
    details = args.out / 'frames.json.gz'
    details.write_bytes(gzip.compress(json.dumps(results, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    manifest = dict(run_id='direction-resolution-01', code=code, version=source['version'], fixtures=fixtures,
        split=source['splits'], capture=source['capture'], fixed_before_development=True,
        method='Archived Wall Separator followed by frozen strip-mlsd. Only the final normal token count changes.',
        normal_tokens=TOKENS, inference_settings=SETTINGS, underside_settings=STRIPS,
        provider=SETTINGS['provider'], workers=args.workers, runtime=ort.__version__, pillow=pillow_version,
        numpy=np.__version__, normal_checkpoint=MODELS['normal'][1:], depth_checkpoint=MODELS['depth'][1:],
        line_archive_sha256=digest(LINES), reference_sha256=digest(REFERENCE),
        wall_manifest_sha256=digest(WALL_RUN / 'manifest.json'), frames='frames.json.gz',
        frames_sha256=digest(details), high_detail_fields='runs/direction-resolution-01/<image-sha256>.npy',
        control='Fresh 1200-token fields compared byte-for-byte with the signed per-image cache.',
        inference_timing='Per unique image, net.run only, two conditions in order; simultaneous jobs may contend.',
        per_image=fields, all=summarize(results),
        by_split={s:summarize([r for r in results if r['split'] == s]) for s in SPLITS},
        by_layout={k:summarize([r for r in results if r['layout'] == k]) for k in sorted({r['layout'] for r in results})},
        reproduce='python -m benchmark.direction_resolution_probe')
    save(args.out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
