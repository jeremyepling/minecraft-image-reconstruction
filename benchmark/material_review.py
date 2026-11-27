"""Build the offline material review from decoded exports and visible reference labels."""
import argparse
from base64 import b64encode
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark import typed_structures as typed
from benchmark.mask_blocks import decoded_pack
from benchmark.pavement_opening_eval import visible_faces
from benchmark.texture_camera_eval import checked, read
from benchmark.thin_face_placement_run import ref_inputs
from training.visible_labels import visible_labels


@lru_cache(maxsize=1)
def release(directory):
    summary = read(directory / 'summary.json')
    sites = json.loads(gzip.decompress(checked(directory / 'sites.json.gz', summary['sites_sha256']).read_bytes()))
    cases = read(checked(directory / 'cases.json', summary['cases_sha256']))
    return {r['id']: r for r in cases}, sites


def material_color(name):
    # Stable categorical swatches so the same ID keeps its color across source images.
    return tuple(55 + int(x) * 190 // 255 for x in sha256(name.encode()).digest()[:3])


@lru_cache(maxsize=1)
def structure_names():
    layouts = read(typed.viewer.ROOT / 'facade-placement-01/worst/ranking.json')['all_layouts']
    counts = Counter(r['name'] for r in layouts)
    result = {}
    for layout in layouts:
        name = layout['name']
        if counts[name] > 1:
            capture, sep, _ = layout['frame_ids'][0].partition('--')
            name += ' · ' + (capture.replace('-26.3', '') if sep else 'original')
        result.update(dict.fromkeys(layout['frame_ids'], name))
    return result


def overlay(path, labels, ids, colors, size):
    palette = np.zeros((len(labels['faces']) + 1, 4), np.uint8)
    for face in labels['faces']:
        point = tuple(face['xyz_world'])
        color = colors.get(point)
        if color:
            light = (0.86, 1., 0.75)[np.argmax(np.abs(face['normal_world']))]
            palette[face['id']] = [*(int(v * light) for v in color), 228]
    # Draw the foremost face once. No overlap ordering is left to the browser.
    pixels = palette[ids]
    edge = np.zeros(ids.shape, bool)
    edge[1:] |= ids[1:] != ids[:-1]
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    pixels[edge, :3] //= 2
    image = Image.fromarray(pixels).resize(size, Image.Resampling.NEAREST)
    image.convert('P', palette=Image.Palette.ADAPTIVE, colors=255).save(path, optimize=True)


def hit_map(labels, ids, size, numbers):
    # Encode the foremost block at every displayed pixel. A data URL allows the
    # offline viewer to sample a canvas even when opened directly from disk.
    lookup = np.zeros(len(labels['faces']) + 1, np.int32)
    for face in labels['faces']:
        lookup[face['id']] = numbers[tuple(face['xyz_world'])]
    codes = np.array(Image.fromarray(lookup[ids]).resize(size, Image.Resampling.NEAREST))
    rgb = np.stack((codes % 256, codes // 256, np.zeros_like(codes)), axis=-1).astype(np.uint8)
    data = BytesIO()
    Image.fromarray(rgb).save(data, format='PNG', optimize=True)
    return 'data:image/png;base64,' + b64encode(data.getvalue()).decode('ascii')


def paint(task):
    directory, out, job = task
    split, frame, _, _ = job
    key = frame['id']
    cases, all_sites = release(directory)
    row, sites = cases[key], all_sites[key]
    rgb = checked(typed.source_path(split, frame), row['source_hash'])
    if frame['files']['rgb.png'] != row['source_hash'] or any(
        sites[k] != row[k] for k in ('source_hash', 'geometry_pack_sha256', 'pack_sha256')
    ):
        raise ValueError(f'Mismatched material artifacts for {key}')
    with ZipFile(directory / 'typed-exports.zip') as archive:
        pack = archive.read(key + '/reconstruction.zip')
    if sha256(pack).hexdigest() != row['pack_sha256']:
        raise ValueError(f'Mismatched typed export for {key}')
    cells = decoded_pack(BytesIO(pack))
    site_rows = {tuple(r[:3]): tuple(r[3:]) for r in sites['rows']}
    if cells != {p: v[0] for p, v in site_rows.items()}:
        raise ValueError(f'Sites differ from decoded block IDs for {key}')
    _, _, transform, _ = ref_inputs(job)
    world = typed.aligned_sites(list(cells), transform, row['translation'])
    predicted = dict(zip(world, cells.values(), strict=True))
    labels, _ = visible_faces(split, frame)
    truth = {tuple(b['xyz_world']): b['block_id'] for b in labels['blocks']}
    exact = {p for p in predicted.keys() & truth.keys() if predicted[p] == truth[p]}
    if len(exact) != row['result']['material']['correct']:
        raise ValueError(f'Material score differs from overlay for {key}')
    camera_path = rgb.with_name('camera.json') if rgb.name == 'rgb.png' else rgb.with_suffix('.json')
    pose = read(checked(camera_path, frame['files']['camera.json']))
    dummy = dict(block_id='minecraft:smooth_stone', state='Block{minecraft:smooth_stone}')
    all_positions = truth.keys() | predicted.keys()
    mask, ids, _ = visible_labels(pose, dict.fromkeys(all_positions, dummy), (0, 0, 0))
    stem = row['source_hash'][:16] + '-' + row['pack_sha256'][:12]
    assets = out / 'viewer-assets'
    with Image.open(rgb) as source:
        source.thumbnail((960, 640), Image.Resampling.LANCZOS)
        size = source.size
        source.convert('RGB').save(assets / (stem + '.jpg'), quality=85, optimize=True)
    colors = {p: (52, 207, 133) if p in exact else (255, 181, 45) if p in truth and p in predicted
              else (65, 153, 255) if p in truth else (247, 77, 88) for p in all_positions}
    overlay(assets / (stem + '-errors.png'), mask, ids, colors, size)
    local = dict(zip(world, cells, strict=True))
    positions = sorted(all_positions)
    numbers = {p: i for i, p in enumerate(positions, 1)}
    names = [None, *sorted(set(truth.values()) | set(predicted.values()))]
    name_id = {name: i for i, name in enumerate(names)}
    lookup = dict(names=names, blocks=[None, *[[p, local.get(p), name_id[truth.get(p)], name_id[predicted.get(p)]]
                                               for p in positions]], errors=hit_map(mask, ids, size, numbers))
    if predicted:
        pmask, pids, _ = visible_labels(pose, dict.fromkeys(predicted, dummy), (0, 0, 0))
        overlay(assets / (stem + '-ids.png'), pmask, pids, {p: material_color(v) for p,v in predicted.items()}, size)
        lookup['ids'] = hit_map(pmask, pids, size, numbers)
    else:
        Image.new('RGBA', size).save(assets / (stem + '-ids.png'))
    script = 'window.blockInspector(' + json.dumps(stem) + ',' + json.dumps(lookup, separators=(',', ':')) + ');'
    (assets / (stem + '-inspect.js')).write_text(script, encoding='utf-8')
    wrong = Counter((truth[p], predicted[p]) for p in (predicted.keys() & truth.keys()) - exact)
    return dict(id=key, layout=structure_names()[key], asset=stem, geometry=row['geometry'], result=row['result'],
                baseline=row['baseline'], geometryHash=row['geometry_pack_sha256'],
                palette=[[v, n, '#%02x%02x%02x' % material_color(v)] for v,n in Counter(predicted.values()).most_common()],
                wrong=[[a,b,n] for (a,b),n in wrong.most_common(5)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=typed.REVIEW)
    parser.add_argument('--out', type=Path, help='Write viewer files here; defaults to the scored release')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    jobs = list(typed.SUITE.jobs(typed.SUITE.framed.EVALUATION))
    if len(jobs) != 104:
        raise ValueError('Expected the complete fixed suite')
    out = args.out or args.release
    (out / 'viewer-assets').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(paint, [(args.release, out, j) for j in jobs]))
    summary = read(args.release / 'summary.json')
    data = dict(views=rows, cohorts=summary['cohorts'], independent=summary['other_unseen'])
    script = 'const materialReview = ' + json.dumps(data, separators=(',', ':')).replace('<', r'\u003c') + ';'
    (out / 'viewer-data.js').write_text(script, encoding='utf-8')
    template = Path(__file__).with_name('material_review.html')
    (out / 'viewer.html').write_text(template.read_text(encoding='utf-8'), encoding='utf-8')


if __name__ == '__main__':
    main()
