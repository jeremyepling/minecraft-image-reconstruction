"""Read back geometry-only structure packs built from frozen masks and measured counts."""
from collections import Counter
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image

from benchmark.camera_metrics import registration
from benchmark.camera_round import ROOT, digest, read, save
from benchmark.mask_blocks import SETTINGS, _blocks, _carriers, _footprints, _score, _write_read, registered
from engine.camera_grid import predict


def _parts(regions, measurements, method):
    """Accept only image-derived fields; labels on scored measurement rows never cross this boundary."""
    parts, phases = [], Counter(resolved=0, partial=0, unresolved=0)
    for region, row in zip(regions, measurements, strict=True):
        if row['id'] != region['id']:
            raise ValueError('Count measurements must identify the original surface')
        measured = row['measures'][method]
        counts = measured['counts']
        # The existing lift needs two measured axes or a neighbor with known scale.
        pair = counts if all(x is not None for x in counts) else None
        parts.append(dict(id=region['id'], support=region['support'], quad=row['quad'], pitch=dict(counts=pair)))
        phase = measured.get('phase') or [None, None]
        known = sum(x is not None for x in phase)
        phases[('unresolved', 'partial', 'resolved')[known]] += 1
    return parts, dict(phases)


def _totals(pictures, methods):
    summary = {}
    for method in methods:
        rows = [p['methods'][method] for p in pictures]
        totals = dict(nonempty_pictures=sum(r['score']['serialized']>0 for r in rows))
        for name in ('score', 'regions', 'phase'):
            values = Counter()
            for row in rows:
                values.update(row[name])
            totals[name] = dict(values)
        summary[method] = totals
    return summary


def _archive_pack(archive, member, path):
    # Preserve every exported file byte. Fix only the ZIP timestamps for reproducible artifacts.
    packed = BytesIO()
    with ZipFile(path) as original, ZipFile(packed, 'w') as dest:
        for name in original.namelist():
            dest.writestr(ZipInfo(name), original.read(name), compress_type=ZIP_DEFLATED)
    archive.writestr(ZipInfo(member), packed.getvalue(), compress_type=ZIP_DEFLATED)


def export_counts(results, source_rows, lookup, out, selected):
    """Compare counters through the unchanged lift, occupancy rules and Minecraft NBT exporter."""
    from benchmark.count_round import FINDER, cached_regions

    methods = list(dict.fromkeys(('legacy-repeat', selected)))
    records = {r['key']: r for r in source_rows}
    frozen = read(FINDER/'manifest.json')
    pictures = []
    archive_path = Path(out)/'geometry-packs.zip'
    with TemporaryDirectory() as folder, ZipFile(archive_path, 'x', compression=ZIP_DEFLATED) as archive:
        for result in results:
            key = result['key']
            case = lookup[key]
            source = case.path('image')
            if digest(source) != result['sha256']:
                raise ValueError(f'Unexpected screenshot for {key}')
            regions = cached_regions(records[key], frozen)
            with Image.open(source) as src:
                image = Image.fromarray(np.asarray(src.convert('RGB')))
            camera = predict(image, '26.3', SETTINGS['camera'])
            P = None if camera['P'] is None else np.asarray(camera['P'])
            masks = _footprints(regions, image.size, True)
            cache, emitted, reports = {}, {}, {}
            for method in methods:
                parts, phase = _parts(regions, result['surfaces'], method)
                carriers = _carriers(image, parts, masks, P)
                blocks, outcomes, details = _blocks(masks, carriers, P, cache)
                filename = method+'.zip'
                local_path = Path(folder)/filename
                emitted[method] = _write_read(local_path, blocks)
                member = key+'/'+filename
                _archive_pack(archive, member, local_path)
                reports[method] = dict(pack=member, regions=outcomes, phase=phase, details=details)

            # The declared origin is arbitrary. Align both packs from their shared predicted camera
            # only after writing and decoding every block, then compare visible coordinates.
            pose = read(case.path('pose'))
            target = {tuple(face['xyz']) for face in read(case.path('labels'))['faces']}
            T = np.eye(4) if P is None else registration(P, pose)
            for method in methods:
                reports[method]['score'] = _score(registered(emitted[method], T), target)
            pictures.append(dict(key=key, image_sha256=result['sha256'], cohort=result['cohort'], layout=result['layout'],
                                 camera_status=camera['scale_status'], methods=reports))
    summary = dict(selected=selected, methods=methods, all=_totals(pictures,methods), schema=1,
                   cohorts={c:_totals([p for p in pictures if p['cohort']==c],methods)
                            for c in sorted({p['cohort'] for p in pictures})},
                   camera_status=dict(Counter(p['camera_status'] for p in pictures)),
                   pictures=pictures, packs=archive_path.name, pack_archive_sha256=digest(archive_path),
                   settings={**SETTINGS, 'pitch':'per-arm measured counts on the identical sampling chart'},
                   reference_use='After pack readback: register the arbitrary coordinate frame and score visible blocks.',
                   phase_use='Camera grid supplies lattice origin. Local texture phase is reported independently.',
                   material='minecraft:smooth_stone', in_game_verified=False,
                   export_implementation='benchmark/mask_blocks.py', export_source_sha256=digest(ROOT/'benchmark/mask_blocks.py'))
    save(Path(out)/'placed-geometry.json',summary)
    return summary
