"""Package the saved open-sky surface and counter handoffs for offline review."""
import argparse
from base64 import b64encode
from io import BytesIO
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read
from benchmark.expanded_scores import load_reference
from benchmark.floating_cloud_replay import ANCHOR_EXTENSION, ASSEMBLY, NATIVE_COUNTER, SHARED_ROWS, check_handoff, source_handoff
from benchmark.harbor_carrier_audit import native_mask
from benchmark.relative_face_round import SHARED

OUTPUT = ROOT / 'runs/shared-corner-viewer-01'
PRIOR = WORKTREES / 'relative-face-assembly/runs/relative-face-stage-review-01'


def filtered_overlays(scene, planes, output, renderer):
    source = np.asarray(Image.open(PRIOR / scene['source']).convert('RGB'))
    colors = source.copy()
    directions = source.copy()
    owner = np.zeros(source.shape[:2], dtype=np.uint8)
    active = [mask for mask in scene['masks'] if mask['id'] in planes]
    for mask in active:
        pixels = native_mask(planes[mask['id']], tuple(scene['size']))
        edge = renderer['border'](pixels)
        owner[pixels] = mask['index']
        for canvas, color in [(colors, mask['color']),
                              (directions, renderer['DIRECTION_COLORS'][mask['direction']])]:
            renderer['tint'](canvas, pixels, color)
            canvas[edge] = renderer['rgb'](color)
    for key, array in [('masks', colors), ('directions', directions)]:
        canvas = Image.fromarray(array)
        draw = ImageDraw.Draw(canvas)
        for mask in active[:12]:
            x, y = mask['pin']
            color = mask['color']
            if key == 'directions':
                color = renderer['DIRECTION_COLORS'][mask['direction']]
                if mask['normal']:
                    nx, ny, _ = mask['normal']
                    if nx*nx + ny*ny > .014:
                        end = (x + nx*52, y + ny*52)
                        draw.line([(x,y), end], fill=color, width=3)
                        draw.ellipse((end[0]-3,end[1]-3,end[0]+3,end[1]+3), fill=color)
            renderer['badge'](draw, (x, y), mask['label'], color, True)
        canvas.save(output / scene['assets'][key], quality=91, subsampling=0)
    # The prior depth panel uses the same field and colour scale and has no badges.
    depth = np.array(Image.open(PRIOR / scene['assets']['depth']).convert('RGB'))
    depth[owner == 0] = source[owner == 0]
    Image.fromarray(depth).save(output / scene['assets']['depth'], quality=91, subsampling=0)
    encoded = BytesIO()
    Image.fromarray(owner).save(encoded, format='PNG')
    return 'data:image/png;base64,' + b64encode(encoded.getvalue()).decode('ascii')


def main(output=OUTPUT):
    renderer = runpy.run_path(str(PRIOR / 'render_review.py'))
    raw = (PRIOR / 'data.js').read_text(encoding='utf-8')
    stages = json.loads(raw.split('=', 1)[1].strip().rstrip(';'))
    archive = WORKTREES / 'shared-corner-rows/runs'
    pitch_path = archive / 'native-crop-audit-01/pitch-controls.json'
    pitch = read(pitch_path) if pitch_path.exists() else {}
    line_dir = archive / 'castle-feedback-01/rows'
    line_report = read(line_dir / 'measurements.json') if (line_dir / 'measurements.json').exists() else {}
    audit_dir = pitch_path.parent
    audit_path = audit_dir / 'castle-S04-distortion.json'
    audit = read(audit_path) if audit_path.exists() else {}
    trim_audit = read(ROOT / 'benchmark/reviews/sky-mask-feedback-01/report.json')
    trim_cases = {frame['id']: frame for frame in trim_audit['frames']}
    cases = {row['id']: row for row in read(SHARED / 'fixtures.json')['cases']}
    scenes = []
    for scene in stages:
        ident = scene['id']
        original = WORKTREES / 'floating-build-local-grid/runs/floating-build-grid-01/sky' / ident / 'source-handoff'
        handoff = source_handoff(ident)
        manifest = read(handoff / 'build-manifest.json')
        assert manifest['stages']['background_filter'] == 'border-connected-whole-sky-v1'
        rejected = set(manifest['surface_finder']['boundary_settings']['background_filter']['rejected_ids'])
        original_planes = {p['id']: p for p in read(original / 'surfaces.json')['planes']}
        planes = {p['id']: p for p in read(handoff / 'surfaces.json')['planes']}
        assert planes.keys() == original_planes.keys() - rejected
        original_counts = {r['id']: r for r in read(original / 'counts.json')['surfaces']}
        counts = {r['id']: r for r in read(handoff / 'counts.json')['surfaces']}
        labels = {mask['id']: mask['label'] for mask in scene['masks']}
        seam_report = read(SHARED_ROWS / 'sky' / ident / 'seams.json')
        joins = [dict(seam, donor=labels[seam['donor_id']], receiver=labels[seam['receiver_id']])
                 for seam in seam_report['seams']]
        sha = digest(PRIOR / scene['source'])
        assert sha == manifest['image_sha256'] == seam_report['source_sha256']
        assert sha == read(original / 'build-manifest.json')['image_sha256'] == trim_cases[ident]['source_sha256']
        native_path = NATIVE_COUNTER / 'sky' / ident / 'native-evidence.json'
        evidence = read(native_path)
        check_handoff(seam_report, handoff, sha)
        check_handoff(evidence, handoff, sha)
        if evidence['source_sha256'] != sha:
            raise ValueError(f'Native crop evidence belongs to a different source: {ident}')
        native = {r['id']: r for r in evidence.get('surfaces', [])}
        revision = digest(handoff / 'surfaces.json') + '/' + digest(handoff / 'counts.json')
        annotation_revision = digest(original / 'surfaces.json') + '/' + digest(original / 'counts.json')
        folder = output / ident / 'samples'
        folder.mkdir(parents=True, exist_ok=True)
        masks = []
        for mask in scene['masks']:
            removed = mask['id'] in rejected
            assets = original if removed else handoff
            plane = (original_planes if removed else planes)[mask['id']]
            counter = (original_counts if removed else counts).get(mask['id'], {})
            images = {}
            for key, path in [('crop', plane.get('source_crop', {}).get('path')),
                              ('context', plane.get('context_crop', {}).get('path')),
                              ('sample', counter.get('sample'))]:
                if path:
                    dest = folder / f'{mask["label"]}-{key}.png'
                    shutil.copyfile(assets / path, dest)
                    images[key] = dest.relative_to(output).as_posix()
            reference_sample = None
            level = line_report.get(ident, {}).get(mask['label'])
            sky_note = 'Rejected by the cloud filter. The entire proposal matched border-connected sky.' if removed else None
            if not removed and mask['id'] in trim_cases[ident]['mixed_trim']:
                sky_note = 'Mixed sky and surface evidence. The whole-proposal cloud filter retained the complete mask.'
            if level and level['slope'] is not None:
                target = folder / f'{mask["label"]}-row-level.png'
                shutil.copyfile(line_dir / ident / f'{mask["label"]}.png', target)
                images['level'] = target.relative_to(output).as_posix()
            if audit.get('frame') == ident and audit['surface_mask'] == mask['id']:
                assert audit['source']['source_sha256'] == sha
                target = folder / f'{mask["label"]}-reference.png'
                shutil.copyfile(audit_dir / 'castle-S04-reference-rectified.png', target)
                reference_sample = target.relative_to(output).as_posix()
            masks.append(dict(id=mask['id'], index=mask['index'], label=mask['label'],
                              rejected=removed, dimensions=(counter.get('dimensions') or {}).get('blocks', [None, None]), direction=mask['direction'], images=images,
                              window=plane.get('context_crop', {}).get('xyxy') or mask['bounds'],
                              cropBox=plane.get('source_crop', {}).get('xyxy'),
                              pitch=pitch.get(ident, {}).get(mask['id']),
                              native=native.get(mask['id']),
                              lineReview=level,
                              skyNote=sky_note,
                              referenceSample=reference_sample,
                              axes=counter.get('evidence', {}).get('axes', []),
                              quad=counter.get('quad'), reason=counter.get('reason')))
        shutil.copyfile(PRIOR / scene['source'], output / scene['source'])
        hit = filtered_overlays(scene, planes, output, renderer)
        panels = dict(directions=scene['assets']['directions'], depth=scene['assets']['depth'],
                      grids=scene['methods']['image']['assets']['grids'])
        shutil.copyfile(PRIOR / panels['grids'], output / panels['grids'])
        rows = SHARED_ROWS / 'sky' / ident
        panels['joins'] = f'{ident}/grids.jpg'
        shutil.copyfile(rows / 'source-overlay.jpg', output / panels['joins'])
        for key, mode, scored, charts in [
            ('placed', 'row-transfer', rows, rows / 'charts.json'),
            ('cropPlaced', 'registered-seam', ASSEMBLY / 'registered-seam/sky' / ident, NATIVE_COUNTER / 'registered-seam/sky' / ident / 'charts.json'),
            ('connectedPlaced', 'connected-atlas', ASSEMBLY / 'connected-atlas/sky' / ident,
             ANCHOR_EXTENSION / 'connected/sky' / ident / 'charts.json')
        ]:
            receipt = read(scored / 'score.json')
            check_handoff(read(charts), handoff, sha)
            assert receipt['chart_sha256'] == digest(charts)
            assert receipt['source_sha256'] == sha and receipt['pack_sha256'] == digest(scored / 'reconstruction.zip')
            aligned = {tuple(xyz): mat for xyz, mat in read(scored / 'placements.json')['aligned']}
            frame = Path(cases[ident]['frame'])
            reference = load_reference(cases[ident])
            assert receipt['score']['fixed']['geometry']['strict']['correct'] == len(aligned.keys() & reference.truth.keys())
            with np.load(frame / 'labels.npz', allow_pickle=False) as native_labels:
                panels[key] = renderer['placements'](
                    Image.open(PRIOR / scene['source']).convert('RGB'), native_labels['face_id'],
                    native_labels['depth_z'], read(frame / 'labels.json'), reference, aligned,
                    output / ident, mode)
        scenes.append(dict(id=ident, title=scene['title'], sha=sha, revision=revision, source=scene['source'],
                           annotationRevision=annotation_revision, panels=panels, size=scene['size'], joins=joins,
                           masksImage=scene['assets']['masks'], hit=hit, masks=masks))
    (output / 'sample-data.js').write_text('const scenes=' + json.dumps(scenes) + ';', encoding='utf-8')
    shutil.copyfile(ROOT / 'benchmark/surface_samples.html', output / 'surface-samples.html')
    for name in ('pillar-grid.png', 'placement-comparison.jpg'):
        shutil.copyfile(ROOT / 'benchmark/reviews/garden-feedback-01' / name, output / name)
    active = sum(not mask['rejected'] for scene in scenes for mask in scene['masks'])
    total = sum(len(scene['masks']) for scene in scenes)
    print(f"{len(scenes)} builds, {active} active surfaces, {total-active} archived proposals")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    main(parser.parse_args().output)
