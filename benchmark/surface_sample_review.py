"""Package the saved open-sky surface and counter handoffs for offline review."""
import json
import runpy
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read
from benchmark.expanded_scores import load_reference
from benchmark.relative_face_round import SHARED

OUTPUT = ROOT / 'runs/shared-corner-viewer-01'
PRIOR = WORKTREES / 'relative-face-assembly/runs/relative-face-stage-review-01'


def main():
    raw = (PRIOR / 'data.js').read_text(encoding='utf-8')
    stages = json.loads(raw.split('=', 1)[1].strip().rstrip(';'))
    archive = WORKTREES / 'shared-corner-rows/runs'
    pitch_path = archive / 'native-crop-audit-01/pitch-controls.json'
    pitch = read(pitch_path) if pitch_path.exists() else {}
    line_dir = archive / 'castle-feedback-01/rows'
    line_report = read(line_dir / 'measurements.json') if (line_dir / 'measurements.json').exists() else {}
    sky_path = WORKTREES / 'sky-mask-feedback/runs/sky-mask-feedback-01/report.json'
    sky_cases = {s['id']: s for s in read(sky_path)['frames']} if sky_path.exists() else {}
    audit_dir = pitch_path.parent
    audit_path = audit_dir / 'castle-S04-distortion.json'
    audit = read(audit_path) if audit_path.exists() else {}
    cases = {row['id']: row for row in read(SHARED / 'fixtures.json')['cases']}
    scenes = []
    for scene in stages:
        ident = scene['id']
        handoff = WORKTREES / 'floating-build-local-grid/runs/floating-build-grid-01/sky' / ident / 'source-handoff'
        planes = {p['id']: p for p in read(handoff / 'surfaces.json')['planes']}
        counts = {r['id']: r for r in read(handoff / 'counts.json')['surfaces']}
        joins = read(ROOT / 'runs/shared-corner-rows-01/sky' / ident / 'seams.json')['seams']
        sha = digest(PRIOR / scene['source'])
        assert sha == read(handoff / 'build-manifest.json')['image_sha256']
        sky_case = sky_cases.get(ident, {})
        assert sky_case.get('source_sha256', sha) == sha
        native_path = WORKTREES / 'surface-crop-first/runs/surface-crop-first-01/sky' / ident / 'native-evidence.json'
        evidence = read(native_path) if native_path.exists() else {}
        if evidence.get('source_sha256', sha) != sha:
            raise ValueError(f'Native crop evidence belongs to a different source: {ident}')
        native = {r['id']: r for r in evidence.get('surfaces', [])}
        revision = digest(handoff / 'surfaces.json') + '/' + digest(handoff / 'counts.json')
        folder = OUTPUT / ident / 'samples'
        folder.mkdir(parents=True, exist_ok=True)
        masks = []
        for mask in scene['masks']:
            plane = planes[mask['id']]
            counter = counts.get(mask['id'], {})
            images = {}
            for key, path in [('crop', plane.get('source_crop', {}).get('path')),
                              ('context', plane.get('context_crop', {}).get('path')),
                              ('sample', counter.get('sample'))]:
                if path:
                    dest = folder / f'{mask["label"]}-{key}.png'
                    shutil.copyfile(handoff / path, dest)
                    images[key] = dest.relative_to(OUTPUT).as_posix()
            reference_sample = None
            level = line_report.get(ident, {}).get(mask['label'])
            trim = sky_case.get('mixed_trim', {}).get(mask['id'])
            sky_removed = mask['id'] in sky_case.get('removed_ids', [])
            sky_note = None
            if trim:
                sky_note = f"Cloud filter trial removed {trim['rejected']:,} background pixels and preserved {trim['retained']} pixels."
            if sky_removed:
                sky_note = 'Cloud filter trial excluded this entire mask. The reference records no visible build pixels in it.'
            if level and level['slope'] is not None:
                target = folder / f'{mask["label"]}-row-level.png'
                shutil.copyfile(line_dir / ident / f'{mask["label"]}.png', target)
                images['level'] = target.relative_to(OUTPUT).as_posix()
            if audit.get('frame') == ident and audit['surface_mask'] == mask['id']:
                assert audit['source']['source_sha256'] == sha
                target = folder / f'{mask["label"]}-reference.png'
                shutil.copyfile(audit_dir / 'castle-S04-reference-rectified.png', target)
                reference_sample = target.relative_to(OUTPUT).as_posix()
            masks.append(dict(id=mask['id'], index=mask['index'], label=mask['label'],
                              dimensions=mask['dims'], direction=mask['direction'], images=images,
                              window=plane.get('context_crop', {}).get('xyxy') or mask['bounds'],
                              cropBox=plane.get('source_crop', {}).get('xyxy'),
                              pitch=pitch.get(ident, {}).get(mask['id']),
                              native=native.get(mask['id']),
                              lineReview=level,
                              skyNote=sky_note,
                              referenceSample=reference_sample,
                              axes=counter.get('evidence', {}).get('axes', []),
                              quad=counter.get('quad'), reason=counter.get('reason')))
        for path in [scene['source'], scene['assets']['masks']]:
            shutil.copyfile(PRIOR / path, OUTPUT / path)
        panels = dict(directions=scene['assets']['directions'], depth=scene['assets']['depth'],
                      grids=scene['methods']['image']['assets']['grids'])
        for path in panels.values():
            shutil.copyfile(PRIOR / path, OUTPUT / path)
        panels.update(joins=f'{ident}/grids.jpg', placed=f'{ident}/row-transfer-placed.jpg')
        for key, file in [('joins','grids.jpg'),('placed','row-transfer-placed.jpg')]:
            old = archive / 'shared-corner-viewer-01' / ident / file
            target = OUTPUT / panels[key]
            if old.is_file() and old.resolve()!=target.resolve():
                shutil.copyfile(old,target)
        for key, mode in [('cropPlaced','registered-seam'), ('connectedPlaced','connected-atlas')]:
            scored = ROOT / 'runs/relative-face-assembly-01' / mode / 'sky' / ident
            if not (scored/'placements.json').exists():
                continue
            receipt = read(scored / 'score.json')
            assert receipt['source_sha256'] == sha and receipt['pack_sha256'] == digest(scored / 'reconstruction.zip')
            aligned = {tuple(xyz): mat for xyz, mat in read(scored / 'placements.json')['aligned']}
            frame = Path(cases[ident]['frame'])
            reference = load_reference(cases[ident])
            assert receipt['score']['fixed']['geometry']['strict']['correct'] == len(aligned.keys() & reference.truth.keys())
            with np.load(frame / 'labels.npz', allow_pickle=False) as native_labels:
                panels[key] = runpy.run_path(str(PRIOR / 'render_review.py'))['placements'](
                    Image.open(PRIOR / scene['source']).convert('RGB'), native_labels['face_id'],
                    native_labels['depth_z'], read(frame / 'labels.json'), reference, aligned,
                    OUTPUT / ident, mode)
        scenes.append(dict(id=ident, title=scene['title'], sha=sha, revision=revision, source=scene['source'],
                           panels=panels, size=scene['size'], joins=joins,
                           masksImage=scene['assets']['masks'], hit=scene['hit'], masks=masks))
    (OUTPUT / 'sample-data.js').write_text('const scenes=' + json.dumps(scenes) + ';', encoding='utf-8')
    shutil.copyfile(ROOT / 'benchmark/surface_samples.html', OUTPUT / 'surface-samples.html')
    for name in ('pillar-grid.png', 'placement-comparison.jpg'):
        shutil.copyfile(ROOT / 'benchmark/reviews/garden-feedback-01' / name, OUTPUT / name)
    print(f"{len(scenes)} builds, {sum(len(s['masks']) for s in scenes)} surfaces")


if __name__ == '__main__':
    main()
