"""Package the saved open-sky surface and counter handoffs for offline review."""
import json
import shutil
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read

OUTPUT = ROOT / 'runs/shared-corner-viewer-01'
PRIOR = WORKTREES / 'relative-face-assembly/runs/relative-face-stage-review-01'


def main():
    raw = (PRIOR / 'data.js').read_text(encoding='utf-8')
    stages = json.loads(raw.split('=', 1)[1].strip().rstrip(';'))
    scenes = []
    for scene in stages:
        ident = scene['id']
        handoff = WORKTREES / 'floating-build-local-grid/runs/floating-build-grid-01/sky' / ident / 'source-handoff'
        planes = {p['id']: p for p in read(handoff / 'surfaces.json')['planes']}
        counts = {r['id']: r for r in read(handoff / 'counts.json')['surfaces']}
        joins = read(ROOT / 'runs/shared-corner-rows-01/sky' / ident / 'seams.json')['seams']
        sha = digest(PRIOR / scene['source'])
        assert sha == read(handoff / 'build-manifest.json')['image_sha256']
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
            masks.append(dict(id=mask['id'], index=mask['index'], label=mask['label'],
                              dimensions=mask['dims'], direction=mask['direction'], images=images,
                              window=plane.get('context_crop', {}).get('xyxy') or mask['bounds'],
                              axes=counter.get('evidence', {}).get('axes', []),
                              quad=counter.get('quad'), reason=counter.get('reason')))
        for path in [scene['source'], scene['assets']['masks']]:
            shutil.copyfile(PRIOR / path, OUTPUT / path)
        panels = dict(directions=scene['assets']['directions'], depth=scene['assets']['depth'],
                      grids=scene['methods']['image']['assets']['grids'])
        for path in panels.values():
            shutil.copyfile(PRIOR / path, OUTPUT / path)
        panels.update(joins=f'{ident}/grids.jpg', placed=f'{ident}/row-transfer-placed.jpg')
        scenes.append(dict(id=ident, title=scene['title'], sha=sha, revision=revision, source=scene['source'],
                           panels=panels, size=scene['size'], joins=joins,
                           masksImage=scene['assets']['masks'], hit=scene['hit'], masks=masks))
    (OUTPUT / 'sample-data.js').write_text('const scenes=' + json.dumps(scenes) + ';', encoding='utf-8')
    shutil.copyfile(ROOT / 'benchmark/surface_samples.html', OUTPUT / 'surface-samples.html')
    print(f"{len(scenes)} builds, {sum(len(s['masks']) for s in scenes)} surfaces")


if __name__ == '__main__':
    main()
