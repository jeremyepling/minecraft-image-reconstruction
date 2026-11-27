"""Frozen-input adapter for the production surface-stage join, including recount and export."""
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark.texture_camera_eval import STABLE
from engine import guarded_wall_outlines, surface_model
from engine.atlas_plane_depth import place_ranked
from engine.surface_perimeter import with_outline


def _measure(row):
    return dict(id=row['id'], counts=row.get('counts'), dimensions=row.get('dimensions'),
                complete=len(row.get('visible_cells', [])), small=len(row.get('under_five_px_faces', [])))


def predict(image, parts, counts, camera, prior):
    changed, events = guarded_wall_outlines.refine(image, parts)
    if not events:
        return prior, dict(events=[])
    lookup = {c['id']:c for c in counts['surfaces']}
    fresh = [with_outline(p) for p in changed if p['id'] not in lookup]
    atlas_path = STABLE/'runs/26.3/atlas'
    method = next(iter({lookup[key]['method'] for e in events for key in e['inputs']}))
    atlas = None
    if 'atlas_phase' in counts:
        from engine.atlas_count import load_atlas
        atlas = load_atlas(atlas_path,'26.3')
    with TemporaryDirectory() as tmp:
        recomputed = surface_model._count_regions(image, fresh, '26.3', method, atlas, Path(tmp))
    lookup.update((r['id'],r) for r in recomputed['surfaces'])
    for e in events:
        e['before_counts'] = [_measure(lookup[key]) for key in e['inputs']]
        e['after_count'] = _measure(lookup[e['output']])
    next_counts = dict(counts,surfaces=[lookup[p['id']] for p in changed])
    cells, _ = place_ranked(image,'26.3',changed,next_counts,camera,atlas_path=atlas_path)
    return cells, dict(events=events)
