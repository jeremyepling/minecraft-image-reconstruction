"""Apply the ordinary grouped Minecraft material matchers to visible grid readings."""
from dataclasses import dataclass
from pathlib import Path

from benchmark.camera_round import ROOT, digest, read
from benchmark.template_block_seam import ATLAS_SHA256
from engine.atlas_count import INDEX_SHA256, load_atlas, rank_tile
from engine.block_face_groups import assign_block_materials
from engine.blocker_shape import ShapeMatcher
from engine.face_evidence import MIN_FACE_WIDTH, effective_width
from engine.surface_model import canonical
from engine.surface_tiles import complete_footprints, tile_sample


@dataclass
class MaterialMatcher:
    method: str
    model: object
    provenance: dict

    def rank(self, tile):
        if self.method == 'atlas':
            return rank_tile(tile, self.model)
        result = self.model.rank_patch(tile['rgb'])
        return dict(block_id='minecraft:' + result['block_id'])


def create_matcher(method, atlas_path=None):
    if method not in ('shape', 'atlas'):
        raise ValueError(f'Unknown material matcher {method}')
    if atlas_path is None:
        candidates = [ROOT/'runs/26.3/atlas', ROOT/'runs/atlas',
                      ROOT.parent.parent/'workspace/builds/mcir-port-validation/final-cli-26.3/atlas']
        atlas_path = next((p for p in candidates if (p/'atlas.json').is_file()), None)
        if atlas_path is None:
            raise FileNotFoundError('Supply the calibrated Java 26.3 atlas directory')
    path = Path(atlas_path)
    catalog = read(path/'atlas.json')
    atlas_sha = digest(path/catalog['atlas'])
    placement_sha = digest(path/'placement-index.json')
    if catalog['version'] != '26.3' or atlas_sha != ATLAS_SHA256 or placement_sha != INDEX_SHA256:
        raise ValueError('Material evaluation requires the frozen Java 26.3 atlas and placement index')
    model = ShapeMatcher(path) if method == 'shape' else load_atlas(path)
    modules = ('blocker_shape' if method == 'shape' else 'atlas_count',
               'block_face_groups', 'face_evidence', 'surface_tiles', 'surface_model')
    provenance = dict(method=method, version='26.3', atlas_sha256=atlas_sha,
                      atlas_json_sha256=digest(path/'atlas.json'), placement_index_sha256=placement_sha,
                      client_sha256=catalog['recipe']['client_sha256'], minimum_source_width=MIN_FACE_WIDTH,
                      group_method='convex-shared-edges', sample_size=32,
                      source_sha256={m:digest(ROOT/'engine'/(m+'.py')) for m in modules})
    return MaterialMatcher(method, model, provenance)


def predict_ids(image, parts, samples, readings, matcher):
    """Select and rank one source face per block; include unresolved and inherited faces."""
    faces = []
    for part, sample, reading in zip(parts, samples, readings, strict=True):
        for cell, quad in complete_footprints(sample, reading):
            faces.append(dict(proposal=part['id'], cell=cell, quad=canonical(quad),
                              effective_width_px=effective_width(quad), normal_camera=part.get('normal_camera')))
    assign_block_materials(image, faces, matcher.rank)
    for row in faces:
        row['block_id'] = row.pop('material', {}).get('block_id')
    return faces


def match_known(image, quad, matcher):
    """Single reference face diagnostic, with the same sampling and source-width gate."""
    quad = canonical(quad)
    if effective_width(quad) < MIN_FACE_WIDTH:
        return None
    return matcher.rank(tile_sample(image, quad)).get('block_id')
