"""Visible labels for opaque Java 26.3 full cubes and selected decorative shapes."""
from collections import Counter, defaultdict
from itertools import product
import io
import json
from pathlib import Path
import re
from typing import NamedTuple
from zipfile import ZipFile

import numpy as np
from PIL import Image

from benchmark.camera_labels import DIRECTIONS
from calibration.world import GROUND
from engine.atlas import discover, resource_id
from engine.visible_raster import rasterize
from training.visible_labels import _camera, _number_surfaces


MODEL_KINDS = ("fence_gate", "trapdoor", "fence", "wall", "slab", "stairs")
NORMALS = {name: np.eye(3, dtype=int)[axis] * sign
           for axis, pair in enumerate(DIRECTIONS)
           for name, sign in zip(pair, (-1, 1))}
STATE = re.compile(r"(?:Block\{([^}]+)\}|([^\[]+))(?:\[([^\]]+)\])?")
TILE_SIZE = 32


class Quad(NamedTuple):
    corners: np.ndarray
    normal: np.ndarray
    direction: str
    cull: tuple[int, int, int] | None


class Geometry(NamedTuple):
    kind: str
    quads: tuple[Quad, ...]


def properties_for(row):
    state = STATE.fullmatch(row["state"])
    if not state or resource_id(state[1] or state[2]) != resource_id(row["block_id"]):
        raise ValueError(f"State and block ID disagree: {row}")
    return dict(term.split("=", 1) for term in state[3].split(",")) if state[3] else {}


def matches(conditions, properties):
    if "OR" in conditions:
        return any(matches(part, properties) for part in conditions["OR"])
    if "AND" in conditions:
        return all(matches(part, properties) for part in conditions["AND"])
    return all(properties.get(key) in allowed.split("|") for key, allowed in conditions.items())


def model_rotation(spec):
    """Blockstate rotations act clockwise, first about X and then about Y."""
    x, y = (spec.get(axis, 0) for axis in "xy")
    if x % 90 or y % 90:
        raise ValueError(f"Non-orthogonal blockstate rotation: {spec}")
    sin_x, sin_y = (round(np.sin(np.deg2rad(-value))) for value in (x, y))
    cos_x, cos_y = (round(np.cos(np.deg2rad(-value))) for value in (x, y))
    rx = np.array(((1, 0, 0), (0, cos_x, -sin_x), (0, sin_x, cos_x)))
    ry = np.array(((cos_y, 0, sin_y), (0, 1, 0), (-sin_y, 0, cos_y)))
    return ry @ rx


def model_face(low, high, normal):
    axis = np.flatnonzero(normal)[0]
    sign = normal[axis]
    start = low.copy()
    start[axis] = high[axis] if sign > 0 else low[axis]
    u, v = ((high - low) * np.eye(3))[[((axis + i) % 3) for i in (1, 2)]]
    corners = np.stack((start, start + u, start + u + v, start + v))
    return corners if sign > 0 else corners[::-1]


def quad_signature(quads):
    return sorted((q.direction, tuple(q.cull or (0, 0, 0)), sorted(map(tuple, q.corners)))
                  for q in quads)


class VanillaModels:
    def __init__(self, archive):
        self.archive = archive
        self.documents = {}
        self.inherited = {}
        self.opaque = {}
        self.geometry = {}

    @staticmethod
    def path(folder, identifier, extension):
        namespace, key = resource_id(identifier).split(":", 1)
        return f"assets/{namespace}/{folder}/{key}.{extension}"

    def document(self, folder, identifier):
        path = self.path(folder, identifier, "json")
        if path not in self.documents:
            self.documents[path] = json.loads(self.archive.read(path))
        return self.documents[path]

    def model(self, identifier):
        if identifier not in self.inherited:
            data = self.document("models", identifier)
            textures, elements = self.model(data["parent"]) if "parent" in data else ({}, [])
            self.inherited[identifier] = ({**textures, **data.get("textures", {})},
                                          data.get("elements", elements))
        return self.inherited[identifier]

    def verify_opaque(self, texture, variables):
        seen = set()
        while isinstance(texture, str) and texture.startswith("#") and texture not in seen:
            seen.add(texture)
            texture = variables.get(texture[1:])
            if isinstance(texture, dict):
                texture = texture.get("sprite")
        if not isinstance(texture, str) or texture.startswith("#"):
            raise ValueError(f"Unresolved texture: {texture}")
        if texture not in self.opaque:
            content = self.archive.read(self.path("textures", texture, "png"))
            with Image.open(io.BytesIO(content)) as image:
                self.opaque[texture] = image.convert("RGBA").getchannel("A").getextrema() == (255, 255)
        if not self.opaque[texture]:
            raise ValueError(f"Unsupported non-opaque texture: {texture}")

    def quads(self, spec):
        textures, elements = self.model(spec["model"])
        rotation = model_rotation(spec)
        for element in elements:
            if element.get("rotation", {}).get("angle", 0):
                raise ValueError(f"Unsupported rotated element in {spec}")
            low, high = np.array([element["from"], element["to"]], dtype=float) / 16
            if np.any(low < 0) or np.any(high > 1) or np.any(high <= low):
                raise ValueError(f"Unsupported model bounds in {spec}")
            for direction, face in element["faces"].items():
                self.verify_opaque(face["texture"], textures)
                local_normal = NORMALS[direction]
                corners = (model_face(low, high, local_normal) - 0.5) @ rotation.T + 0.5
                normal = rotation @ local_normal
                axis = np.flatnonzero(normal)[0]
                direction = DIRECTIONS[axis][int(normal[axis] > 0)]
                cull = tuple(rotation @ NORMALS[face["cullface"]]) if "cullface" in face else None
                yield Quad(corners, normal, direction, cull)

    def resolve(self, row):
        key = (row["block_id"], row["state"])
        if key in self.geometry:
            return self.geometry[key]
        block = resource_id(row["block_id"])
        properties = properties_for(row)
        if properties.get("waterlogged") == "true":
            raise ValueError(f"Unsupported waterlogged state: {row['state']}")
        data = self.document("blockstates", block)
        if "variants" in data:
            specs = [spec for condition, spec in data["variants"].items()
                     if not condition or matches(dict(term.split("=") for term in condition.split(",")),
                                                 properties)]
            if len(specs) != 1:
                raise ValueError(f"Cannot select one model for {row['state']}")
        else:
            specs = [item["apply"] for item in data["multipart"]
                     if matches(item.get("when", {}), properties)]
        if not specs:
            raise ValueError(f"Model has no parts: {row['state']}")
        quads = []
        for spec in specs:
            # Weighted material variants are safe when all their geometry is identical.
            variants = [tuple(self.quads(item)) for item in (spec if isinstance(spec, list) else [spec])]
            first = variants[0]
            if len(variants) > 1 and any(quad_signature(other) != quad_signature(first)
                                         for other in variants[1:]):
                raise ValueError(f"Unsupported randomized geometry: {row['state']}")
            quads.extend(first)

        if (len(quads) == 6 and len({q.direction for q in quads}) == 6
                and all(np.isin(q.corners, (0, 1)).all() for q in quads)):
            kind = "full_cube"
        else:
            kind = next((suffix for suffix in MODEL_KINDS if block.endswith("_" + suffix)), None)
            if not kind or not quads:
                raise ValueError(f"Unsupported partial model: {row['state']}")
        self.geometry[key] = Geometry(kind, tuple(quads))
        return self.geometry[key]


def externally_facing(geometry, eye):
    for point, model in sorted(geometry.items()):
        for quad in model.quads:
            normal = quad.normal
            corners = quad.corners + point
            if (eye - corners[0]) @ normal <= 0:
                continue
            if quad.cull is not None:
                neighbor = tuple(np.asarray(point) + quad.cull)
                if neighbor[1] <= GROUND or (
                    neighbor in geometry and geometry[neighbor].kind == "full_cube"
                ):
                    continue
            yield point, quad, corners


def screen_tiles(low, high, width, height):
    low = np.maximum(low, 0)
    high = np.minimum(high, (width - 1, height - 1))
    return product(*(range(int(a // TILE_SIZE), int(b // TILE_SIZE) + 1)
                     for a, b in zip(low, high)))


def corner_visibility(probes, scene, width, height):
    """Test subpixel face-corner probes against the same projected mesh as the dense map."""
    project = probes.reshape(-1, 3)
    xy = project[:, :2] / project[:, 2:]
    visible = ((xy >= 0) & (xy < (width, height))).all(axis=1) & (project[:, 2] > 0)
    by_tile = defaultdict(list)
    for index in np.flatnonzero(visible):
        by_tile[tuple((xy[index] // TILE_SIZE).astype(int))].append(index)
    by_tile = {tile: np.asarray(indices) for tile, indices in by_tile.items()}
    for quad in scene:
        if np.any(quad[:, 2] <= 0):
            continue
        q = quad[:, :2] / quad[:, 2:]
        quad_low, quad_high = q.min(axis=0), q.max(axis=0)
        if np.any(quad_high < 0) or np.any(quad_low >= (width, height)):
            continue
        polygon_normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
        divisor = polygon_normal @ quad[0]
        if abs(divisor) < 1e-12:
            continue
        depth_coefficients = polygon_normal / divisor
        edges = np.roll(q, -1, axis=0) - q
        winding = np.sign(np.sum(q[:, 0] * edges[:, 1] - q[:, 1] * edges[:, 0]))
        for tile in screen_tiles(quad_low, quad_high, width, height):
            indices = by_tile.get(tile)
            if indices is None:
                continue
            indices = indices[visible[indices] & (project[indices] @ depth_coefficients > 1 + 1e-11)]
            if len(indices):
                offsets = xy[indices, None, :] - q
                cross = edges[:, 0] * offsets[..., 1] - edges[:, 1] * offsets[..., 0]
                visible[indices[(winding * cross >= 0).all(axis=1)]] = False
    return visible.reshape(-1, 4)


def visible_labels(pose, states, origin, *, jar=None):
    origin = tuple(map(int, origin))
    eye, projection, camera = _camera(pose)
    width, height = pose["image_size"]
    if pose.get("minecraft_version", "26.3") != "26.3":
        raise ValueError("Decorative model fixtures require Java 26.3")
    with ZipFile(jar or discover("26.3")) as archive:
        if json.loads(archive.read("version.json"))["id"] != "26.3":
            raise ValueError("Decorative model assets require the Java 26.3 client JAR")
        models = VanillaModels(archive)
        geometry = {point: models.resolve(row) for point, row in states.items()
                    if row["block_id"].removeprefix("minecraft:") != "air"}

    face_ids = np.zeros((height, width), np.int32)
    inverse_depth = np.zeros((height, width), np.float64)
    full_ids = np.zeros_like(face_ids)
    full_depth = np.zeros_like(inverse_depth)
    scene, candidates = [], []
    for point, quad, corners in externally_facing(geometry, eye):
        screen = (corners - eye) @ projection[:, :3].T + projection[:, 3]
        raster = rasterize(screen, width, height)
        scene.append(screen)
        if raster is None:
            continue
        bounds, inside, inv_z, _, xy = raster
        candidates.append((point, quad, corners, xy))
        nearer = inside & (inv_z > inverse_depth[bounds])
        inverse_depth[bounds][nearer] = inv_z[nearer]
        face_ids[bounds][nearer] = len(candidates)
        if geometry[point].kind == "full_cube":
            nearer = inside & (inv_z > full_depth[bounds])
            full_depth[bounds][nearer] = inv_z[nearer]
            full_ids[bounds][nearer] = len(candidates)

    counts = np.bincount(face_ids.ravel(), minlength=len(candidates) + 1)
    winners = np.flatnonzero(counts[1:]) + 1
    if not 0 < len(winners) <= np.iinfo(np.uint16).max:
        raise ValueError(f"Visible face count outside supported range: {len(winners)}")
    selected = [candidates[i - 1] for i in winners]
    probes_world = np.stack([corners + (corners.mean(0) - corners) * 1e-6 + quad.normal * 1e-8
                             for _, quad, corners, _ in selected])
    probes = (probes_world - eye) @ projection[:, :3].T + projection[:, 3]
    clear = corner_visibility(probes, scene, width, height)

    blocks, faces, block_indices = [], [], {}
    cube_faces, decorative_faces = [], []
    kinds = Counter()
    remap = np.zeros(len(candidates) + 1, np.uint16)
    for old_id, (point, quad, corners, xy), visible in zip(winners, selected, clear):
        kind = geometry[point].kind
        if point not in block_indices:
            row = states[point]
            block_indices[point] = len(blocks) + 1
            kinds[kind] += 1
            blocks.append(dict(id=block_indices[point], xyz_world=list(point),
                               xyz_local=[p - o for p, o in zip(point, origin)],
                               block_id=resource_id(row["block_id"]), state=row["state"], geometry_kind=kind))
        face_id = len(faces) + 1
        remap[old_id] = face_id
        in_frame = ((xy >= 0) & (xy < (width, height))).all(axis=1)
        face = dict(id=face_id, block=block_indices[point], xyz_world=list(point),
                    direction=quad.direction, normal_world=quad.normal.tolist(),
                    visible_pixels=int(counts[old_id]), corners_world=corners.tolist(),
                    corners_px=xy.tolist(), corner_visible=(in_frame & visible).tolist())
        faces.append(face)
        (cube_faces if kind == "full_cube" else decorative_faces).append(face)
    num_surfaces = _number_surfaces(cube_faces)
    for face in decorative_faces:
        num_surfaces += 1
        face["surface_id"] = num_surfaces

    # Count full blocks that own pixels in the real image and have other pixels
    # hidden by a decorative piece. The counterfactual view is never exported.
    decorative_ids = np.asarray([False] + [geometry[p].kind != "full_cube"
                                          for p, _, _, _ in candidates])
    behind = full_ids[decorative_ids[face_ids] & (full_depth < inverse_depth)]
    behind_counts = np.bincount(behind, minlength=len(candidates) + 1)
    mixed_blocks = set()
    mixed_pixels = 0
    for index in np.flatnonzero(behind_counts[1:]) + 1:
        point = candidates[index - 1][0]
        if point in block_indices:
            mixed_blocks.add(point)
            mixed_pixels += int(behind_counts[index])

    depth = np.full(face_ids.shape, np.nan, np.float32)
    np.divide(1., inverse_depth, out=depth, where=inverse_depth > 0)
    metadata = dict(schema="visible-decorative-block-labels/v1", origin_world=list(origin),
                    camera=camera, frame_id=pose["id"], blocks=blocks, faces=faces,
                    surface_count=num_surfaces, block_kinds=dict(sorted(kinds.items())),
                    overlapping_visible_full_blocks=len(mixed_blocks), occluded_overlap_pixels=mixed_pixels,
                    surface_definition="Edge-connected visible unit cube faces, plus each visible decorative model quad",
                    background="face_id 0 and depth_z NaN = no labeled building surface at pixel center",
                    visibility="Only the nearest visible opaque modeled surface at each native pixel center is labeled.")
    return metadata, remap[face_ids], depth


def write_labels(pose, states, origin, output_dir, *, jar=None):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    metadata, face_ids, depth = visible_labels(pose, states, origin, jar=jar)
    (folder / "labels.json").write_text(json.dumps(metadata, separators=(",", ":"), allow_nan=False)
                                        + "\n", encoding="utf-8")
    np.savez_compressed(folder / "labels.npz", face_id=face_ids, depth_z=depth)
    return dict(labels="labels.json", raster="labels.npz", visible_blocks=len(metadata["blocks"]),
                visible_faces=len(metadata["faces"]), visible_pixels=int(np.count_nonzero(face_ids)),
                visible_surfaces=metadata["surface_count"],
                visible_full_blocks=metadata["block_kinds"].get("full_cube", 0),
                visible_decorative_blocks=len(metadata["blocks"]) - metadata["block_kinds"].get("full_cube", 0),
                visible_block_kinds=metadata["block_kinds"],
                visible_full_blocks_behind_decorations=metadata["overlapping_visible_full_blocks"],
                visible_overlap_occluded_pixels=metadata["occluded_overlap_pixels"])
