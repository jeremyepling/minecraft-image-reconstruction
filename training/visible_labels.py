"""Per-view ground truth for the opaque, full-cube exterior capture fixtures."""
import json
from pathlib import Path

import numpy as np

from benchmark.camera_labels import exposed_faces, rasterize, screen_matrix
from benchmark.textures import clear_rays


def _camera(pose):
    projection = screen_matrix(pose)
    eye = np.asarray(pose["eye_world"], dtype=float)
    width, height = pose["image_size"]
    # Screen coordinates use edge origin, with the first pixel center at (0.5, 0.5).
    scale = np.linalg.norm(projection[2, :3])
    projection = projection / scale
    forward = projection[2, :3]
    cx, cy = projection[:2, :3] @ forward
    right = projection[0, :3] - cx * forward
    down = projection[1, :3] - cy * forward
    fx, fy = np.linalg.norm(right), np.linalg.norm(down)
    rotation = np.stack((right / fx, down / fy, forward))
    world_matrix = projection.copy()
    world_matrix[:, 3] -= projection[:, :3] @ eye
    camera = dict(image_size=[width, height], eye_world=eye.tolist(),
                  K=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                  world_to_camera_rotation=rotation.tolist(),
                  world_to_pixel_homogeneous=world_matrix.tolist(),
                  camera_axes=["right", "down", "forward"],
                  pixel_center_offset=[0.5, 0.5], depth_unit="blocks along camera forward")
    return eye, projection, camera


def _number_surfaces(faces):
    """Edge-connected, visibly represented faces on the same plane, across materials."""
    lookup = {(tuple(f["xyz_world"]), f["direction"]): f for f in faces}
    surface_id = 0
    for face in faces:
        if "surface_id" in face:
            continue
        surface_id += 1
        face["surface_id"] = surface_id
        queue = [face]
        while queue:
            current = queue.pop()
            for axis, normal in enumerate(current["normal_world"]):
                if normal:
                    continue
                for offset in [-1, 1]:
                    neighbor = current["xyz_world"].copy()
                    neighbor[axis] += offset
                    candidate = lookup.get((tuple(neighbor), current["direction"]))
                    if candidate is not None and "surface_id" not in candidate:
                        candidate["surface_id"] = surface_id
                        queue.append(candidate)
    return surface_id


def visible_labels(pose, states, origin):
    """Compute dense target-face ownership and depth from a single native camera pose.

    states contains building cells in world coordinates, excluding the flat floor.
    The capture contract requires opaque full cubes and a camera above ground.
    """
    origin = tuple(map(int, origin))
    eye, projection, camera = _camera(pose)
    width, height = pose["image_size"]
    occupied = {point for point, row in states.items()
                if row["block_id"].removeprefix("minecraft:") != "air"}
    face_ids = np.zeros((height, width), np.int32)
    inverse_depth = np.zeros((height, width), np.float64)
    candidates = []
    for point, normal, direction, corners in exposed_faces(occupied, eye):
        screen = (corners - eye) @ projection[:, :3].T + projection[:, 3]
        raster = rasterize(screen, width, height)
        if raster is None:
            continue
        bounds, inside, inv_z, _, xy = raster
        closer = inside & (inv_z > inverse_depth[bounds])
        inverse_depth[bounds][closer] = inv_z[closer]
        face_ids[bounds][closer] = len(candidates) + 1
        candidates.append((point, normal, direction, corners, xy))

    counts = np.bincount(face_ids.ravel(), minlength=len(candidates) + 1)
    winners = np.flatnonzero(counts[1:]) + 1
    if not len(winners):
        raise ValueError(f"No visible building faces in {pose.get('id')}")
    if len(winners) > np.iinfo(np.uint16).max:
        raise ValueError("Frame exceeds uint16 face table")
    # Probe just inside the four geometric corners, as in the legacy face reference.
    visible_candidates = [candidates[i - 1] for i in winners]
    probes = np.concatenate([corners + (corners.mean(0) - corners) * 1e-6 + normal * 1e-8
                             for _, normal, _, corners, _ in visible_candidates])
    clear = clear_rays(occupied, eye, probes).reshape(-1, 4)
    blocks, faces, block_indices = [], [], {}
    remap = np.zeros(len(candidates) + 1, np.uint16)
    for old_id, (point, normal, direction, corners, xy), unobstructed in zip(
            winners, visible_candidates, clear):
        if point not in block_indices:
            row = states[point]
            block_indices[point] = len(blocks) + 1
            block_id = row["block_id"].removeprefix("minecraft:")
            blocks.append(dict(id=block_indices[point], xyz_world=list(point),
                               xyz_local=[p - o for p, o in zip(point, origin)],
                               block_id="minecraft:" + block_id, state=row["state"]))
        face_id = len(faces) + 1
        remap[old_id] = face_id
        in_frame = ((xy >= 0) & (xy < (width, height))).all(axis=1)
        faces.append(dict(id=face_id, block=block_indices[point], xyz_world=list(point),
                          direction=direction, normal_world=normal.tolist(),
                          visible_pixels=int(counts[old_id]), corners_world=corners.tolist(),
                          corners_px=xy.tolist(), corner_visible=(in_frame & unobstructed).tolist()))
    num_surfaces = _number_surfaces(faces)
    depth = np.full(face_ids.shape, np.nan, np.float32)
    np.divide(1., inverse_depth, out=depth, where=inverse_depth > 0)
    metadata = dict(schema="visible-block-labels/v1", origin_world=list(origin), camera=camera,
                    frame_id=pose["id"], blocks=blocks, faces=faces,
                    surface_definition="Edge-connected visible unit faces with equal plane and normal",
                    surface_count=num_surfaces,
                    background="face_id 0 and depth_z NaN = no labeled building surface at pixel center",
                    visibility="Ray segment before each target hit is observed free space within the fixture; space behind a hit is unknown. Background alone does not label world cells as air.")
    return metadata, remap[face_ids], depth


def write_labels(pose, states, origin, output_dir):
    """Write one frame in a private output directory and return relative asset names and counts."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    metadata, face_ids, depth = visible_labels(pose, states, origin)
    (folder / "labels.json").write_text(json.dumps(metadata, separators=(",", ":"), allow_nan=False)
                                        + "\n", encoding="utf-8")
    np.savez_compressed(folder / "labels.npz", face_id=face_ids, depth_z=depth)
    return dict(labels="labels.json", raster="labels.npz", visible_blocks=len(metadata["blocks"]),
                visible_faces=len(metadata["faces"]), visible_pixels=int(np.count_nonzero(face_ids)),
                visible_surfaces=metadata["surface_count"])
