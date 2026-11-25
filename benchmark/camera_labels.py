"""Evaluation-only face and grid labels from the recorded renderer and server state."""
import json
from collections import Counter

import numpy as np
from PIL import Image

from benchmark.camera_fixture import VERSION, write
from benchmark.textures import clear_rays
from calibration.world import GROUND
from engine.atlas import digest


DIRECTIONS = (("west", "east"), ("down", "up"), ("north", "south"))


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def screen_matrix(pose):
    """Native JOML projection is column-major and operates on camera-relative XYZ."""
    m = np.array(pose["view_rotation_projection"]).reshape(4, 4).T
    w, h = pose["image_size"]
    return np.stack(((m[0] + m[3]) * w / 2, (m[3] - m[1]) * h / 2, m[3]))


def exposed_faces(occupied, eye):
    for point in sorted(occupied):
        for axis in range(3):
            sign = 1 if eye[axis] >= point[axis] + 1 else -1
            if 0 <= eye[axis] - point[axis] <= 1:
                continue
            normal = np.eye(3, dtype=int)[axis] * sign
            neighbor = tuple(np.asarray(point) + normal)
            if neighbor in occupied or neighbor[1] <= GROUND:
                continue
            start = np.array(point)
            start[axis] += int(sign > 0)
            u, v = np.eye(3, dtype=int)[[(axis + 1) % 3, (axis + 2) % 3]]
            corners = np.array([start, start + u, start + u + v, start + v])
            if sign < 0:
                corners = corners[::-1]
            yield point, normal, DIRECTIONS[axis][sign > 0], corners


def rasterize(quad3, width, height):
    """Return a screen-clipped pixel-center mask, reciprocal depth and full-face area."""
    x, y, w = quad3.T
    if any(np.all(side <= 0) for side in (w, x, y, width*w-x, height*w-y)):
        return None
    # Crossing the camera plane can make a projected polygon unbounded. These fixtures
    # arrange the close views so that such side faces are already outside the frustum.
    if w.min() <= 0:
        raise ValueError("A face crossing the camera plane intersects the scoring viewport")
    q = quad3[:, :2] / w[:, None]
    low = np.maximum(0, np.floor(q.min(0)).astype(int))
    high = np.minimum((width, height), np.ceil(q.max(0)).astype(int))
    if np.any(low >= high):
        return None
    px, py = np.meshgrid(np.arange(low[0], high[0]) + .5, np.arange(low[1], high[1]) + .5)
    polygon = np.ones(px.shape, dtype=bool)
    area = .5 * (q[:, 0] @ np.roll(q[:, 1], -1) - q[:, 1] @ np.roll(q[:, 0], -1))
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        cross = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        polygon &= np.sign(area) * cross >= 0
    if not polygon.any():
        return None
    a, b, c = np.linalg.solve(quad3[:3], np.ones(3))
    depth = a * px + b * py + c
    complete = np.all(q >= 0) and np.all(q <= (width, height))
    denominator = polygon.sum() if complete else abs(area)
    return (slice(low[1], high[1]), slice(low[0], high[0])), polygon, depth, denominator, q


def face_labels(pose, states, roles):
    w, h = pose["image_size"]
    eye = np.asarray(pose["eye_world"])
    projection = screen_matrix(pose)
    occupied = {p for p, b in states.items() if b["block_id"] != "air"}
    ids = np.zeros((h, w), dtype=np.int32)
    depths = np.zeros((h, w))
    records = []
    for point, normal, direction, corners in exposed_faces(occupied, eye):
        raster = rasterize((corners - eye) @ projection[:, :3].T + projection[:, 3], w, h)
        if raster is None:
            continue
        box, polygon, depth, denominator, xy = raster
        closer = polygon & (depth > depths[box])
        depths[box][closer] = depth[closer]
        ids[box][closer] = len(records) + 1
        records.append((point, direction, normal, corners, xy, denominator))
    if not records:
        raise ValueError(f"No visible structure in {pose['id']}")
    counts = np.bincount(ids.ravel(), minlength=len(records) + 1)[1:]
    covered = [(n, rec) for n, rec in zip(counts, records) if n > 0]
    # Approach each corner from within its own face to disambiguate silhouette ties.
    probes = np.concatenate([c + (c.mean(0) - c) * 1e-6 + n * 1e-8
                             for _, (_, _, n, c, _, _) in covered])
    clear = clear_rays(occupied, eye, probes).reshape(-1, 4)
    faces = []
    for (pixels, (p, direction, normal, corners, xy, denominator)), visibility in zip(covered, clear):
        in_frame = np.all((xy >= 0) & (xy < (w, h)), axis=1)
        faces.append(dict(xyz=list(p), normal=normal.tolist(), direction=direction, role=roles[p],
                          block_id="minecraft:" + states[p]["block_id"], state=states[p]["state"],
                          world_corners=corners.tolist(), image_corners=xy.tolist(),
                          corner_visible=(in_frame & visibility).tolist(), visible_pixels=int(pixels),
                          visible_fraction=min(1., float(pixels / denominator))))
    return faces


def label_capture(root):
    layout = load(root / "layout.json")
    views = load(root / "views.json")
    if layout["version"] != VERSION:
        raise ValueError("Fixture game version mismatch")
    pages = {}
    for part in layout["blocks"]:
        point = tuple(part[axis] for axis in "xyz")
        pages.setdefault(part["page"], {})[point] = part
    (root / "labels").mkdir(exist_ok=True)
    frames, readbacks, faces_total = [], 0, 0
    for v in views:
        stem = "captures/" + v["id"]
        pose_path, image, states_file = stem + ".json", stem + ".png", stem + "-blocks.json"
        pose = load(root / pose_path)
        with Image.open(root / image) as im:
            if (list(im.size) != pose["image_size"] or pose["id"] != v["id"]
                or pose["clock_ticks"] % 24000 != 6000 or pose["thundering"] or pose["raining"]
                or pose["resource_packs"] != ["vanilla"] or not pose["hud_hidden"]
                or pose["minecraft_version"] != VERSION):
                raise ValueError(f"Capture condition mismatch in {image}")
        expected = pages[v["page"]]
        rows = load(root / states_file)
        actual = {tuple(b[axis] for axis in "xyz"): b for b in rows}
        if (len(actual) != len(rows) or actual.keys() != expected.keys()
            or any(b["block_id"] != expected[p]["block_id"]
                   or b["state"] != "Block{minecraft:" + b["block_id"] + "}" for p, b in actual.items())):
            raise ValueError(f"Server state disagrees with staged world in {states_file}")
        faces = face_labels(pose, actual, {p: b["role"] for p, b in expected.items()})
        labels = f"labels/{v['id']}.json"
        write(root / labels, dict(id=v["id"], faces=faces))
        readbacks += len(rows)
        faces_total += len(faces)
        frame = {k: v[k] for k in ("id", "layout", "split", "palette", "angle", "condition")}
        frames.append(dict(**frame, image=image, pose=pose_path, states=states_file, labels=labels,
                           image_sha256=digest(root/image), label_sha256=digest(root/labels)))
    library = dict(version=VERSION, client_sha256=layout["client_sha256"], ground_y=GROUND,
                   layout="layout.json", buildings="buildings.json", views="views.json", frames=frames,
                   visibility="Frame-clipped, occlusion-tested native pixel-center count divided by full-face pixel count (continuous projected area for clipped faces). Only faces owning a pixel are listed.",
                   verified=dict(frames=len(frames), state_readbacks=readbacks, visible_faces=faces_total,
                                 by_split=dict(Counter(v["split"] for v in frames))))
    write(root / "index.json", library)
    print(f"Verified {len(frames)} frames, {readbacks} server state readbacks, {faces_total} visible faces.", flush=True)
