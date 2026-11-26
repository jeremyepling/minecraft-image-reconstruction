"""Evaluation-only face and grid labels from the recorded renderer and server state."""
import json
import gzip
from collections import Counter

import numpy as np
from PIL import Image

from benchmark.camera_fixture import VERSION, write
from benchmark.textures import clear_rays
from calibration.world import GROUND
from engine.atlas import digest
from engine.nbt import read_nbt


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
    frames, readbacks, floor_readbacks, faces_total = [], 0, 0, 0
    floor = layout.get("floor", "smooth_stone")
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
        floor_sites = {p for p, b in expected.items() if b["role"] == "floor_probe"}
        if "floor" in layout and (not floor_sites or
                any(actual[p]["block_id"] != floor or p[1] != GROUND for p in floor_sites)):
            raise ValueError(f"Unverified {floor} floor in {states_file}")
        floor_readbacks += len(floor_sites)
        building = {p: b for p, b in actual.items() if p not in floor_sites}
        faces = face_labels(pose, building, {p: b["role"] for p, b in expected.items()})
        labels = f"labels/{v['id']}.json"
        write(root / labels, dict(id=v["id"], faces=faces))
        readbacks += len(rows)
        faces_total += len(faces)
        frame = {k: v[k] for k in ("id", "layout", "split", "palette", "angle", "condition")}
        frames.append(dict(**frame, floor=floor, image=image, pose=pose_path, states=states_file, labels=labels,
                           image_sha256=digest(root/image), label_sha256=digest(root/labels)))
    library = dict(version=VERSION, client_sha256=layout["client_sha256"], ground_y=GROUND, floor=floor,
                   layout="layout.json", buildings="buildings.json", views="views.json", frames=frames,
                   visibility="Frame-clipped, occlusion-tested native pixel-center count divided by full-face pixel count (continuous projected area for clipped faces). Only faces owning a pixel are listed.",
                   verified=dict(frames=len(frames), state_readbacks=readbacks, visible_faces=faces_total,
                                 floor_readbacks=floor_readbacks,
                                 by_split=dict(Counter(v["split"] for v in frames))))
    write(root / "index.json", library)
    print(f"Verified {len(frames)} frames, {readbacks} server state readbacks, {faces_total} visible faces.", flush=True)


def floor_evidence(root, index):
    floor = index.get("floor", "smooth_stone")
    result = dict(block_id="minecraft:"+floor, floor_readbacks=index["verified"].get("floor_readbacks", 0))
    cfg = root / "client/saves/Palette Lab/data/minecraft/world_gen_settings.dat"
    if cfg.is_file():
        dim = read_nbt(gzip.decompress(cfg.read_bytes()))["data"]["dimensions"]["minecraft:overworld"]
        layers = dim["generator"]["settings"]["layers"]
        if layers[-1] != dict(block=result["block_id"], height=1):
            raise ValueError("Saved superflat surface disagrees with the expected material")
        result.update(saved_worldgen_layers=layers, saved_worldgen_sha256=digest(cfg))
    return result


def verify_pair(reference, alternate):
    """Compare checked server state, native cameras, labels and sampled face pixels."""
    left, right = (load(p / "index.json") for p in (reference, alternate))
    reference_frames = {f["id"]: f for f in left["frames"]}
    if (reference_frames.keys() != {f["id"] for f in right["frames"]}
        or left["client_sha256"] != right["client_sha256"]):
        raise ValueError("Paired captures require the same view set and Java client")
    for key in ("buildings", "views"):
        if load(reference / left[key]) != load(alternate / right[key]):
            raise ValueError(f"Paired {key} differ")
    expected_blocks = []
    for root, index in ((reference, left), (alternate, right)):
        layout = load(root / index["layout"])
        expected_blocks.append([b for b in layout["blocks"] if b["role"] != "floor_probe"])
    if expected_blocks[0] != expected_blocks[1]:
        raise ValueError("Paired building layouts differ")
    rows = []
    for frame in right["frames"]:
        old = reference_frames[frame["id"]]
        views = ((reference, old), (alternate, frame))
        for field in ("id", "split", "layout", "palette", "condition", "angle"):
            if old[field] != frame[field]:
                raise ValueError(f"{field} differs for {frame['id']}")
        files = [(load(root/v["pose"]), load(root/v["labels"]),
                  [b for b in load(root/v["states"]) if b["y"] > GROUND]) for root, v in views]
        if files[0] != files[1]:
            raise ValueError(f"Paired camera, conditions, projected visible faces or server state differ for {frame['id']}")
        _, labels, states = files[1]
        images = []
        for root, v in views:
            with Image.open(root/v["image"]) as im:
                images.append(np.asarray(im.convert("RGB")))
        a, b = images
        diff = np.any(a != b, axis=2)
        # Fully visible face centers avoid seams, sky, neighboring cells and ground.
        xy = np.array([np.mean(f["image_corners"], axis=0) for f in labels["faces"]
                       if f["visible_fraction"] == 1 and f["visible_pixels"] >= 16])
        x, y = np.floor(xy).astype(int).T
        rows.append(dict(id=frame["id"], matching_building_readbacks=len(states),
                         matching_face_labels=len(labels["faces"]), matching_pose_and_conditions=True,
                         changed_rgb_pixels=int(diff.sum()), face_center_samples=len(x),
                         changed_face_center_samples=int(diff[y, x].sum())))
    report = dict(game_version=right["version"], reference_index_sha256=digest(reference/"index.json"),
                  alternate_index_sha256=digest(alternate/"index.json"),
                  reference_floor=floor_evidence(reference, left), alternate_floor=floor_evidence(alternate, right),
                  checked="Native pose and projection, HUD, weather, clock, resource packs, frame resolution, layout cells, block states and light levels, projected faces, visible fractions and corner visibility.",
                  frames=rows)
    write(alternate / "pair-verification.json", report)
    return report
