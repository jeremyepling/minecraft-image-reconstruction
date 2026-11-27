"""Build independently rebased planar charts from native visible full-cube labels."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from benchmark.camera_labels import screen_matrix


UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
# Each projected edge runs from its smaller tangent coordinate to its larger one.
ENDS = [(0, 1), (1, 2), (3, 2), (0, 3)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


def unit_projection(face, axes):
    world = np.asarray(face["corners_world"])
    uv = world[:, axes]
    relative = uv - uv.min(axis=0)
    if sorted(map(tuple, relative)) != sorted(map(tuple, UNIT)):
        raise ValueError(f"Non-unit corners on face {face['id']}")
    vertices = dict(zip(map(tuple, relative), face["corners_px"]))
    return np.array([vertices[tuple(c)] for c in UNIT])


def fit_homography(local_uv, pixels):
    # Fit a map from a single well-observed quad with its own zero-origin frame.
    system, values = [], []
    for (u, v), (x, y) in zip(UNIT, pixels):
        system.extend(([u, v, 1, 0, 0, 0, -u*x, -v*x],
                       [0, 0, 0, u, v, 1, -u*y, -v*y]))
        values.extend((x, y))
    h = np.append(np.linalg.solve(system, values), 1).reshape(3, 3)
    u, v = local_uv
    h = h @ np.array([[1, 0, -u], [0, 1, -v], [0, 0, 1]])
    return h / h[2, 2]


def owner_edge_samples(pixels, face_id, owners):
    # Sixteen samples per projected segment, each checked against the native
    # owner on the inside. Bit positions increase along the world tangent.
    p = pixels[[a for a, _ in ENDS]]
    d = pixels[[b for _, b in ENDS]] - p
    inward = np.column_stack((-d[:, 1], d[:, 0]))
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-10)
    inward *= np.sign(np.sum(inward * (pixels.mean(axis=0) - p), axis=1))[:, None]
    samples = p[:, None, :] + d[:, None, :] * ((np.arange(16) + .5) / 16)[None, :, None]
    # Owner presence further than one native pixel from the projected segment
    # does not count. For narrow faces try both distances.
    hit = np.zeros((4, 16), dtype=bool)
    height, width = owners.shape
    for offset in (.30, .75):
        q = np.floor(samples + offset * inward[:, None, :]).astype(int)
        x, y = q[:, :, 0], q[:, :, 1]
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        hit[valid] |= owners[y[valid], x[valid]] == face_id
    return (hit.astype(int) @ (1 << np.arange(16))).tolist()


def components(plane):
    remaining = set(plane)
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = [start]
        for u, v in queue:
            for uv in ((u-1, v), (u+1, v), (u, v-1), (u, v+1)):
                if uv in remaining:
                    remaining.remove(uv)
                    queue.append(uv)
        yield queue


def view(case, output):
    frame = Path(case["frame"])
    source = Path(case["variants"]["sky"]["image"])
    hashes = {name: digest(frame / name) for name in case["reference_sha256"]}
    source_hash = digest(source)
    if hashes != case["reference_sha256"] or source_hash != case["variants"]["sky"]["source_sha256"]:
        raise ValueError(f"Frozen input changed on {case['id']}")
    refs = json.loads((frame / "labels.json").read_text())
    pose = json.loads((frame / "camera.json").read_text())
    recorded_projection = screen_matrix(pose)
    recorded_eye = np.asarray(pose["eye_world"])
    max_recorded_error = 0.0
    with np.load(frame / "labels.npz") as arrays:
        owners = arrays["face_id"]
    counts = np.bincount(owners.ravel().astype(int))
    blocks = {b["id"]: b for b in refs["blocks"]}
    planes = defaultdict(dict)
    full_blocks = set()
    for face in refs["faces"]:
        block = blocks[face["block"]]
        if counts[face["id"]] != face["visible_pixels"] or face["visible_pixels"] <= 0:
            raise ValueError("Mismatch between native ownership and recorded visible face")
        if block.get("geometry_kind", "full_cube") != "full_cube":
            continue
        camera_relative = np.asarray(face["corners_world"]) - recorded_eye
        q = camera_relative @ recorded_projection[:, :3].T + recorded_projection[:, 3]
        error_px = np.linalg.norm(q[:, :2] / q[:, 2:] - face["corners_px"], axis=1)
        max_recorded_error = max(max_recorded_error, float(error_px.max()))
        full_blocks.add(block["id"])
        n = face["normal_world"]
        axis = next(i for i, direction in enumerate(n) if direction)
        signed = n[axis] * (axis + 1)
        axes = tuple(i for i in range(3) if i != axis)
        xyz = face["xyz_world"]
        uv = tuple(xyz[i] for i in axes)
        plane = planes[(signed, xyz[axis])]
        if uv in plane:
            raise ValueError("Duplicate unit face")
        plane[uv] = (face, block["block_id"], unit_projection(face, axes))

    groups = [(signed, plane, cells) for (signed, _), plane in planes.items()
              for cells in components(plane)]
    rng = random.Random(f"rebased-unit-faces-v1/{case['id']}")
    rng.shuffle(groups)
    charts, max_error = [], 0.0
    for i, (signed, plane, cells) in enumerate(groups):
        axes = [k+1 for k in range(3) if k+1 != abs(signed)]
        uv_min, uv_max = np.min(cells, axis=0), np.max(cells, axis=0)
        shift = uv_min - [rng.randint(-12, 12), rng.randint(-12, 12)]
        anchor = max(cells, key=lambda uv: plane[uv][0]["visible_pixels"])
        h = fit_homography(np.array(anchor) - shift, plane[anchor][2])
        local_cells = []
        for world_uv in cells:
            face, material, px = plane[world_uv]
            uv = np.array(world_uv) - shift
            q = np.column_stack((UNIT + uv, np.ones(4))) @ h.T
            max_error = max(max_error, float(np.linalg.norm(q[:, :2]/q[:, 2:] - px, axis=1).max()))
            bits = owner_edge_samples(px, face["id"], owners)
            local_cells.append(dict(u=int(uv[0]), v=int(uv[1]), block_id=material,
                                    visible_edges=[e for e, b in enumerate(bits) if b],
                                    edge_samples=bits))
        rng.shuffle(local_cells)
        charts.append(dict(id=f"chart-{i+1:04d}", H=h.tolist(), cells=local_cells, axes=axes,
                           normal_axis=signed, confidence=1.0,
                           size_uv=(uv_max - uv_min + 1).tolist()))
    if max(max_error, max_recorded_error) > 1e-4:
        raise ValueError(f"Corner discrepancy: chart {max_error}, capture {max_recorded_error} pixels")
    out = output / "sky" / case["id"] / "charts.json"
    dump(out, dict(id=case["id"], source_sha256=source_hash, charts=charts))
    return dict(id=case["id"], split=case["split"], source_sha256=source_hash,
                reference_sha256=hashes, charts_sha256=digest(out), charts=len(charts),
                cells=sum(len(c["cells"]) for c in charts), full_cube_owners=len(full_blocks),
                excluded_non_cube_owners=len(blocks)-len(full_blocks),
                visible_edges=sum(len(x["visible_edges"]) for c in charts for x in c["cells"]),
                max_corner_reprojection_error_px=max_error, max_recorded_corner_error_px=max_recorded_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    with ThreadPoolExecutor(max_workers=6) as pool:
        cases = list(pool.map(lambda c: view(c, args.output), fixtures["cases"]))
    dump(args.output / "oracle-manifest.json", dict(
        method="oracle-connected-full-cube-charts-v1", revision=fixtures["revision"],
        generator_sha256=digest(Path(__file__)), fixtures_sha256=digest(args.fixtures),
        oracle_fields=["frontmost unit face coverage", "coplanar four-neighbor grouping",
                       "exact projected lattice corners", "signed axes", "block material"],
        local_frame="Independent random integral U,V rebase per chart; randomly ordered charts and cells",
        edge_samples="Four masks of 16 bins sampled on the inward side at 0.30 and 0.75 pixels. "
                     "Lower V, upper U, upper V, lower U; bins increase along tangent. "
                     "Exact corners extend under occlusion; masks record native owner support.",
        cases=cases))
    for case in cases:
        print(case["id"], case["charts"], case["cells"], case["full_cube_owners"], flush=True)


if __name__ == "__main__":
    main()
