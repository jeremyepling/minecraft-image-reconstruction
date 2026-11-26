"""Check a visible-builds-v1 download against its manifest and geometric evidence."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def face_depths(face_ids, depth, labels, fid):
    foreground = face_ids != 0
    require(np.all(np.isnan(depth[~foreground])), f"Unexpected background depth: {fid}")
    require(np.all(np.isfinite(depth[foreground]) & (depth[foreground] > 0)), f"Invalid hit depths: {fid}")
    counts = np.bincount(face_ids.ravel())
    expected = [f["visible_pixels"] for f in labels["faces"]]
    require(counts[1:].tolist() == expected and min(expected) > 0, f"Face ownership mismatch: {fid}")
    # Test up to 4096 uniformly distributed foreground samples in world space.
    pixels = np.flatnonzero(foreground)
    pixels = pixels[np.linspace(0, len(pixels) - 1, min(4096, len(pixels)), dtype=int)]
    y, x = np.unravel_index(pixels, foreground.shape)
    camera = labels["camera"]
    uv1 = np.column_stack((x + .5, y + .5, np.ones(len(x))))
    rays = uv1 @ np.linalg.inv(camera["K"]).T @ np.asarray(camera["world_to_camera_rotation"])
    world = np.asarray(camera["eye_world"]) + rays * depth[y, x, None]
    points = np.asarray([f["xyz_world"] for f in labels["faces"]])[face_ids[y, x] - 1]
    normals = np.asarray([f["normal_world"] for f in labels["faces"]])[face_ids[y, x] - 1]
    offset = world - points
    eps = 2e-4  # Float32 depth, rounded native camera matrix, and block coordinate units.
    plane_error = ((offset - np.maximum(normals, 0)) * normals).sum(axis=1)
    require(np.all(np.abs(plane_error) <= eps) and np.all(offset >= -eps) and np.all(offset <= 1 + eps),
            f"Sampled depths fail face-plane/bounds check: {fid}")


def verify_frame(archive, row, listed, source):
    fid = row["id"]
    directory = row["directory"] + "/"
    require(row["directory"] == "frames/" + fid, f"Frame path mismatch: {fid}")
    def read(name):
        return archive.read(directory + name)

    require(json.loads(read("frame.json")) == row, f"Frame index mismatch: {fid}")
    assets = {name: read(name) for name in row["files"]}
    for name, payload in assets.items():
        require(digest(payload) == row["files"][name], f"Asset checksum mismatch: {fid}/{name}")
    require(row["files"]["rgb.png"] == listed["rgb_sha256"], f"RGB checksum mismatch: {fid}")
    if row["split"] == "regression":
        for key, filename in (("image", "rgb.png"), ("pose", "camera.json"), ("states", "world-readback.json")):
            require(row["files"][filename] == row["source_sha256"][key], f"Original changed: {fid}/{key}")
    pose = json.loads(assets["camera.json"])
    labels = json.loads(assets["labels.json"])
    with Image.open(io.BytesIO(assets["rgb.png"])) as image:
        image.load()
        require(image.format == "PNG" and list(image.size) == pose["image_size"] == labels["camera"]["image_size"],
                f"Image dimensions disagree: {fid}")
    require(labels["frame_id"] == pose["id"] and labels["camera"]["eye_world"] == pose["eye_world"],
            f"Camera identifiers/positions disagree: {fid}")
    actual = {tuple(b[k] for k in "xyz"): b for b in json.loads(assets["world-readback.json"])}
    origin = labels["origin_world"]
    require(origin == row["origin_world"], f"Origin mismatch: {fid}")
    blocks = {}
    for block in labels["blocks"]:
        xyz = tuple(block["xyz_world"])
        state = actual.get(xyz)
        require(state and state["block_id"].removeprefix("minecraft:") != "air" and xyz[1] > row["ground_y"]
                and block["block_id"] == "minecraft:" + state["block_id"].removeprefix("minecraft:")
                and block["state"] == state["state"] and block["id"] not in blocks
                and block["xyz_local"] == [v - o for v, o in zip(xyz, origin)], f"Block readback mismatch: {fid}")
        blocks[block["id"]] = xyz
    require(len(set(blocks.values())) == len(blocks), f"Duplicate visible cell: {fid}")
    require({f["block"] for f in labels["faces"]} == set(blocks), f"Unobserved or missing blocks: {fid}")
    for i, face in enumerate(labels["faces"], 1):
        require(face["id"] == i and tuple(face["xyz_world"]) == blocks[face["block"]],
                f"Face references invalid block: {fid}")
    with np.load(io.BytesIO(assets["labels.npz"]), allow_pickle=False) as raster:
        face_ids, depth = raster["face_id"], raster["depth_z"]
        require(face_ids.dtype == np.uint16 and depth.dtype == np.float32 and
                face_ids.shape == depth.shape == tuple(reversed(pose["image_size"])), f"Invalid raster: {fid}")
        face_depths(face_ids, depth, labels, fid)
    if "reference-faces.json" in assets:
        reference = assets["reference-faces.json"]
        if row["split"] == "regression":
            require(digest(reference) == row["source_sha256"]["labels"], f"Archived face checksum mismatch: {fid}")
            if source:
                original = (source / row["source_paths"]["labels"]).read_bytes()
                require(original == reference, f"Original face checksum mismatch: {fid}")
        old = {(tuple(f["xyz"]), f["direction"]): f["visible_pixels"] for f in json.loads(reference)["faces"]}
        new = {(tuple(f["xyz_world"]), f["direction"]): f["visible_pixels"] for f in labels["faces"]}
        require(old == new, f"Reference face ownership changed: {fid}")
    return len(blocks), len(labels["faces"])


def verify(manifest_path, archive_dir, source=None):
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest["dataset"] == "visible-builds-v1" and manifest["schema_version"] == 1,
            "Unsupported dataset/version")
    capture_record = manifest["capture_records"]
    capture_path = archive_dir / capture_record["file"]
    require(capture_path.stat().st_size == capture_record["bytes"] and
            digest(capture_path.read_bytes()) == capture_record["sha256"], "Capture records checksum mismatch")
    with ZipFile(capture_path) as capture:
        require(capture.testzip() is None, "Corrupt capture records")
        new_rows = {r["id"]: r for r in json.loads(capture.read("index.json"))["frames"]}
    listed = {row["id"]: row for row in manifest["frames"]}
    require(len(listed) == len(manifest["frames"]), "Duplicate manifest frame")
    seen, layout_splits, family_splits, image_splits = set(), {}, {}, {}
    total_blocks = total_faces = 0
    for split, shard in manifest["shards"].items():
        path = archive_dir / shard["file"]
        with path.open("rb") as stream:
            require(path.stat().st_size == shard["bytes"] and hashlib.file_digest(stream, "sha256").hexdigest()
                    == shard["sha256"], f"Shard checksum mismatch: {path.name}")
        with ZipFile(path) as archive:
            require(len(archive.namelist()) == len(set(archive.namelist())), f"Duplicate ZIP entry: {path.name}")
            rows = [json.loads(line) for line in archive.read("frames.jsonl").splitlines()]
            require(len(rows) == shard["frames"] and len({r["layout"] for r in rows}) == shard["layouts"],
                    f"Incorrect shard counts: {split}")
            for row in rows:
                fid = row["id"]
                require(fid not in seen and fid in listed and row["split"] == split and
                        all(listed[fid][k] == row[k] for k in ("split", "layout", "palette")),
                        f"Frame split or identity mismatch: {fid}")
                require(layout_splits.setdefault(row["layout"], split) == split, f"Layout leaks across splits: {fid}")
                require(family_splits.setdefault(row.get("family", row["layout"]), split) == split,
                        f"Layout family leaks across splits: {fid}")
                require(image_splits.setdefault(listed[fid]["rgb_sha256"], split) == split,
                        f"Image leaks across splits: {fid}")
                if split != "regression":
                    original = new_rows.pop(fid)
                    require(original["image_sha256"] == row["files"]["rgb.png"] and
                            original["label_sha256"] == row["files"]["reference-faces.json"] and
                            original["source_sha256"]["pose"] == row["files"]["camera.json"] and
                            original["source_sha256"]["readback"] == row["files"]["world-readback.json"],
                            f"Capture records disagree with split: {fid}")
                blocks, faces = verify_frame(archive, row, listed[fid], source)
                total_blocks += blocks
                total_faces += faces
                seen.add(fid)
    require(seen == set(listed), "Manifest and shard membership differ")
    require(not new_rows, "Capture index contains extra frames")
    return dict(frames=len(seen), layouts=len(layout_splits), visible_block_appearances=total_blocks,
                visible_face_appearances=total_faces, checked_archived_face_counts=bool(source))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archives", type=Path, help="Directory containing split ZIP files (default: beside manifest)")
    parser.add_argument("--source", type=Path, help="Repository checkout for original 56-frame face-count verification")
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest, args.archives or args.manifest.parent, args.source), indent=2))
