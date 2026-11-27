"""Package native morning and afternoon re-captures of the 152 Visible Builds v1 views."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageDraw

from training.package_dataset import digest, encode
from training.visible_labels import visible_labels


CLOCK = {"morning": 2500, "afternoon": 9500}
ORIGINAL_SPLITS = ("train", "validation", "test", "regression")
SHARDS = (*ORIGINAL_SPLITS, "historical-diagnostic")
POSE_FIELDS = ("image_size", "vertical_fov_deg", "raining", "thundering", "hud_hidden",
               "minecraft_version", "resource_packs")
POSE_LIMITS = {"eye_world": 0.00002, "yaw_pitch_deg": 0.00002, "view_rotation_projection": 0.000002}
LABEL_FILES = ("labels.json", "labels.npz", "reference-faces.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def sources(data, frame_ids):
    pinned = read(Path(__file__).with_name("visible-builds-v1.json"))
    expected = {row["id"]: row for row in pinned["frames"]}
    audit = read(Path(__file__).with_name("framing-audit-v2.json"))
    historical = {row["id"]: row["reason"] for row in audit["frames"] if not row["accepted"]}
    require(len(historical) == 8, "Unexpected v1 historical diagnostic cohort")
    frames = {}
    for split in ORIGINAL_SPLITS:
        for line in (data / split / "frames.jsonl").read_bytes().splitlines():
            row = json.loads(line)
            fid = row["id"]
            require(fid in expected and fid not in frames and row["split"] == split
                    and row["directory"] == "frames/" + fid, f"Unexpected v1 frame: {fid}")
            require(all(row[key] == expected[fid][key] for key in ("layout", "palette", "split"))
                    and row["files"]["rgb.png"] == expected[fid]["rgb_sha256"], f"Original index mismatch: {fid}")
            frames[fid] = row, data / split / row["directory"], historical.get(fid)
    require(len(frames) == 152 and set(frames) == set(expected), "Original data must contain the pinned 152 frames")
    require(set(frame_ids).issubset(frames), "Unknown smoke-test source frame")
    return {fid: frames[fid] for fid in sorted(frame_ids or frames)}


def compare_camera(pose, original, fid, lighting):
    capture_id = f"{fid}-{lighting}"
    require(pose["id"] == capture_id and pose["clock_ticks"] % 24000 == CLOCK[lighting],
            f"Wrong capture identity or clock: {capture_id}")
    require(original["clock_ticks"] % 24000 == 6000, f"Expected noon source: {fid}")
    require(all(pose[key] == original[key] for key in POSE_FIELDS), f"Camera configuration differs: {capture_id}")
    require(not pose["raining"] and not pose["thundering"] and pose["hud_hidden"], f"Capture not clear and HUD-free: {capture_id}")
    errors = {}
    for key, limit in POSE_LIMITS.items():
        value, reference = np.asarray(pose[key]), np.asarray(original[key])
        require(value.shape == reference.shape, f"Camera vector length differs: {capture_id}/{key}")
        error = float(np.max(np.abs(value - reference)))
        require(error <= limit, f"Camera drift exceeds {limit}: {capture_id}/{key} ({error})")
        errors[key] = error
    return errors


def state_map(payload, fid):
    rows = json.loads(payload)
    states = {tuple(row[axis] for axis in "xyz"): row for row in rows}
    require(len(states) == len(rows), f"Duplicate readback coordinate: {fid}")
    return states


def state_signature(states):
    return {xyz: (row["block_id"], row["state"]) for xyz, row in states.items()}


def state_checksum(signature):
    return checksum(encode([(*xyz, *state) for xyz, state in sorted(signature.items())]))


def png_rgb(payload, fid):
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        require(image.format == "PNG" and image.mode in {"RGB", "RGBA"}, f"Expected native color PNG: {fid}")
        return np.asarray(image.convert("RGB"))


def verify_label_geometry(pose, original_pose, states, ground_y, labels, raster, errors):
    require(labels["frame_id"] == original_pose["id"] and labels["camera"]["eye_world"] == original_pose["eye_world"],
            f"Original camera and labels disagree: {pose['id']}")
    if not any(errors.values()):
        return
    # Camera drift can change the winning surface at a silhouette by a single pixel.
    # Reuse the frozen targets only if every pixel still has the same owner and depth.
    build = {point: row for point, row in states.items() if point[1] > ground_y}
    new, face_ids, depth = visible_labels(pose, build, labels["origin_world"])
    with np.load(io.BytesIO(raster), allow_pickle=False) as old:
        identity = [(face["xyz_world"], face["direction"], face["corner_visible"]) for face in labels["faces"]]
        recomputed = [(face["xyz_world"], face["direction"], face["corner_visible"]) for face in new["faces"]]
        require(identity == recomputed and np.array_equal(face_ids, old["face_id"])
                and np.allclose(depth, old["depth_z"], atol=2e-5, rtol=0, equal_nan=True),
                f"Fresh camera changes the original visible labels: {pose['id']}")


def package_frame(original, folder, historical_reason, capture, lighting, dataset):
    fid = original["id"]
    captured_id = f"{fid}-{lighting}"
    files = {name: (folder / name).read_bytes() for name in original["files"]}
    require({name: checksum(payload) for name, payload in files.items()} == original["files"],
            f"Original frame files changed: {fid}")
    assets = {name: files[name] for name in LABEL_FILES}
    assets.update({"rgb.png": (capture / (captured_id + ".png")).read_bytes(),
                   "camera.json": (capture / (captured_id + ".json")).read_bytes(),
                   "world-readback.json": (capture / (captured_id + "-blocks.json")).read_bytes(),
                   "noon-camera.json": files["camera.json"]})
    native_pose = json.loads(assets["camera.json"])
    noon_pose = json.loads(assets["noon-camera.json"])
    errors = compare_camera(native_pose, noon_pose, fid, lighting)
    states = state_map(assets["world-readback.json"], captured_id)
    original_signature = state_signature(state_map(files["world-readback.json"], fid))
    require(state_signature(states) == original_signature,
            f"Block coordinates or states differ from noon: {captured_id}")
    labels = json.loads(assets["labels.json"])
    verify_label_geometry(native_pose, noon_pose, states, original["ground_y"], labels, assets["labels.npz"], errors)
    lit_rgb = png_rgb(assets["rgb.png"], captured_id)
    noon_rgb = png_rgb(files["rgb.png"], fid)
    require(lit_rgb.shape == noon_rgb.shape and list(lit_rgb.shape[:2][::-1]) == native_pose["image_size"],
            f"PNG dimensions differ: {captured_id}")
    with np.load(io.BytesIO(assets["labels.npz"]), allow_pickle=False) as raster:
        visible = raster["face_id"] > 0
        delta = np.abs(lit_rgb[visible].astype(np.int16) - noon_rgb[visible])
        changed = dict(mean_absolute_channel_delta=float(delta.mean()),
                       fraction_visible_pixels_max_channel_delta_ge_8=float((delta.max(axis=1) >= 8).mean()))
    hashes = {name: checksum(payload) for name, payload in assets.items()}
    keys = ("id", "split", "layout", "palette", "angle", "floor", "ground_y", "origin_world", "counts", "client_sha256")
    row = {key: original[key] for key in keys}
    row.update(dataset=dataset, directory=f"frames/{fid}", family=original.get("family", original["layout"]),
               lighting=lighting, capture_id=captured_id, capture_cohort=capture.parent.name,
               clock_ticks=native_pose["clock_ticks"],
               shard="historical-diagnostic" if historical_reason else original["split"],
               whole_structure=not historical_reason, historical_reason=historical_reason,
               noon_file_sha256=original["files"], noon_state_sha256=state_checksum(original_signature),
               camera_max_delta=errors, verified_state_cells=len(states),
               visible_rgb_change_vs_noon=changed, identical_rgb_to_noon=bool(np.array_equal(lit_rgb, noon_rgb)),
               files=hashes)
    return row, assets


def archive_asset(archive, member, payload):
    info = ZipInfo(member, (2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED if member.endswith((".png", ".npz")) else ZIP_DEFLATED
    archive.writestr(info, payload)


def verify(release):
    manifest = read(release / "manifest.json")
    require(manifest["source_dataset"] == "visible-builds-v1" and manifest["schema_version"] == 1,
            "Unexpected release format")
    require(manifest["dataset"] == f"visible-builds-v1-{manifest['lighting']}" and
            manifest["clock_ticks"] == CLOCK[manifest["lighting"]] and set(manifest["shards"]) == set(SHARDS),
            "Lighting release configuration disagrees")
    complete = manifest["complete_152_frame_replay"]
    if complete:
        require({name: shard["frames"] for name, shard in manifest["shards"].items()}
                == dict(train=72, validation=12, test=12, regression=48, **{"historical-diagnostic": 8}),
                "Wrong complete release shard counts")
    listed = {row["id"]: row for row in manifest["frames"]}
    require(len(listed) == len(manifest["frames"]), "Duplicate release frame")
    seen, family_splits = set(), {}
    worst = dict.fromkeys(POSE_LIMITS, 0.)
    light = manifest["lighting"]
    state_cells = 0
    maximum_changed_fraction = 0.
    identical_images = identical_visible_pixels = 0
    for shard_name, shard in manifest["shards"].items():
        path = release / shard["file"]
        require(path.stat().st_size == shard["bytes"] and digest(path) == shard["sha256"], f"Invalid shard: {path.name}")
        with ZipFile(path) as archive:
            rows = [json.loads(line) for line in archive.read("frames.jsonl").splitlines()]
            require(len(rows) == shard["frames"] and len({r["layout"] for r in rows}) == shard["layouts"],
                    f"Wrong shard counts: {shard_name}")
            members = ["frames.jsonl"]
            for row in rows:
                fid = row["id"]
                require(row == listed.get(fid) and fid not in seen and row["shard"] == shard_name,
                        f"Frame membership disagrees: {fid}")
                require(family_splits.setdefault(row["family"], row["split"]) == row["split"], f"Family split leak: {fid}")
                require(row["whole_structure"] == (shard_name != "historical-diagnostic") and
                        (row["split"] if row["whole_structure"] else "historical-diagnostic") == shard_name,
                        f"Framing cohort disagrees: {fid}")
                directory = row["directory"] + "/"
                require(json.loads(archive.read(directory + "frame.json")) == row, f"Frame metadata disagrees: {fid}")
                assets = {name: archive.read(directory + name) for name in row["files"]}
                require({name: checksum(payload) for name, payload in assets.items()} == row["files"], f"Asset hashes disagree: {fid}")
                require(all(row["files"][name] == row["noon_file_sha256"][name] for name in LABEL_FILES)
                        and row["files"]["noon-camera.json"] == row["noon_file_sha256"]["camera.json"],
                        f"Frozen asset provenance disagrees: {fid}")
                pose = json.loads(assets["camera.json"])
                errors = compare_camera(pose, json.loads(assets["noon-camera.json"]), fid, light)
                require(errors == row["camera_max_delta"] and row["clock_ticks"] == pose["clock_ticks"]
                        and row["capture_id"] == pose["id"],
                        f"Camera audit disagrees: {fid}")
                states = state_map(assets["world-readback.json"], fid)
                require(len(states) == row["verified_state_cells"] and
                        state_checksum(state_signature(states)) == row["noon_state_sha256"],
                        f"Block state audit disagrees: {fid}")
                for key, error in errors.items():
                    worst[key] = max(worst[key], error)
                maximum_changed_fraction = max(maximum_changed_fraction,
                    row["visible_rgb_change_vs_noon"]["fraction_visible_pixels_max_channel_delta_ge_8"])
                identical_images += row["identical_rgb_to_noon"]
                identical_visible_pixels += row["visible_rgb_change_vs_noon"]["mean_absolute_channel_delta"] == 0
                state_cells += row["verified_state_cells"]
                members.extend(directory + name for name in (*assets, "frame.json"))
                seen.add(fid)
            require(sorted(archive.namelist()) == sorted(members), f"Duplicate or unindexed archive files: {shard_name}")
    require(seen == set(listed), "Manifest and archive membership differ")
    return dict(frames=len(seen), layouts=len({row["layout"] for row in listed.values()}),
                clock_ticks=CLOCK[light], verified_state_cell_appearances=state_cells, max_camera_delta=worst,
                maximum_visible_pixels_max_rgb_delta_ge_8_fraction=maximum_changed_fraction,
                pixel_identical_to_noon=identical_images, visible_structure_identical_to_noon=identical_visible_pixels)


def package(data, capture, lighting, out, only=None):
    frames = sources(data, only or [])
    selected = {}
    for path in capture.glob(f"*/captures/*-{lighting}.png"):
        fid = path.stem.removesuffix(f"-{lighting}")
        require(fid not in selected, f"Duplicate capture for {fid}")
        selected[fid] = path.parent
    require(set(frames).issubset(selected), f"Missing native captures: {sorted(set(frames) - set(selected))}")
    if not only:
        require(set(frames) == set(selected), f"Unexpected native captures: {sorted(set(selected) - set(frames))}")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset = f"visible-builds-v1-{lighting}"
    with TemporaryDirectory(dir=out.parent) as temporary:
        release = Path(temporary) / "release"
        release.mkdir()
        records, shards = [], {}
        for shard in SHARDS:
            rows = []
            name = f"{dataset}-{shard}.zip"
            target = release / name
            with ZipFile(target, "w") as archive:
                for fid, (original, folder, reason) in frames.items():
                    if ("historical-diagnostic" if reason else original["split"]) != shard:
                        continue
                    row, assets = package_frame(original, folder, reason, selected[fid], lighting, dataset)
                    for asset, payload in assets.items():
                        archive_asset(archive, f"{row['directory']}/{asset}", payload)
                    archive_asset(archive, f"{row['directory']}/frame.json", encode(row))
                    rows.append(row)
                archive_asset(archive, "frames.jsonl", b"".join(map(encode, rows)))
            records.extend(rows)
            shards[shard] = dict(file=name, frames=len(rows), layouts=len({r["layout"] for r in rows}),
                                 bytes=target.stat().st_size, sha256=digest(target))
        manifest = dict(dataset=dataset, schema_version=1, source_dataset="visible-builds-v1", minecraft_version="26.3",
                        lighting=lighting, clock_ticks=CLOCK[lighting], complete_152_frame_replay=len(records) == 152,
                        shards=shards, frames=records)
        (release / "manifest.json").write_bytes(encode(manifest))
        result = verify(release)
        (release / "verification.json").write_bytes(encode(result))
        release.rename(out)
    return result


def preview(data, morning, afternoon, out):
    examples = ("train-arcbridge3-cool-high-right", "train-watchpost3-earth-low-left",
                "test-twinspire2-cool-low-left", "legacy-smooth_stone-holdout-courtyard-plaster-high-right")
    noon_rows = sources(data, examples)
    captures = [("Morning  2500", morning), ("Noon  6000", None), ("Afternoon  9500", afternoon)]
    display = Image.new("RGB", (1468, 1368), "#182027")
    draw = ImageDraw.Draw(display)
    for column, (title, _) in enumerate(captures):
        draw.text((14 + 486 * column, 10), title, fill="white", font_size=22)
    for index, fid in enumerate(examples):
        row, source, _ = noon_rows[fid]
        top = 46 + 329 * index
        draw.text((14, top), fid, fill="white", font_size=16)
        for column, (_, release) in enumerate(captures):
            if release is None:
                payload = (source / "rgb.png").read_bytes()
            else:
                manifest = read(release / "manifest.json")
                shard = manifest["shards"][row["split"]]
                with ZipFile(release / shard["file"]) as archive:
                    payload = archive.read(row["directory"] + "/rgb.png")
            with Image.open(io.BytesIO(payload)) as image:
                tile = image.convert("RGB").resize((480, 300), Image.Resampling.LANCZOS)
                display.paste(tile, (8 + 486 * column, top + 23))
    with out.open("xb") as stream:
        display.save(stream, "JPEG", quality=93, subsampling=0)
    return dict(examples=len(examples), preview=str(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("build", help="Package and verify native lighting captures")
    build.add_argument("--original-data", type=Path, required=True)
    build.add_argument("--captures", type=Path, required=True, help="Parent of cohort capture directories")
    build.add_argument("--lighting", choices=CLOCK, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--only", nargs="+", help="Package only these original frame IDs for a smoke test")
    check = actions.add_parser("verify", help="Verify a previously packaged release")
    check.add_argument("--release", type=Path, required=True)
    contact_sheet = actions.add_parser("preview", help="Draw paired morning, noon and afternoon views")
    contact_sheet.add_argument("--original-data", type=Path, required=True)
    contact_sheet.add_argument("--morning", type=Path, required=True)
    contact_sheet.add_argument("--afternoon", type=Path, required=True)
    contact_sheet.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "build":
        result = package(args.original_data, args.captures, args.lighting, args.out, args.only)
    elif args.action == "preview":
        result = preview(args.original_data, args.morning, args.afternoon, args.out)
    else:
        result = verify(args.release)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
