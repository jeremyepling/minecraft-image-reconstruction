"""Package and verify the native full-cube and decorative datasets."""
import argparse
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image, ImageDraw

from training.framing import MIN_MARGIN_FRACTION, check_frame
from training.package_dataset import add, digest, encode, read
from training.verify_dataset import require, verify_frame
from training.visible_labels import write_labels


SPLITS = ("train", "validation", "test")
ASSETS = {"rgb.png": ".png", "camera.json": ".json", "world-readback.json": "-blocks.json"}
DATASETS = {"visible-builds-v3": (24, 144), "decorative-builds-v1": (10, 40)}
PREDECESSOR = {"dataset": "visible-builds-v2", "manifest": "training/visible-builds-v2.json",
               "comparison": "benchmark/reviews/workshop-foreground-01/cases.json",
               "comparison_frames": 204, "fixed_score_frames": 104}


def xyz(row):
    return tuple(row[axis] for axis in "xyz")


def properties(state):
    # Compare parsed property values because the server emits canonical property order.
    match = re.search(r"\[([^\]]*)\]$", state)
    return dict(field.split("=", 1) for field in match[1].split(",")) if match and match[1] else {}


def prepare(job):
    source, data, dataset, row, expected, site, ground, client_hash, wait = job
    fid = row["id"]
    stem = source / "captures" / fid
    pose_file = stem.with_suffix(".json")
    if wait:
        deadline = time.monotonic() + 180
        while not pose_file.is_file() or not pose_file.stat().st_size:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Capture did not arrive: {fid}")
            time.sleep(0.5)
    pose = read(pose_file)
    require(pose["id"] == fid and pose["minecraft_version"] == "26.3"
            and pose["clock_ticks"] % 24_000 == 6000 and pose["hud_hidden"]
            and not pose["raining"] and not pose["thundering"],
            f"Unexpected capture conditions: {fid}")
    readback = read(stem.with_name(stem.name + "-blocks.json"))
    actual = {xyz(block): block for block in readback}
    require(len(actual) == len(readback) == len(expected) and actual.keys() == expected.keys(),
            f"Readback coordinates differ: {fid}")
    states = {}
    for point, planned in expected.items():
        block = actual[point]
        require(block["block_id"] == planned["block_id"], f"Unexpected block at {fid}/{point}: {block}")
        if "placement_state" in planned:
            specified = properties(planned["placement_state"])
            found = properties(block["state"])
            require(specified.items() <= found.items(), f"State changed at {fid}/{point}: {block['state']}")
        if point[1] > ground:
            states[point] = block

    split, layout = row["split"], row["layout"]
    frame = data / split / "frames" / fid
    frame.mkdir(parents=True)
    for name, suffix in ASSETS.items():
        shutil.copyfile(stem.with_name(stem.name + suffix), frame / name)
    origin, end = site["bounds"]
    if dataset == "decorative-builds-v1":
        from training.decorative_labels import write_labels as write_decorative_labels

        counts = write_decorative_labels(pose, states, origin, frame)
        require(counts["visible_decorative_blocks"] and counts["visible_full_blocks_behind_decorations"],
                f"Missing visible foreground/backdrop overlap: {fid}")
    else:
        counts = write_labels(pose, states, origin, frame)
    materials = Counter(b["block_id"] for b in states.values())
    metadata = dict(row, dataset=dataset, family=layout.rsplit("-", 1)[0], directory=f"frames/{fid}",
                    client_sha256=client_hash, origin_world=origin, ground_y=ground,
                    build_size_xyz=[hi - lo for lo, hi in zip(origin, end)],
                    counts={k: v for k, v in counts.items() if k.startswith("visible_")},
                    material_ids=sorted(materials), clock_ticks=pose["clock_ticks"] % 24_000,
                    files={name: digest(frame / name) for name in (*ASSETS, "labels.json", "labels.npz")})
    check = check_frame_with_metadata(frame, metadata)
    metadata["framing"] = check
    (frame / "frame.json").write_bytes(encode(metadata))
    return metadata


def check_frame_with_metadata(frame, metadata):
    (frame / "frame.json").write_bytes(encode(metadata))
    framing = check_frame(frame)
    require(framing["accepted"], f"Incomplete framing for {metadata['id']}: {framing}")
    return framing


def contact_sheet(data, rows, destination):
    examples = {}
    for row in rows:
        examples.setdefault(row["layout"], row)
    columns, width, height = 4, 320, 222
    sheet = Image.new("RGB", (columns * width, height * ((len(examples) + columns - 1) // columns)), "#161b22")
    draw = ImageDraw.Draw(sheet)
    for i, (layout, row) in enumerate(sorted(examples.items())):
        x, y = i % columns * width, i // columns * height
        with Image.open(data / row["split"] / row["directory"] / "rgb.png") as original:
            shot = original.convert("RGB")
            shot.thumbnail((304, 190), Image.Resampling.LANCZOS)
            sheet.paste(shot, (x + 8, y + 8))
        w, h, d = row["build_size_xyz"]
        draw.text((x + 8, y + 201), f"{layout}  |  {w} x {h} x {d}", fill="#f0f4f8")
    sheet.save(destination, quality=92)


def verify(manifest_path):
    manifest = read(manifest_path)
    root = manifest_path.parent
    dataset = manifest["dataset"]
    require(dataset in DATASETS and manifest["schema_version"] == 1, "Unsupported dataset")
    require(set(manifest["shards"]) == set(SPLITS)
            and manifest["min_margin_fraction"] == MIN_MARGIN_FRACTION, "Incorrect splits or framing policy")
    listed = {row["id"]: row for row in manifest["frames"]}
    require(len(listed) == len(manifest["frames"]), "Duplicate frame ID")
    seen, images, layouts, families = set(), set(), {}, {}
    blocks = faces = overlaps = 0
    for split, shard in manifest["shards"].items():
        path = root / shard["file"]
        require(path.stat().st_size == shard["bytes"] and digest(path) == shard["sha256"], f"Corrupt shard: {split}")
        with ZipFile(path) as archive:
            import json

            rows = [json.loads(line) for line in archive.read("frames.jsonl").splitlines()]
            require(len(rows) == shard["frames"] and len({r["layout"] for r in rows}) == shard["layouts"],
                    f"Unexpected shard size: {split}")
            paths = {"frames.jsonl"}
            for row in rows:
                fid, layout, family = row["id"], row["layout"], row["family"]
                image_hash = row["files"]["rgb.png"]
                require(fid not in seen and fid in listed and image_hash not in images
                        and listed[fid]["rgb_sha256"] == image_hash
                        and all(listed[fid][k] == row[k] for k in ("layout", "family", "split", "palette"))
                        and row["split"] == split, f"Repeated or inconsistent frame: {fid}")
                require(layouts.setdefault(layout, split) == split and families.setdefault(family, split) == split,
                        f"Layout/family shared between splits: {fid}")
                visible, face_count = verify_frame(archive, row, listed[fid], None, dataset)
                blocks += visible
                faces += face_count
                paths.update(row["directory"] + "/" + name for name in (*row["files"], "frame.json"))
                overlaps += bool(row["counts"].get("visible_full_blocks_behind_decorations"))
                seen.add(fid)
                images.add(image_hash)
            require(set(archive.namelist()) == paths and len(archive.namelist()) == len(paths),
                    f"Unexpected archive entries: {split}")
    require(seen == listed.keys() and (len(layouts), len(seen)) == DATASETS[dataset], "Wrong dataset membership")
    provenance = manifest["capture_records"]
    path = root / provenance["file"]
    require(path.stat().st_size == provenance["bytes"] and digest(path) == provenance["sha256"],
            "Corrupt capture records")
    with ZipFile(path) as archive:
        require(archive.testzip() is None, "Invalid capture records ZIP")
    return dict(frames=len(seen), layouts=len(layouts), families=len(families),
                visible_block_appearances=blocks, visible_face_appearances=faces,
                frames_with_visible_foreground_overlap=overlaps)


def build(source, dataset, out, workers, wait):
    if out.exists():
        raise FileExistsError(f"Use a fresh release path: {out}")
    views = read(source / "views.json")
    fixture = read(source / "layout.json")
    sites = {site["page"]: site for site in read(source / "buildings.json")}
    pages = defaultdict(dict)
    for block in fixture["blocks"]:
        pages[block["page"]][xyz(block)] = block
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temporary:
        release = Path(temporary) / "release"
        data = release / "data"
        jobs = [(source, data, dataset, view, pages[view["page"]], sites[view["page"]],
                 fixture["ground_y"], fixture["client_sha256"], wait) for view in views]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = []
            for row in pool.map(prepare, jobs):
                rows.append(row)
                print(f"[PACKAGE] {len(rows)}/{len(views)} {row['id']}", flush=True)
        shards = {}
        for split in SPLITS:
            selected = [r for r in rows if r["split"] == split]
            root = data / split
            (root / "frames.jsonl").write_bytes(b"".join(encode(r) for r in selected))
            target = release / f"{dataset}-{split}.zip"
            with ZipFile(target, "w") as archive:
                for path in sorted(p for p in root.rglob("*") if p.is_file()):
                    add(archive, path, path.relative_to(root).as_posix())
            shards[split] = dict(file=target.name, sha256=digest(target), bytes=target.stat().st_size,
                                 frames=len(selected), layouts=len({r["layout"] for r in selected}))
        records = release / f"{dataset}-capture-records.zip"
        with ZipFile(records, "w") as archive:
            for name in ("views.json", "buildings.json", "layout.json", "runtime-block-registry.json"):
                add(archive, source / name, name)
            add(archive, source / "client/options.txt", "saved-client-options.txt")
        frames = [dict(id=r["id"], layout=r["layout"], family=r["family"], split=r["split"],
                       palette=r["palette"], rgb_sha256=r["files"]["rgb.png"]) for r in rows]
        manifest = dict(dataset=dataset, schema_version=1, minecraft_version="26.3",
                        min_margin_fraction=MIN_MARGIN_FRACTION, shards=shards, frames=frames,
                        capture_records=dict(file=records.name, sha256=digest(records), bytes=records.stat().st_size))
        if dataset == "visible-builds-v3":
            manifest["predecessor"] = PREDECESSOR
        manifest_path = release / "manifest.json"
        manifest_path.write_bytes(encode(manifest))
        result = verify(manifest_path)
        result["material_count"] = len({m for row in rows for m in row["material_ids"]})
        result["max_size_xyz"] = [max(r["build_size_xyz"][axis] for r in rows) for axis in range(3)]
        result["minimum_projected_border_px"] = min(min(r["framing"]["projected_margins_px"]) for r in rows)
        (release / "verification.json").write_bytes(encode(result))
        contact_sheet(data, rows, release / "preview.jpg")
        release.rename(out)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait-for-capture", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(verify(args.out / "manifest.json"))
    elif args.capture and args.dataset and args.workers > 0:
        build(args.capture.resolve(), args.dataset, args.out.resolve(), args.workers, args.wait_for_capture)
    else:
        parser.error("packaging requires --capture, --dataset and at least one worker")
