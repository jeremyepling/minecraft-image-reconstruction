"""Package the original 204 placement views as a companion to Visible Builds v3."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from benchmark.facade_inset_round import SHELTER, framed, jobs
from benchmark.round_report import FIXTURE, LAYOUTS, load_release
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.thin_face_placement_run import COHORTS
from calibration.world import GROUND
from training.package_dataset import add, digest, encode, read
from training.verify_dataset import verify_frame
from training.visible_labels import write_labels


DATASET = "visible-builds-v3-placement-reference"
ARCHIVE = "visible-builds-v3-original-204.zip"
MANIFEST = "visible-builds-v3-original-204-manifest.json"
V2_SPLITS = ("train", "validation", "test", "regression")


def asset_paths(split, frame):
    if split in V2_SPLITS:
        folder = framed.TRAINING_DATA / "data" / split / frame["directory"]
        return {key: folder / key for key in frame["files"]}
    if split == "native":
        rgb = framed.native.rgb_path(frame, framed.native.BUILDS)
    else:
        root = {"new": framed.NEW_INPUTS, "shelter": SHELTER}.get(split, framed.transfer.BUILDS)
        rgb = root / frame["directory"] / "captures" / (frame["source_id"] + ".png")
    paths = {"rgb.png": rgb, "camera.json": rgb.with_suffix(".json"),
             "world-readback.json": rgb.with_name(rgb.stem + "-blocks.json")}
    if "raw-readback.json" in frame["files"]:
        paths["raw-readback.json"] = root / "native/captures" / (rgb.stem + "-blocks.json")
        for name in ("labels.json", "labels.npz"):
            paths[name] = root / "truth" / frame["id"] / name
    return paths


def verify_target(row, labels, face_ids, reference):
    strict, facade = (reference["result"][key] for key in ("strict", "facade"))
    visible_blocks = len(labels["blocks"])
    optional = len(horizontal_slivers(labels, face_ids))
    if (visible_blocks != strict["correct"] + strict["missed"]
            or visible_blocks - optional != facade["correct"] + facade["missed"]):
        raise ValueError(f"Visible target differs from the original scoreboard: {row['id']}")


def prepare(task):
    split, frame, case, cohort, layout, paths, output = task
    frame_id = frame["id"]
    directory = "frames/" + frame_id
    label_folder = output / directory
    assets = {name: digest(path) for name, path in paths.items()}
    if assets != frame["files"] or assets["rgb.png"] != case["source_hash"]:
        raise ValueError(f"Frozen input fingerprint mismatch: {frame_id}")
    fields = ("family", "layout", "shape", "palette", "angle", "floor", "condition",
              "source_id", "source_paths", "source_sha256")
    row = {key: frame[key] for key in fields if key in frame}
    row.update(id=frame_id, split=split, benchmark_group=cohort,
               benchmark_cohort="fixed_104" if cohort == "fixed" else "controls_100",
               dataset=DATASET, directory=directory)
    if layout:
        row["benchmark_layout"], row["benchmark_shape_sha256"] = layout
    if "labels.json" not in paths:
        pose = read(paths["camera.json"])
        actual = read(paths["world-readback.json"])
        ground = frame.get("ground_y", GROUND)
        states = {tuple(block[key] for key in "xyz"): block for block in actual
                  if block["y"] > ground and block["block_id"].removeprefix("minecraft:") != "air"}
        origin = [min(pos[axis] for pos in states) for axis in range(3)]
        write_labels(pose, states, origin, label_folder)
        for name in ("labels.json", "labels.npz"):
            paths[name] = label_folder / name
            assets[name] = digest(paths[name])
        row["label_source"] = "reconstructed from original native camera and server readback"
    else:
        row["label_source"] = ("byte-identical Visible Builds v2" if split in V2_SPLITS
                               else "byte-identical original capture labels")

    labels = read(paths["labels.json"])
    with np.load(paths["labels.npz"], allow_pickle=False) as raster:
        face_ids = raster["face_id"]
        verify_target(row, labels, face_ids, case)
        row["counts"] = dict(visible_blocks=len(labels["blocks"]), visible_faces=len(labels["faces"]),
                             visible_pixels=int(np.count_nonzero(face_ids)),
                             visible_surfaces=labels["surface_count"])
    row.update(files=assets, ground_y=frame.get("ground_y", GROUND), origin_world=labels["origin_world"])
    return row, paths


def add_bytes(archive, name, payload):
    entry = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    entry.compress_type = ZIP_DEFLATED
    archive.writestr(entry, payload)


def verify(path, manifest, references, layout_index):
    with ZipFile(path) as archive:
        if json.loads(archive.read("manifest.json")) != manifest:
            raise ValueError("Archive manifest changed")
        original = archive.read("placement-cases.json")
        if original != (FIXTURE / "cases.json").read_bytes():
            raise ValueError("Original scoreboard changed")
        members = {"manifest.json", "frames.jsonl", "placement-cases.json"}
        indexed = [json.loads(line) for line in archive.read("frames.jsonl").splitlines()]
        expected = {row["id"]: row for row in manifest["frames"]}
        snapshots = Counter()
        cohorts = Counter()
        seen = set()
        for row in indexed:
            frame_id = row["id"]
            reference = references[frame_id]
            if frame_id in seen or row["split"] != reference["split"]:
                raise ValueError(f"Duplicate frame or changed split: {frame_id}")
            if (row["benchmark_cohort"] == "fixed_104") != (frame_id in layout_index):
                raise ValueError(f"Changed benchmark cohort: {frame_id}")
            verify_frame(archive, row, expected[frame_id], None, DATASET)
            snapshots[row["label_source"]] += 1
            cohorts[row["benchmark_cohort"]] += 1
            members.update(f"{row['directory']}/{name}" for name in (*row["files"], "frame.json"))
            seen.add(frame_id)
        if seen != references.keys() or sorted(archive.namelist()) != sorted(members):
            raise ValueError("Unexpected archive membership")
        if cohorts != {"fixed_104": 104, "controls_100": 100}:
            raise ValueError(f"Wrong cohort totals: {cohorts}")
    return dict(frames=len(seen), members=len(members), cohorts=dict(cohorts), label_sources=dict(snapshots))


def package(out, workers):
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    scoreboard, references = load_release(FIXTURE)
    layout_index = {frame_id: (layout["name"], layout["shape_sha256"])
                    for layout in read(LAYOUTS)["all_layouts"] for frame_id in layout["frame_ids"]}
    source_jobs = list(jobs([*framed.SIZES, "shelter"]))
    split_to_group = {split: group for group, splits in COHORTS.items() for split in splits}
    if (len(source_jobs) != 204 or len(layout_index) != 104
            or {frame["id"] for _, frame, *_ in source_jobs} != references.keys()
            or {frame["id"] for split, frame, *_ in source_jobs if split_to_group[split] == "fixed"}
            != layout_index.keys()):
        raise ValueError("Expected exactly the original 204 benchmark views and their fixed layout membership")

    with TemporaryDirectory(dir=out.parent) as scratch:
        work = Path(scratch)
        pending = work / "release"
        pending.mkdir()
        tasks = [(split, frame, references[frame["id"]], split_to_group[split],
                  layout_index.get(frame["id"]), asset_paths(split, frame), work)
                 for split, frame, *_ in source_jobs]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = sorted(pool.map(prepare, tasks), key=lambda item: item[0]["id"])
        print(f"Prepared and checked {len(rows)} original fixture views.", flush=True)
        manifest = dict(dataset=DATASET, schema_version=1, minecraft_version="26.3",
                        purpose="Historical placement benchmark replay. Preserve the original splits.",
                        cohorts={"fixed_104": 104, "controls_100": 100},
                        groups=dict(Counter(row["benchmark_group"] for row, _ in rows)),
                        splits=dict(Counter(row["split"] for row, _ in rows)),
                        benchmark_cases_sha256=scoreboard["cases_sha256"],
                        frames=[dict(id=row["id"], rgb_sha256=row["files"]["rgb.png"], split=row["split"],
                                     benchmark_group=row["benchmark_group"],
                                     benchmark_cohort=row["benchmark_cohort"]) for row, _ in rows])
        path = pending / ARCHIVE
        with ZipFile(path, "w") as archive:
            add_bytes(archive, "manifest.json", encode(manifest))
            add_bytes(archive, "frames.jsonl", b"".join(encode(row) for row, _ in rows))
            add(archive, FIXTURE / "cases.json", "placement-cases.json")
            for row, paths in rows:
                add_bytes(archive, f"{row['directory']}/frame.json", encode(row))
                for name, source in sorted(paths.items()):
                    add(archive, source, f"{row['directory']}/{name}")

        result = verify(path, manifest, references, layout_index)
        result.update(file=path.name, bytes=path.stat().st_size, sha256=digest(path))
        (pending / MANIFEST).write_bytes(encode(dict(**manifest, archive=result)))
        pending.rename(out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    package(args.out.resolve(), args.workers)
