"""Package original RGB, native camera/readback and per-frame visible labels."""
import argparse
import hashlib
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from PIL import Image
from training.visible_labels import write_labels

DATASET = "visible-builds-v1"
EXPECTED = {"train": 72, "validation": 12, "test": 12, "regression": 56}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode("utf8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(job):
    source, row, destination = job
    fid = row["id"]
    if not re.fullmatch(r"[\w.-]+", fid) or fid in {".", ".."}:
        raise ValueError(f"Invalid frame id {fid}")
    frame = destination / row["split"] / "frames" / fid
    frame.mkdir(parents=True, exist_ok=False)
    pose = read(source / row["pose"])
    points = read(source / row["readback"])
    states = {tuple(b[k] for k in "xyz"): b for b in points
              if b["block_id"].removeprefix("minecraft:") != "air" and b["y"] > row["ground_y"]}
    origin = [min(p[i] for p in states) for i in range(3)]
    if origin != row["origin_world"]:
        raise ValueError(f"Origin disagrees with readback: {fid}")

    files = {}
    for key, name in (("image", "rgb.png"), ("pose", "camera.json"), ("readback", "world-readback.json")):
        target = frame / name
        shutil.copyfile(source / row[key], target)
        files[name] = digest(target)
    if files["rgb.png"] != digest(source / row["image"]):
        raise ValueError(f"RGB content changed: {fid}")
    with Image.open(frame / "rgb.png") as image:
        image.load()
        if list(image.size) != pose["image_size"] or image.format != "PNG":
            raise ValueError(f"Image and camera disagree: {fid}")
    reference_faces = row.get("visible_faces", row.get("labels"))
    if reference_faces:
        shutil.copyfile(source / reference_faces, frame / "reference-faces.json")
        files["reference-faces.json"] = digest(frame / "reference-faces.json")
    counts = write_labels(pose, states, tuple(origin), frame)
    for name in ("labels.json", "labels.npz"):
        files[name] = digest(frame / name)
    metadata = {k: v for k, v in row.items()
                if k not in ("image", "pose", "readback", "visible_faces", "labels", "states")}
    metadata["counts"] = {k: v for k, v in counts.items() if k.startswith("visible_")}
    metadata.update(files=files, directory="frames/" + fid, dataset=DATASET)
    (frame / "frame.json").write_bytes(encode(metadata))
    return metadata


def add(archive, path, name):
    info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED if path.suffix in {".npz", ".png"} else ZIP_DEFLATED
    archive.writestr(info, path.read_bytes())


def build(sources, out, workers):
    if out.exists():
        raise FileExistsError(f"Use a new destination: {out}")
    jobs = [(source, row, out / "data") for source in sources for row in read(source / "index.json")["frames"]]
    ids = [r["id"] for _, r, _ in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate frame IDs")
    groups = {}
    for _, r, _ in jobs:
        name = r["layout"]
        if groups.setdefault(name, r["split"]) != r["split"]:
            raise ValueError(f"Layout crosses splits: {name}")
    families = {}
    for _, r, _ in jobs:
        group = r.get("family", r["layout"])
        if families.setdefault(group, r["split"]) != r["split"]:
            raise ValueError(f"Family crosses splits: {group}")
    for split, count in EXPECTED.items():
        if sum(r["split"] == split for _, r, _ in jobs) != count:
            raise ValueError(f"Wrong number of frames in {split}")
    if len(jobs) != sum(EXPECTED.values()):
        raise ValueError("Unexpected split")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = sorted(pool.map(prepare, jobs), key=lambda r: r["id"])
    shards = {}
    for split in EXPECTED:
        root = out / "data" / split
        (root / "frames.jsonl").write_bytes(b"".join(encode(r) for r in records if r["split"] == split))
        target = out / f"{DATASET}-{split}.zip"
        with ZipFile(target, "w") as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    add(archive, path, path.relative_to(root).as_posix())
        with ZipFile(target) as archive:
            if archive.testzip():
                raise ValueError(f"Archive integrity failure: {target}")
        shards[split] = dict(file=target.name, sha256=digest(target), bytes=target.stat().st_size,
                             frames=EXPECTED[split], layouts=sum(v == split for v in groups.values()))
    capture = next(src for src in sources if (src / "views.json").is_file())
    sources_zip = out / f"{DATASET}-capture-records.zip"
    options = {}
    fields = {"ao", "gamma", "biomeBlendRadius", "mipmapLevels", "entityShadows", "renderClouds",
              "particles", "renderDistance", "simulationDistance", "bobView", "fovEffectScale"}
    for line in (capture / "client/options.txt").read_text(encoding="utf8").splitlines():
        key, _, value = line.partition(":")
        if key in fields:
            options[key] = value
    with ZipFile(sources_zip, "w") as archive:
        for name in ("views.json", "buildings.json", "layout.json", "index.json"):
            add(archive, capture / name, name)
        archive.writestr(ZipInfo("saved-client-options.json", (2026, 1, 1, 0, 0, 0)), encode(options))
    manifest = dict(dataset=DATASET, schema_version=1, minecraft_version="26.3", shards=shards,
                    capture_records=dict(file=sources_zip.name, sha256=digest(sources_zip),
                                         bytes=sources_zip.stat().st_size),
                    frames=[dict(id=r["id"], layout=r["layout"], split=r["split"], palette=r["palette"],
                                 rgb_sha256=r["files"]["rgb.png"]) for r in records])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(args.sources, args.out, args.workers)
