"""Package original RGB, native camera/readback and per-frame visible labels."""
import argparse
import hashlib
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from PIL import Image
from training.framing import MIN_MARGIN_FRACTION, check_frame
from training.visible_labels import write_labels

DATASET = "visible-builds-v2"
SPLITS = ("train", "validation", "test", "regression")


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


def capture_records(capture):
    options = {}
    fields = {"ao", "gamma", "biomeBlendRadius", "mipmapLevels", "entityShadows", "renderClouds",
              "particles", "renderDistance", "simulationDistance", "bobView", "fovEffectScale"}
    for line in (capture / "client/options.txt").read_text(encoding="utf8").splitlines():
        key, _, value = line.partition(":")
        if key in fields:
            options[key] = value
    docs = {name: read(capture / name) for name in ("views.json", "buildings.json", "layout.json", "index.json")}
    docs["saved-client-options.json"] = options
    return docs


def stage_v1(download, candidates):
    # Read only the five archives fingerprinted by the original published manifest.
    manifest = read(Path(__file__).with_name("visible-builds-v1.json"))
    records = []
    for split, info in manifest["shards"].items():
        path = download / info["file"]
        if digest(path) != info["sha256"]:
            raise ValueError(f"Original split checksum mismatch: {path}")
        with ZipFile(path) as archive:
            archive.extractall(candidates / split)
            records.extend(json.loads(line) for line in archive.read("frames.jsonl").splitlines())
    info = manifest["capture_records"]
    path = download / info["file"]
    if digest(path) != info["sha256"]:
        raise ValueError(f"Original capture records checksum mismatch: {path}")
    with ZipFile(path) as archive:
        docs = {name: json.loads(archive.read(name)) for name in archive.namelist()}
    return records, docs


def write_capture_records(path, docs, records):
    ids = {r["id"] for r in records if r["split"] != "regression"}
    index = docs["index.json"]
    original = index["frames"]
    index["frames"] = [r for r in original if r["id"] in ids]
    if len(index["frames"]) != len(original):
        # The acquisition-level audit describes the original collection.
        index.pop("verified", None)
        index.pop("palette_audit", None)
    docs["views.json"] = [v for v in docs["views.json"] if v["id"] in ids]
    pages = {v["page"] for v in docs["views.json"]}
    docs["buildings.json"] = [b for b in docs["buildings.json"] if b["page"] in pages]
    layout = docs["layout.json"]
    layout["blocks"] = [b for b in layout["blocks"] if b["page"] in pages]
    with ZipFile(path, "w") as archive:
        for name in sorted(docs):
            info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, encode(docs[name]))


def write_release(candidates, rows, docs, out):
    records = []
    for split in SPLITS:
        (out / "data" / split / "frames").mkdir(parents=True)
    for row in sorted(rows, key=lambda r: r["id"]):
        frame = candidates / row["split"] / row["directory"]
        if check_frame(frame)["accepted"]:
            row["dataset"] = DATASET
            (frame / "frame.json").write_bytes(encode(row))
            frame.rename(out / "data" / row["split"] / row["directory"])
            records.append(row)

    shards = {}
    for split in SPLITS:
        root = out / "data" / split
        rows = [r for r in records if r["split"] == split]
        (root / "frames.jsonl").write_bytes(b"".join(encode(r) for r in rows))
        target = out / f"{DATASET}-{split}.zip"
        with ZipFile(target, "w") as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    add(archive, path, path.relative_to(root).as_posix())
        shards[split] = dict(file=target.name, sha256=digest(target), bytes=target.stat().st_size,
                             frames=len(rows), layouts=len({r["layout"] for r in rows}))
    sources_zip = out / f"{DATASET}-capture-records.zip"
    write_capture_records(sources_zip, docs, records)
    manifest = dict(dataset=DATASET, schema_version=1, minecraft_version="26.3", shards=shards,
                    min_margin_fraction=MIN_MARGIN_FRACTION,
                    capture_records=dict(file=sources_zip.name, sha256=digest(sources_zip),
                                         bytes=sources_zip.stat().st_size),
                    frames=[dict(id=r["id"], layout=r["layout"], split=r["split"], palette=r["palette"],
                                 rgb_sha256=r["files"]["rgb.png"]) for r in records])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")


def build(sources, out, workers=4, from_v1=None):
    from training.verify_dataset import verify
    if out.exists():
        raise FileExistsError(f"Use a new destination: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as scratch:
        candidates = Path(scratch) / "candidates"
        release = Path(scratch) / "release"
        if from_v1:
            records, docs = stage_v1(from_v1, candidates)
        else:
            jobs = [(src, row, candidates) for src in sources for row in read(src / "index.json")["frames"]]
            if any(row["split"] not in SPLITS for _, row, _ in jobs):
                raise ValueError("Unexpected split")
            with ProcessPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(prepare, jobs))
            docs = capture_records(next(src for src in sources if (src / "views.json").is_file()))
        write_release(candidates, records, docs, release)
        result = verify(release / "manifest.json", release)
        (release / "verification.json").write_bytes(encode(result))
        release.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sources", type=Path, nargs=2, help="Legacy and native capture staging directories")
    inputs.add_argument("--from-v1", type=Path, help="Directory with the five frozen v1 archives")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(args.sources, args.out, args.workers, args.from_v1)
