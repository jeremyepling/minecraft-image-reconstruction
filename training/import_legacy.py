"""Stage the original exterior regression captures, preserving their published bytes."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (
    "camera-round-01",
    "camera-gray-concrete-v1",
    "surface-confirm-v1",
    "normal-confirm-v1",
)
REFERENCE = "benchmark/data/visible-geometry-v1.zip"
REFERENCE_SHA256 = "91efa6d3e992b1a68672eae1fdd6d380e87724742a1a4b6cbe5f84d284f731a1"
ARTIFACTS = dict(image="image.png", pose="pose.json", states="readback.json", labels="faces.json")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def stage(source, destination):
    reference = source / REFERENCE
    if sha256(reference.read_bytes()) != REFERENCE_SHA256:
        raise ValueError(f"Unexpected frozen reference archive: {reference}")
    with ZipFile(reference) as archive:
        references = json.loads(archive.read("index.json"))["frames"]
    rows = []
    for fixture in FIXTURES:
        directory = source / "benchmark/data" / fixture
        index = json.loads((directory / "index.json").read_bytes())
        floor = index.get("floor", "smooth_stone")
        for frame in index["frames"]:
            key = f"{floor}/{frame['id']}"
            sample_id = "legacy-" + key.replace("/", "-")
            folder = destination / sample_id
            folder.mkdir()
            expected = references[key]["inputs"]
            paths = {}
            for field, filename in ARTIFACTS.items():
                path = directory / frame[field]
                payload = path.read_bytes()
                if sha256(payload) != expected[field]:
                    raise ValueError(f"Checksum mismatch for {path}")
                (folder / filename).write_bytes(payload)
                paths[field] = path.relative_to(source).as_posix()

            ground = index["ground_y"]
            structure = [
                block for block in json.loads((folder / "readback.json").read_bytes())
                if block["y"] > ground and block["block_id"] != "air"
            ]
            origin = [min(block[axis] for block in structure) for axis in "xyz"]
            rows.append(dict(
                id=sample_id, split="regression", layout=frame["layout"],
                floor=floor, ground_y=ground, origin_world=origin,
                palette=frame["palette"], angle=frame["angle"], condition=frame["condition"],
                image=f"{sample_id}/image.png", pose=f"{sample_id}/pose.json",
                readback=f"{sample_id}/readback.json", visible_faces=f"{sample_id}/faces.json",
                legacy_key=key, legacy_split=frame["split"], legacy_fixture=fixture,
                source_paths=paths, source_sha256=expected, client_sha256=index["client_sha256"],
            ))
    if len(rows) != 56 or {row["legacy_key"] for row in rows} != set(references):
        raise ValueError("Staging must include the 56 archived frames exactly once")
    manifest = dict(frames=rows, reference_archive=REFERENCE, reference_sha256=REFERENCE_SHA256)
    (destination / "index.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT, help="Checkout with original exterior captures")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/dataset-legacy", help="New staging directory")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as scratch:
        destination = Path(scratch) / "staged"
        destination.mkdir()
        stage(args.source.resolve(), destination)
        destination.rename(args.out)
    print(f"Verified and staged all 56 original regression captures in {args.out}")


if __name__ == "__main__":
    main()
