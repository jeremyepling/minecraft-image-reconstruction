"""Build and load the layout-level selection for floating reconstruction work."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


MANIFEST = Path(__file__).with_name("floating-active-cohort-v1.json")
COHORT = "floating-active-cohort-v1"
RELEASES = {
    "visible-builds-v2": "cd8bfb2953dd3418d0c3666a76cd12f2e0c2193f31e7545a93c1dca13d01e34f",
    "visible-builds-v3": "6e7b25fa56aa39d04a34bb3984551cf11441c11acb1a021b045d4f86e824aa28",
    "decorative-builds-v1": "09d5612ee175718b979c73363eb40fe05959d90f96aab894e82efa9548eaf064",
}
SPLITS = ("train", "validation", "test")
MIN_SMOOTH_PERCENT = 80
NAMED_EXCLUSIONS = {
    "visible-builds-v3": {"amphitheater-01", "amphitheater-02"},
}


def smooth_material(material):
    return material.startswith("minecraft:") and (
        material == "minecraft:terracotta" or material.endswith("_concrete")
        or (material.endswith("_terracotta") and not material.endswith("_glazed_terracotta"))
    )


def read_shard(release, dataset, split, spec, listed):
    path = release / spec["file"]
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != spec["sha256"]:
            raise ValueError(f"Release shard checksum mismatch: {path}")
    with ZipFile(path) as archive:
        rows = [json.loads(line) for line in archive.read("frames.jsonl").splitlines()]
        ids = {row["id"] for row in rows}
        if ids != listed.keys() or len(rows) != len(ids) or len(rows) != spec["frames"]:
            raise ValueError(f"Frame index mismatch: {path}")
        for row in rows:
            frame_id = row["id"]
            index = listed[frame_id]
            # Visible Builds v2 checks framing at packaging time and has no per-row framing field.
            if (row["dataset"] != dataset or row["split"] != split
                    or any(row[key] != index[key] for key in ("layout", "palette"))
                    or row["files"]["rgb.png"] != index["rgb_sha256"]
                    or ("framing" in row and not row["framing"]["accepted"])):
                raise ValueError(f"Frame provenance mismatch: {path}/{frame_id}")
            payload = archive.read(f"{row['directory']}/labels.json")
            if hashlib.sha256(payload).hexdigest() != row["files"]["labels.json"]:
                raise ValueError(f"Visible labels checksum mismatch: {path}/{frame_id}")
            labels = json.loads(payload)
            block_ids = {block["id"]: block["block_id"] for block in labels["blocks"]}
            visible = sum(face["visible_pixels"] for face in labels["faces"])
            smooth = sum(face["visible_pixels"] for face in labels["faces"]
                         if smooth_material(block_ids[face["block"]]))
            if visible <= 0 or visible != row["counts"]["visible_pixels"]:
                raise ValueError(f"Visible pixel count mismatch: {path}/{frame_id}")
            yield dict(id=frame_id, layout=row["layout"], palette=row["palette"],
                       visible_pixels=visible, smooth_pixels=smooth,
                       rgb_sha256=index["rgb_sha256"], reference_sha256={
                           key: row["files"][key] for key in ("camera.json", "labels.json", "labels.npz")
                       })


def score_layout(frames, explicitly_excluded):
    palettes = defaultdict(list)
    for frame in frames:
        palette = frame["palette"]
        palettes[palette].append({key: value for key, value in frame.items() if key not in ("layout", "palette")})
    coverage = {
        palette: dict(
            visible_pixels=sum(frame["visible_pixels"] for frame in views),
            smooth_pixels=sum(frame["smooth_pixels"] for frame in views),
            frames=sorted(views, key=lambda view: view["id"]),
        )
        for palette, views in sorted(palettes.items())
    }
    dominant = [
        palette for palette, counts in coverage.items()
        if 100 * counts["smooth_pixels"] >= MIN_SMOOTH_PERCENT * counts["visible_pixels"]
    ]
    reasons = []
    if explicitly_excluded:
        reasons.append("reviewer-exclusion")
    if dominant:
        reasons.append("smooth-material-coverage")
    return dict(active=not reasons, reasons=reasons, dominant_palettes=dominant, palettes=coverage)


def build(archives_root):
    datasets = {}
    for dataset, expected in RELEASES.items():
        release = archives_root / dataset
        manifest_bytes = (release / "manifest.json").read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != expected:
            raise ValueError(f"Unexpected release manifest: {release}")
        release_manifest = json.loads(manifest_bytes)
        listed = release_manifest["frames"]
        splits = {}
        named = NAMED_EXCLUSIONS.get(dataset, set())
        split_by_layout = {}
        for split in SPLITS:
            spec = release_manifest["shards"][split]
            index = {frame["id"]: frame for frame in listed if frame["split"] == split}
            layouts = defaultdict(list)
            for frame in read_shard(release, dataset, split, spec, index):
                layout = frame["layout"]
                if split_by_layout.setdefault(layout, split) != split:
                    raise ValueError(f"Layout crosses frozen splits in {dataset}: {layout}")
                layouts[layout].append(frame)
            if len(layouts) != spec["layouts"]:
                raise ValueError(f"Layout count mismatch in {dataset}/{split}")
            results = {layout: score_layout(frames, layout in named) for layout, frames in sorted(layouts.items())}
            splits[split] = dict(
                source_archive=spec["file"], source_archive_sha256=spec["sha256"],
                available_frames=spec["frames"],
                active_frames=sum(len(layouts[key]) for key, layout in results.items() if layout["active"]),
                layouts=results,
            )
        if named - split_by_layout.keys():
            raise ValueError(f"Missing named exclusions in {dataset}: {sorted(named - split_by_layout.keys())}")
        datasets[dataset] = dict(release_manifest_sha256=expected, splits=splits)
    return dict(
        cohort=COHORT, min_smooth_percent=MIN_SMOOTH_PERCENT,
        material_scope="Minecraft concrete and unglazed terracotta",
        threshold_scope="frontmost visible building pixels pooled across the camera views of each layout and palette",
        exclusion_scope="all views and palettes of the layout in its original split",
        datasets=datasets,
    )


def frame_index(path=MANIFEST):
    """Index both active and excluded release captures by (dataset, frame ID)."""
    manifest = json.loads(path.read_bytes())
    if manifest["cohort"] != COHORT:
        raise ValueError(f"Unexpected floating cohort: {path}")
    indexed = {}
    for dataset, release in manifest["datasets"].items():
        for split, partition in release["splits"].items():
            for layout, fields in partition["layouts"].items():
                for palette, coverage in fields["palettes"].items():
                    for frame in coverage["frames"]:
                        key = dataset, frame["id"]
                        if key in indexed:
                            raise ValueError(f"Duplicate floating cohort entry: {'/'.join(key)}")
                        indexed[key] = dict(frame, active=fields["active"], split=split, layout=layout, palette=palette)
    return indexed


def select_active_cases(cases, path=MANIFEST):
    """Filter saved benchmark cases by release/layout policy, in original input order."""
    index = frame_index(path)
    for case in cases:
        key = case["dataset"], case["id"]
        if key not in index:
            raise ValueError(f"Case is outside the floating cohort: {'/'.join(key)}")
        frame = index[key]
        if any(case[field] != frame[field] for field in ("rgb_sha256", "split", "layout", "palette") if field in case):
            raise ValueError(f"Case differs from its floating cohort entry: {'/'.join(key)}")
        if frame["active"]:
            yield case


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archives-root", required=True, type=Path,
                     help="Directory with one subdirectory per dataset, containing the published release files")
    cli.add_argument("--out", type=Path, default=MANIFEST)
    args = cli.parse_args()
    result = build(args.archives_root)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
