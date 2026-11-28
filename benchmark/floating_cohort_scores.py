"""Verify archived floating exports and pool their scores on the active cohort."""

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

from benchmark.expanded_baseline import WORKTREES, digest, read, save
from benchmark.floating_extra_budget import LOCATION, paths
from benchmark.mask_blocks import decoded_pack
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.typed_structures import geometry_score, overlap, score
from training.floating_cohort import COHORT, MANIFEST, frame_index, select_active_cases


PANELS = ("original", "fresh", "supplemental", "sky", "holdout", "holdout2", "holdout3", "holdout4", "holdout5")
ARMS = ("floating-default", "baseline", "source-one", "large-contraction")
ISOLATION_REVISION = "e97dd44f84051c644fed5769d0697aaa7a08581b"
METRICS = (("geometry", "strict"), ("geometry", "facade"), ("scores", "strict"), ("scores", "facade"))
COUNTS = ("correct", "missed", "extras")


def reference(case, frame):
    directory = Path(case["frame"])
    for name, expected in frame["reference_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Changed scoring reference: {directory / name}")
    labels = read(directory / "labels.json")
    with np.load(directory / "labels.npz", allow_pickle=False) as regions:
        optional = horizontal_slivers(labels, regions["face_id"])
    truth = {tuple(block["xyz_world"]): block["block_id"] for block in labels["blocks"]}
    return truth, optional


def grade_export(path, expected_source, frame, truth, optional, old_score):
    receipt = read(path / "score.json")
    pack = path / "reconstruction.zip"
    if (receipt["source_sha256"] != expected_source or receipt["reference_sha256"] != frame["reference_sha256"]
            or receipt["pack_sha256"] != digest(pack)):
        raise ValueError(f"Export receipt checksum mismatch: {path}")
    positions = read(path / "placements.json")
    decoded = decoded_pack(pack)
    if decoded != {tuple(p): material for p, material in positions["local"]}:
        raise ValueError(f"Decoded pack disagrees with archived placements: {path}")
    chosen = receipt["score"]["fixed"] or receipt["score"]["relaxed_yaw"]
    if chosen != old_score:
        raise ValueError(f"Archive score differs from receipt: {path}")
    yaw = chosen["yaw_quarters"]
    tx, ty, tz = chosen["translation"]
    aligned = {}
    for (x, y, z), material in decoded.items():
        x, z = ((x, z), (-z, x), (-x, -z), (z, -x))[yaw]
        aligned[x + tx, y + ty, z + tz] = material
    if aligned != {tuple(p): material for p, material in positions["aligned"]}:
        raise ValueError(f"Whole-build registration disagrees with archived placements: {path}")
    measured = dict(geometry=geometry_score(aligned, truth, optional), scores=score(aligned, truth, optional, set()))
    for field, kind in METRICS:
        if measured[field][kind] != chosen[field][kind]:
            raise ValueError(f"Decoded pack changes {field}/{kind} count: {path}")
    return {f"{field}_{kind}": measured[field][kind] for field, kind in METRICS}


def pool(rows):
    return dict(
        image_conditions=len(rows),
        unique_release_captures=len({(row["dataset"], row["id"]) for row in rows}),
        arms={
            arm: {
                metric: overlap(*(sum(row["arms"][arm][metric][count] for row in rows) for count in COUNTS))
                for metric in (f"{field}_{kind}" for field, kind in METRICS)
            }
            for arm in ARMS
        },
    )


def audit(isolation):
    index = frame_index()
    archived = isolation / "benchmark/reviews/floating-extra-isolation-01"
    rows = []
    fingerprints = {}
    for panel in PANELS:
        (source, _, _, cases), baseline, proposal = paths(panel)
        active_ids = {case["id"] for case in select_active_cases(cases)}
        history = archived / f"{panel}.json.gz"
        archive_rows = {row["id"]: row for row in read(history)}
        if archive_rows.keys() != {case["id"] for case in cases}:
            raise ValueError(f"Archived and replayed cases differ: {panel}")
        fingerprints[panel] = digest(history)
        exports = {
            "floating-default": source / ("relative-face-assembly" if panel == "sky" else "assembly"),
            "baseline": baseline,
            "source-one": proposal,
            "large-contraction": isolation / "runs/extra-isolation" / panel / "large-contraction/assembly",
        }
        for case in cases:
            name, dataset = case["id"], case["dataset"]
            frame = index[dataset, name]
            historical = archive_rows[name]
            images = case["variants"]["sky"]
            source_hash = digest(Path(images["image"]))
            if source_hash != images["source_sha256"] or source_hash != historical["source_sha256"]:
                raise ValueError(f"Saved source-image checksum mismatch: {panel}/{name}")
            truth, optional = reference(case, frame)
            arms = {
                arm: grade_export(exports[arm] / LOCATION / name, source_hash, frame, truth, optional,
                                  historical["score"][arm])
                for arm in ARMS
            }
            rows.append(dict(panel=panel, id=name, dataset=dataset, split=frame["split"], layout=frame["layout"],
                             active=name in active_ids, source_sha256=source_hash, arms=arms))
        print(f"{panel}: verified {len(cases)} conditions, {len(active_ids)} active", flush=True)
    active = [row for row in rows if row["active"]]
    manifest = read(MANIFEST)
    return dict(
        cohort=COHORT, cohort_manifest_lf_sha256=sha256(MANIFEST.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
        historical_score_revision=ISOLATION_REVISION, historical_score_sha256=fingerprints,
        method="Decoded exports graded using archived whole-build yaw and XYZ translation and the frozen visible labels.",
        all_archived=pool(rows), active=pool(active),
        active_by_dataset={
            dataset: dict(
                pool([row for row in active if row["dataset"] == dataset]),
                available_active_captures=sum(split["active_frames"] for split in release["splits"].values()),
            )
            for dataset, release in manifest["datasets"].items()
        },
        active_by_panel={panel: pool([row for row in active if row["panel"] == panel]) for panel in PANELS},
        excluded_archive_conditions=dict(Counter(row["layout"] for row in rows if not row["active"])),
        rows=rows,
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--isolation", type=Path, default=WORKTREES / "floating-extra-isolation",
                     help="Registered checkout for the historical floating extra-isolation results")
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    save(args.out, audit(args.isolation))


if __name__ == "__main__":
    main()
