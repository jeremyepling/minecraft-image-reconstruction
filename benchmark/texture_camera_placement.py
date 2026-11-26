"""Run frozen image-only surface/count inputs and a selected camera through datapack decoding."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path

from benchmark.texture_camera_eval import BASE, DATA, ROOT, checked, digest, frames, mark_image_only, read, save
import numpy as np
from PIL import Image
from benchmark.camera_metrics import registration, truth_camera
from benchmark.mask_blocks import _score, _write_read, registered
from engine import visible_placement
from engine.photo_edges import refine_surfaces
from engine.surface_model import _count_regions, DEFAULTS

FIXED = ROOT / "benchmark/reviews/edge-detail-stable-01"
HELD = ROOT / "benchmark/reviews/detail-transfer-01"
SPLITS = ("validation", "test", "regression")


def mask_sources(split, samples):
    archive = FIXED if split == "regression" else HELD
    artifacts = read(archive / "manifest.json")["artifacts"]
    if split == "regression":
        index = {r["key"]: r for r in read(checked(archive / "cases.json", artifacts["cases.json"]))}
    for f in samples:
        if split == "regression":
            row = index[f["legacy_key"]]
            assert row["sha256"] == f["files"]["rgb.png"]
            member = row["predictions"]
        else:
            member = "ordinary/" + f["id"] + ".json.gz"
        yield archive / member, artifacts[member]


def evaluate(job):
    f, folder, mask_info, camera, frame_out = job
    frame_out.mkdir(parents=True, exist_ok=True)
    rgb_path = checked(folder / "rgb.png", f["files"]["rgb.png"])
    with Image.open(rgb_path) as src:
        rgb = Image.fromarray(np.asarray(src.convert("RGB")))
    mask_path, mask_hash = mask_info
    if camera["P"] is not None:
        source = json.loads(gzip.decompress(checked(mask_path, mask_hash).read_bytes()))
        if "legacy_key" in f:
            parts = source["edge_detail:native4-stable-counts"]
        else:
            assert source["manifest"]["image_sha256"] == f["files"]["rgb.png"]
            parts, _ = refine_surfaces(rgb, source["planes"])
        # The placement entry point consumes only counts + quad. Atlas material ranking and
        # its optional sub-block phase do not change these fields or the exported block type.
        counts = _count_regions(rgb, parts, "26.3", read(DEFAULTS)["block_count"], None, frame_out)
    else:
        parts, counts = [], None
    cells, status = visible_placement.place_with_camera(rgb, "26.3", parts, counts, camera)
    pack = frame_out / "reconstruction.zip"
    decoded = _write_read(pack, cells)
    # Open scoring-only camera and labels after creating and verifying the output pack.
    pose = read(checked(folder / "camera.json", f["files"]["camera.json"]))
    labels = read(checked(folder / "labels.json", f["files"]["labels.json"]))["blocks"]
    visible = {tuple(b["xyz_world"]): b["block_id"] for b in labels}
    world = registered(decoded, registration(np.asarray(camera["P"]), pose)) if decoded else set()
    score = _score(world, set(visible))
    score["joint_correct"] = sum(visible.get(x) == "minecraft:smooth_stone" for x in world)
    save(frame_out / "registered.json", dict(registered=sorted(world), correct=sorted(world & visible.keys())))
    return dict(id=f["id"], layout=f["layout"], family=f.get("family", f["layout"]), split=f["split"],
                rgb_sha256=f["files"]["rgb.png"], mask_sha256=mask_hash, pack_sha256=digest(pack),
                status=status, placement=score)


def summary(rows):
    totals = sum((Counter(r["placement"]) for r in rows), Counter())
    for field in ("correct_visible", "extra", "missed", "visible_truth", "serialized", "joint_correct"):
        totals.setdefault(field, 0)
    return dict(frames=len(rows), supported=sum(r["status"]["camera_status"] == "supported" for r in rows),
                **totals, precision=totals["correct_visible"]/totals["serialized"] if totals["serialized"] else None,
                recall=totals["correct_visible"]/totals["visible_truth"])


def recorded_camera(f, folder):
    """Scoring control: express the recorded world camera in a ground-relative, local unit grid."""
    pose = read(checked(folder / "camera.json", f["files"]["camera.json"]))
    transform = np.eye(4)
    transform[:3, 3] = np.floor(pose["eye_world"])
    transform[1, 3] = f["ground_y"] + 1
    P = truth_camera(pose) @ transform
    assert np.array_equal(registration(P, pose), transform)
    return dict(P=P.tolist(), scale_status="supported", reason="Recorded-camera placement diagnostic")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", type=Path, default=BASE,
                   help="Directory containing split/cases.json, or cases.json for one split.")
    p.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--recorded-camera", action="store_true",
                   help="Scoring-only diagnostic. Replace the estimated camera with the recorded pose.")
    p.add_argument("--data", type=Path, default=DATA)
    args = p.parse_args()
    for split in SPLITS if args.split == "all" else [args.split]:
        dest = args.out / split if args.split == "all" else args.out
        dest.mkdir(exist_ok=True, parents=True)
        camera_dir = args.camera if (args.camera / "cases.json").exists() else args.camera / split
        model = read(camera_dir / "manifest.json")
        pred = {r["id"]: r for r in read(checked(camera_dir / "cases.json", model["cases_sha256"]))}
        rows = frames(args.data, split)
        assert set(pred) == {r["id"] for r in rows}
        jobs = []
        for f, mask in zip(rows, mask_sources(split, rows), strict=True):
            r = pred[f["id"]]
            assert r["rgb_sha256"] == f["files"]["rgb.png"]
            folder = args.data / "data" / split / f["directory"]
            camera = recorded_camera(f, folder) if args.recorded_camera else r["prediction"]
            jobs.append((f, folder, mask, camera, dest / f["id"]))
        with ProcessPoolExecutor(max_workers=args.workers, initializer=mark_image_only) as pool:
            results = list(pool.map(evaluate, jobs))
        save(dest / "cases.json", results)
        manifest = dict(camera_cases_sha256=model["cases_sha256"], camera_manifest_sha256=digest(camera_dir / "manifest.json"),
                        recorded_camera_diagnostic=args.recorded_camera,
                        scorer_sha256=digest(Path(__file__)), cases_sha256=digest(dest / "cases.json"),
                        summary=summary(results))
        save(dest / "manifest.json", manifest)
        print(split, json.dumps(manifest["summary"]), flush=True)


if __name__ == "__main__":
    main()
