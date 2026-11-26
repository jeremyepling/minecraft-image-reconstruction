"""Decode and compare wall placements on the native canopy views where the floor camera abstained."""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark.texture_camera_eval import ROOT, checked, digest, mark_image_only, read, save
import numpy as np
from PIL import Image
from benchmark.camera_metrics import registration, truth_camera
from benchmark.mask_blocks import _score, _write_read, decoded_pack, registered
from calibration.world import GROUND
from engine import visible_placement
from training.visible_labels import visible_labels


def inputs(archive, fixtures):
    with ZipFile(archive) as z:
        def rows(member):
            return json.loads(z.read(member))
        index = {r["sha256"]: r for r in rows("native-inputs/index.json")}
        def key_cases(method):
            return {r["rgb_sha256"]: r for r in rows("native70/" + method + "/cases.json")}
        broad, guard, floor = (key_cases(method) for method in ("wall-broad", "wall-guard", "fb9d5b4"))
    assert broad.keys() == guard.keys() == floor.keys() == index.keys()
    out = []
    for sha, row in index.items():
        chosen = broad[sha]["candidate"]
        if floor[sha]["candidate"]["metric"]:
            continue
        if not chosen["metric"]:
            assert not guard[sha]["candidate"]["metric"]
            continue
        path = fixtures / row["fixture_directory"] / "captures"
        out.append(dict(id=row["id"], fixture=row["fixture_directory"], folder=path, rgb_sha256=sha,
                        pose_sha256=broad[sha]["pose_sha256"], states_sha256=broad[sha]["states_sha256"],
                        broad=chosen["prediction"], guard=guard[sha]["candidate"]["prediction"]))
    return out


def evaluate(job):
    f, source, output = job
    key = f["id"]
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(checked(f["folder"] / (key + ".png"), f["rgb_sha256"])) as im:
        photo = Image.fromarray(np.asarray(im.convert("RGB")))
    manifest = read(source / "build-manifest.json")
    assert manifest["image_sha256"] == f["rgb_sha256"] and manifest["through"] == "counts"
    parts, counts = read(source / "surfaces.json")["planes"], read(source / "counts.json")
    def export(name, camera):
        cells, _ = visible_placement.place_with_camera(photo, "26.3", parts, counts, camera)
        path = output / (name + ".zip")
        _write_read(path, cells)
        return decoded_pack(path)
    cameras = {method: f[method] for method in ("broad", "guard")}
    decoded = {method: export(method, camera) for method, camera in cameras.items()}
    # Recorded pose only enters after both ordinary predictions were serialized.
    pose = read(checked(f["folder"] / (key + ".json"), f["pose_sha256"]))
    T = np.eye(4)
    T[:3, 3] = np.floor(pose["eye_world"])
    T[1, 3] = GROUND + 1
    P = truth_camera(pose) @ T
    assert np.array_equal(registration(P, pose), T)
    cameras["recorded"] = dict(P=P.tolist(), scale_status="supported", reason="Recorded-camera diagnostic")
    decoded["recorded"] = export("recorded", cameras["recorded"])
    states = {tuple(b[k] for k in "xyz"): b for b in read(checked(f["folder"] / (key + "-blocks.json"), f["states_sha256"]))}
    labels, _, _ = visible_labels(pose, states, (0, 0, 0))
    target = {tuple(b["xyz_world"]): b["block_id"] for b in labels["blocks"]}
    hidden = {p for p, b in states.items() if b["block_id"].removeprefix("minecraft:") != "air"} - target.keys()
    results = {}
    correct_cells = {}
    for method, cells in decoded.items():
        camera = cameras[method]
        transform = registration(np.asarray(camera["P"]), pose) if cells else None
        world = {}
        for state in set(cells.values()):
            selected = {p for p, s in cells.items() if s == state}
            world.update(dict.fromkeys(registered(selected, transform), state))
        assert len(world) == len(cells)
        matched = set(world) & target.keys()
        correct_cells[method] = matched
        result = dict(**_score(set(world), set(target)),
                      joint_correct=sum(world[p] == target[p] for p in matched),
                      extra_at_hidden_structure=len(world.keys() & hidden),
                      exported_ids=Counter(world.values()), supported=camera["P"] is not None)
        results[method] = result
        save(output / (method + "-registered.json"),
             dict(registered=[[*p, state] for p, state in sorted(world.items())], correct=sorted(matched)))
    for method in ("broad", "guard"):
        results[method]["correct_only_with_recorded"] = len(correct_cells["recorded"] - correct_cells[method])
        results[method]["correct_only_with_estimated"] = len(correct_cells[method] - correct_cells["recorded"])
    return dict(id=key, fixture=f["fixture"], rgb_sha256=f["rgb_sha256"], pose_sha256=f["pose_sha256"],
                states_sha256=f["states_sha256"], surface_sha256=digest(source / "surfaces.json"),
                count_sha256=digest(source / "counts.json"), surface_manifest_sha256=digest(source / "build-manifest.json"),
                camera=cameras, placements=results, smooth_stone_truth=sum(v == "minecraft:smooth_stone" for v in target.values()))


def aggregate(rows, method):
    totals = Counter()
    ids = Counter()
    for row in rows:
        for key, value in row["placements"][method].items():
            if key == "exported_ids":
                ids.update(value)
            else:
                totals[key] += int(value)
    return dict(frames=len(rows), **totals, exported_ids=dict(ids),
                precision=totals["correct_visible"]/totals["serialized"] if totals["serialized"] else None,
                recall=totals["correct_visible"]/totals["visible_truth"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures", type=Path, required=True, help="Native fixture root with saved RGBs, poses and readbacks")
    p.add_argument("--surfaces", type=Path, required=True, help="Directory of through-counts build outputs, one per frame ID")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera-bundle", type=Path, default=ROOT / "benchmark/reviews/texture-camera-01/scoring.zip")
    p.add_argument("--bundle", type=Path, help="Write a self-contained score bundle with frozen surface/count inputs")
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    selected = inputs(args.camera_bundle, args.fixtures)
    assert len(selected) == 15
    jobs = [(f, args.surfaces / f["id"], args.out / "frames" / f["id"]) for f in selected]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=mark_image_only) as pool:
        rows = list(pool.map(evaluate, jobs))
    relevant = [r for r in rows if r["placements"]["guard"]["supported"]]
    assert len(relevant) == 3
    save(args.out / "cases.json", rows)
    summaries = {group: {m: aggregate(rs, m) for m in ("broad", "guard", "recorded")}
                 for group, rs in (("floor_abstained_wall_broad_supported", rows), ("guard_supported", relevant))}
    # Compose the native70 totals with the saved exact scores for the 12 preserved
    # seam cameras. All remaining images have empty outputs in both comparisons.
    stock_path = ROOT / "benchmark/reviews/combined-default-01/scorecard.json"
    stock = read(stock_path)
    visible = stock["camera_screen"]["native70"]["visible"]
    existing = stock["groups"]["supported12"]["placements"]
    all70 = {}
    for method in ("floor", "broad", "guard"):
        gain = {} if method == "floor" else summaries["floor_abstained_wall_broad_supported"][method]
        total = {k: existing[k] + gain.get(k, 0) for k in ("serialized", "correct_visible", "extra", "joint_correct")}
        all70[method] = dict(**total, visible_truth=visible, missed=visible-total["correct_visible"],
                             precision=total["correct_visible"]/total["serialized"], recall=total["correct_visible"]/visible)
    report = dict(camera_bundle_sha256=digest(args.camera_bundle), cases_sha256=digest(args.out / "cases.json"),
                  runner_sha256=digest(Path(__file__)), summary=summaries,
                  prior_native_scorecard_sha256=digest(stock_path), native70_composed=all70)
    save(args.out / "manifest.json", report)
    if args.bundle:
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(args.bundle, "w", ZIP_DEFLATED) as z:
            files = [(args.out / name, name) for name in ("manifest.json", "cases.json")]
            for f, source, out in jobs:
                files.extend((path, "frames/" + f["id"] + "/" + path.name) for path in sorted(out.iterdir()))
                files.extend((source / name, "surfaces/" + f["id"] + "/" + name)
                             for name in ("surfaces.json", "counts.json", "build-manifest.json"))
            for path, name in files:
                z.write(path, name)
            z.writestr("SHA256SUMS.txt", "".join(digest(path) + "  " + name + "\n" for path, name in files))
            z.writestr("REPLAY.md", """# Native wall camera placement

Use the native input RGBs, recorded poses and block readbacks described in the adjacent scoring.zip.
Each selected broad-wall view had an abstention from the promoted floor camera.
Every surface input here came from one ordinary Invoke-WallRecovery.ps1 -Through counts run on its RGB.
To regenerate a surface run, send its original PNG through that wrapper and set -Out to surfaces/FRAME_ID.

Extract this bundle and run the following command from the repository using Python, Pillow and NumPy.
Set FIXTURES to the native fixture root and SURFACES to this bundle's extracted surfaces directory.

    python -m benchmark.texture_camera_native_placement --fixtures FIXTURES --surfaces SURFACES --out NEW

The same image-derived masks and counts are shared across both wall methods and the recorded-camera diagnostic.
The scorer registers decoded local cube centers and compares exact world coordinates and exported block IDs
against native pixel-visible building truth. Floor, occluded and off-image cells cannot earn a match.
The guard-supported summary covers the exact three images on which the guard returned a camera.
""")
        with ZipFile(args.bundle) as z:
            assert z.testzip() is None
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
