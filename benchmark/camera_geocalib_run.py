"""Run GeoCalib after both the ordinary and repeat-only cameras abstain."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

os.environ.update(MCIR_IMAGE_ONLY="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

from PIL import Image
from benchmark.camera_geocalib import solve
from benchmark.camera_geocalib_infer import OUT, WEIGHT_SHA
from benchmark.camera_horizon_run import BASE, CONTROL
from benchmark.expanded_baseline import ROOT, COHORTS, SOURCE, digest, frames, read, save

REPEAT = ROOT.parent / "camera-horizon-floor/runs/camera-horizon-floor-01"


def run(job):
    entry, stage, control = job
    root = OUT/"matched-control" if control else OUT
    key, dataset = entry["id"], entry["dataset"]
    rgb = Path(entry["frame"]) / "rgb.png"
    camera_file = root / "metric" / f"{key}.json"
    if stage == "predict":
        if digest(rgb) != entry["rgb_sha256"]:
            raise ValueError(key)
        proposal = None
        if not control:
            proposal = read(OUT / "proposals" / f"{key}.json")
            if proposal["rgb_sha256"] != entry["rgb_sha256"] or proposal["weights_sha256"] != WEIGHT_SHA:
                raise ValueError(key)
        with Image.open(rgb) as image:
            camera = solve(image, proposal)
        save(camera_file, dict(camera=camera, rgb_sha256=entry["rgb_sha256"]))
    else:
        camera = read(camera_file)["camera"]
        if camera["P"] is not None:
            from benchmark.camera_horizon_replay import replay
            case = replay(entry, camera, BASE/dataset/key, root/"builds"/dataset/key)
            save(root/"scores"/f"{key}.json", case)
    print(key, camera["P"] is not None, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["predict", "build"])
    parser.add_argument("split", choices=["train", "heldout", "all"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--matched-control", action="store_true")
    args = parser.parse_args()
    failures = {c["id"] for c in read(CONTROL/"benchmark/reviews/expanded-baseline-01/cases.json") if not c["camera_usable"]}
    jobs = []
    for entry in frames(SOURCE, COHORTS):
        if entry["id"] not in failures:
            continue
        prior = read(REPEAT/entry["id"]/"repeat.json")
        if prior["rgb_sha256"] != entry["rgb_sha256"]:
            raise ValueError(entry["id"])
        if prior["camera"]["P"] is not None:
            continue
        if args.split == "all" or (entry["split"]=="train") == (args.split=="train"):
            jobs.append((entry, args.stage, args.matched_control))
    with ProcessPoolExecutor(args.workers) as workers:
        list(workers.map(run, jobs))


if __name__ == "__main__":
    main()
