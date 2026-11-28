"""Compare image-only camera fallbacks against the preserved production seam camera."""
import os
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MCIR_IMAGE_ONLY="1")
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from PIL import Image
from benchmark.camera_metrics import score, summarize
from benchmark.camera_round import digest, read, save
from engine.camera_seam import predict as production
from runtime.mode import mark_image_only

ROOT = Path(__file__).resolve().parents[1]


def stable_checkout(root):
    """Resolve the shared checkout for historical run paths without launching Git."""
    link = root / '.git'
    if link.is_file():
        git_dir = root / link.read_text(encoding='utf-8').strip().removeprefix('gitdir: ')
        common = git_dir / 'commondir'
        if common.is_file():
            return (git_dir / common.read_text(encoding='utf-8').strip()).resolve().parent
    return root


STABLE = stable_checkout(ROOT)
RELEASE = ROOT / "training/visible-builds-v1.json"
DATA = STABLE / "runs/visible-builds-v1"
TRAINING_RELEASE = ROOT / "training/visible-builds-v2.json"
TRAINING_DATA = STABLE / "runs/visible-builds-v2"
BASE = STABLE / "runs/texture-camera-rounds/baseline"
SPLITS = ("train", "validation", "test", "regression")


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"SHA256 mismatch for {path}")
    return path


def frames(data, split, manifest=RELEASE):
    release = read(manifest)
    ids = {r["id"]: r for r in release["frames"] if r["split"] == split}
    shard = release["shards"][split]
    with ZipFile(checked(data / shard["file"], shard["sha256"])) as z:
        rows = [json.loads(line) for line in z.read("frames.jsonl").splitlines()]
    assert set(ids) == {r["id"] for r in rows} and len(rows) == len(ids)
    for r in rows:
        assert ids[r["id"]]["rgb_sha256"] == r["files"]["rgb.png"] and r["split"] == split
    return rows


def candidate_fn(candidate_root, candidate):
    if not candidate:
        return None
    sys.path.insert(0, str(candidate_root))
    for name in ("engine", "benchmark"):
        package = importlib.import_module(name)
        p = str(candidate_root / name)
        if Path(p).is_dir() and p not in package.__path__:
            package.__path__ = [p, *package.__path__]
    return importlib.import_module(candidate).predict


def initialize(candidate_root, candidate):
    global challenger
    mark_image_only()
    challenger = candidate_fn(candidate_root, candidate)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def evaluate(job):
    frame, folder, original = job
    sha = frame["files"]
    with Image.open(checked(folder / "rgb.png", sha["rgb.png"])) as opened:
        photo = Image.fromarray(np.asarray(opened.convert("RGB")))
    if original:
        assert original["rgb_sha256"] == sha["rgb.png"] and original["id"] == frame["id"]
        base = original["prediction"]
    else:
        base = production(photo, "26.3")
    used = bool(challenger and base["P"] is None)
    start = perf_counter()
    pred = challenger(photo, "26.3") if used else base
    elapsed = perf_counter() - start
    pred = json.loads(json.dumps(pred, default=plain, allow_nan=False))
    supported = pred["scale_status"] == "supported"
    assert supported == (pred["P"] is not None)
    if supported:
        matrix = np.asarray(pred["P"])
        assert matrix.shape == (3, 4) and np.isfinite(matrix).all()
    if original and not used:
        metrics = original["metrics"]
    else:
        pose = read(checked(folder / "camera.json", sha["camera.json"]))
        faces = read(checked(folder / "reference-faces.json", sha["reference-faces.json"]))["faces"]
        metrics = score(dict(planes=[], **{k: v for k, v in pred.items() if k != "planes"}), pose, faces)
    return dict(id=frame["id"], split=frame["split"], layout=frame["layout"], family=frame.get("family", frame["layout"]),
                angle=frame["angle"], palette=frame["palette"], rgb_sha256=sha["rgb.png"], metric=supported,
                candidate_invoked=used, candidate_seconds=elapsed if used else None, prediction=pred, metrics=metrics)


def aggregate(rows):
    rows = [dict(r, **r["metrics"]) for r in rows]
    s = summarize(rows)
    projected = [r for r in rows if r["projected"]]
    s.update(eligible_hit_rate=s["within_2px"]/max(1, s["vertices"]),
             projected_hit_rate=s["within_2px"]/s["projected"] if s["projected"] else None,
             pooled_mean_projected_px=(sum(r["projected"]*r["mean_px"] for r in projected)/s["projected"]
                                       if s["projected"] else None),
             median_frame_p90_px=float(np.median([r["p90_px"] for r in projected])) if projected else None,
             full_projection_p90_le_2px=sum(r["projected"] == r["vertices"] and r["p90_px"] <= 2 for r in projected))
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate")
    p.add_argument("--candidate-root", type=Path, default=ROOT)
    p.add_argument("--split", choices=(*SPLITS,"all"), required=True)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--baseline", type=Path, default=BASE)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    candidate_root = args.candidate_root.resolve()
    source = candidate_root.joinpath(*args.candidate.split(".")).with_suffix(".py") if args.candidate else None
    for split in SPLITS if args.split == "all" else [args.split]:
        out = args.out / split if args.split == "all" else args.out
        out.mkdir(parents=True, exist_ok=True)
        selected = frames(args.data, split)[:args.limit]
        originals = {}
        cache = args.baseline / split / "cases.json"
        if args.candidate and cache.exists():
            manifest = read(cache.parent / "manifest.json")
            checked(cache, manifest["cases_sha256"])
            assert manifest["production_sha256"] == digest(ROOT / "engine/camera_seam.py")
            assert not manifest["candidate"] and manifest["release_sha256"] == digest(RELEASE)
            originals = {r["id"]: r for r in read(cache)}
        jobs = [(f, args.data / "data" / split / f["directory"], originals.get(f["id"])) for f in selected]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 initargs=(candidate_root, args.candidate)) as pool:
            rows = list(pool.map(evaluate, jobs))
        save(out / "cases.json", rows)
        result = dict(split=split, release_sha256=digest(RELEASE), shard_sha256=read(RELEASE)["shards"][split]["sha256"],
                      production_sha256=digest(ROOT / "engine/camera_seam.py"), candidate=args.candidate,
                      candidate_sha256=digest(source) if source else None, scorer_sha256=digest(Path(__file__)),
                      git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=candidate_root,text=True).strip(),
                      cases_sha256=digest(out / "cases.json"), summary=aggregate(rows))
        save(out / "manifest.json", result)
        print(split, json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
