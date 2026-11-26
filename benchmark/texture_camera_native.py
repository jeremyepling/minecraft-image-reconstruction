"""Independent screen of all 70 archived test frames, including earlier abstentions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from benchmark import texture_camera_eval as runner
import numpy as np
from PIL import Image
from benchmark.camera_labels import face_labels
from benchmark.camera_metrics import score


def evaluate(row):
    root, key = Path(row["root"]), row["id"]
    captures = root / "captures"
    rgb = runner.checked(captures / (key + ".png"), row["sha256"])
    with Image.open(rgb) as src:
        photo = Image.fromarray(np.asarray(src.convert("RGB")))
    base = runner.production(photo, "26.3")
    assert (base["P"] is not None) == row["supported"]
    new = runner.challenger(photo, "26.3") if base["P"] is None else base
    assert (new["P"] is not None) == (new["scale_status"] == "supported")
    pose_path, states_path = captures / (key + ".json"), captures / (key + "-blocks.json")
    states = {tuple(b[k] for k in "xyz"): b for b in runner.read(states_path)}
    faces = face_labels(runner.read(pose_path), states, dict.fromkeys(states, "structure"))
    def case(pred):
        return dict(id=key, layout=key.rsplit("-", 1)[0], metric=pred["P"] is not None, prediction=pred,
                    metrics=score({"planes": [], **pred}, runner.read(pose_path), faces))
    old = case(base)
    return dict(id=key, fixture=root.name, rgb_sha256=row["sha256"], states_sha256=runner.digest(states_path),
                pose_sha256=runner.digest(pose_path), baseline=old, candidate=old if new is base else case(new))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=runner.STABLE.parent / "workspace/builds/supported-placement-holdout/other-cameras.json")
    p.add_argument("--fixtures", type=Path,
                   help="Location of the native suite directories when replaying a portable fixture index.")
    p.add_argument("--candidate", required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    rows = runner.read(args.index)
    if args.fixtures:
        for row in rows:
            row["root"] = str(args.fixtures / row["fixture_directory"])
    assert len({r["sha256"] for r in rows}) == len(rows) == 70
    stock = {r["rgb_sha256"] for r in runner.read(runner.RELEASE)["frames"]}
    assert not any(r["sha256"] in stock for r in rows)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=runner.initialize,
                             initargs=(args.candidate_root.resolve(), args.candidate)) as pool:
        results = list(pool.map(evaluate, rows))
    args.out.mkdir(parents=True, exist_ok=True)
    runner.save(args.out / "cases.json", json.loads(json.dumps(results, default=runner.plain, allow_nan=False)))
    source = args.candidate_root.joinpath(*args.candidate.split(".")).with_suffix(".py")
    report = dict(index_sha256=runner.digest(args.index), candidate_module=args.candidate,
                  candidate_sha256=runner.digest(source), scorer_sha256=runner.digest(Path(__file__)),
                  cases_sha256=runner.digest(args.out / "cases.json"))
    for method in ["baseline", "candidate"]:
        report[method] = runner.aggregate([r[method] for r in results])
    runner.save(args.out / "manifest.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
