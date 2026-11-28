"""Stage the shared review site with its published run index."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


APP_REPOSITORY = "https://github.com/jeremyepling/minecraft-reconstruction-review"
APP_DIRECTORY = "minecraft-reconstruction-review"
SCORED_FILES = ("index.html", "app.js", "styles.css", "vercel.json")
SINGLE_TEMPLATE = Path("single/index.html")
ROOT = Path(__file__).resolve().parents[1]


def app_checkout(path=None):
    configured = path or os.environ.get("MCIR_FLOATING_REVIEW_APP")
    candidates = ([Path(configured)] if configured else
                  [root / APP_DIRECTORY for root in (ROOT.parent, ROOT.parent.parent)])
    for candidate in candidates:
        required = [SINGLE_TEMPLATE, *(Path("review-site") / name for name in SCORED_FILES)]
        if all((candidate / name).is_file() for name in required):
            return candidate.resolve()
    raise FileNotFoundError(f"Clone {APP_REPOSITORY} beside the reconstruction repo or set --app-root")


def stage_scored_app(app, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in SCORED_FILES:
        shutil.copyfile(app / "review-site" / name, destination / name)

RUN_MANIFESTS = Path("benchmark/reviews/mobile-review-runs")


def stage_scored_review(app, destination, *, single_title=None):
    stage_scored_app(app, destination)
    tracked = subprocess.check_output(["git", "ls-files", RUN_MANIFESTS], cwd=ROOT, text=True)
    experiments = []
    for name in sorted(tracked.splitlines()):
        asset = Path(name).relative_to(RUN_MANIFESTS)
        if not re.fullmatch(r"[a-z0-9-]+/manifest\.json", asset.as_posix()):
            raise ValueError(f"Unexpected saved run file: {asset}")
        target = destination / "runs" / asset
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        manifest = json.loads(target.read_text(encoding="utf-8"))
        if manifest["id"] != asset.parent.name:
            raise ValueError(f"Run ID differs from saved manifest path: {asset}")
        experiments.append(dict(id=manifest["id"], title=manifest["title"]))
    single = (dict(title=single_title, review="/single/review", scores="/single/scores")
              if single_title is not None else None)
    data = dict(single=single, experiments=experiments)
    (destination / "review-home.json").write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Stage the scored review app with the existing published run manifests.")
    parser.add_argument("--app-root", type=Path, help="Review app checkout")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new or empty staging directory")
    stage_scored_review(app_checkout(args.app_root), args.out)


if __name__ == "__main__":
    main()
