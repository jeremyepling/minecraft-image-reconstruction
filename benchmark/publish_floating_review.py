"""Prepare a verified mobile review and optionally publish its assets and both production routes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

from benchmark.build_floating_mobile_review import manifest, surface_examples
from benchmark.floating_stage_report import read
from benchmark.floating_review_app import RUN_MANIFESTS, app_checkout, stage_scored_review

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "https://github.com/jeremyepling/minecraft-image-reconstruction"
RAW = "https://raw.githubusercontent.com/jeremyepling/minecraft-image-reconstruction"
LIVE = "https://floating-review.vercel.app"
PROJECT = "floating-review"
SCOPE = "jeremy-eplings-projects-2b30cb9d"


def git(*args):
    return subprocess.check_output(["git", *map(str, args)], cwd=ROOT, text=True).strip()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def public_files(data, assets):
    files = []
    for case in data["cases"]:
        for url in [*(m["url"] for m in case["modes"]), case.get("surfaceEvidence")]:
            if url:
                files.append(assets / case["id"] / url.rsplit("/", 1)[-1])
    return files


def point_assets(data, prefix):
    for case in data["cases"]:
        def pin(url):
            return f"{prefix}/{case['id']}/{url.rsplit('/', 1)[-1]}"
        for mode in case["modes"]:
            mode["url"] = pin(mode["url"])
        if case.get("surfaceEvidence"):
            case["surfaceEvidence"] = pin(case["surfaceEvidence"])


def commit_files(files, message):
    if not files:
        return
    # Only the explicit generated public files can enter this commit.
    git("add", "-f", "--", *files)
    if git("diff", "--cached", "--name-only", "--", *files):
        git("commit", "--only", "-m", message, "--", *files)


def published_bytes(url, expected):
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers={"Cache-Control": "no-cache"}), timeout=20) as response:
                body = response.read()
            if expected(body):
                return
            raise ValueError(f"Unexpected contents at {url}")
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def deploy(data, source_assets, site_files, cli, app_root=None):
    app = app_checkout(app_root)
    branch = git("branch", "--show-current")
    if branch != "main" and not branch.startswith(("feature/floating-", "experiment/floating-")):
        raise ValueError(f"Publish from a named Floating integration or feature branch: {branch}")
    if git("remote", "get-url", "--push", "origin").removesuffix(".git") != REMOTE:
        raise ValueError("Expected the verified Jeremy-owned HTTPS repository.")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Commit or isolate existing work before publishing generated files.")
    if not cli:
        raise ValueError("Provide --vercel or install vercel on PATH.")
    subprocess.run([cli, "whoami", "--no-color"], check=True)
    slug = data["id"]
    dest = Path("benchmark/reviews/mobile-review-assets") / slug
    target = RUN_MANIFESTS / slug / "manifest.json"
    if (ROOT / target).exists() and read(ROOT / target)["comparisonSha256"] != data["comparisonSha256"]:
        raise ValueError("This run ID already identifies a different comparison.")
    files = []
    for source in public_files(data, source_assets):
        relative = dest / source.relative_to(source_assets)
        (ROOT / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, ROOT / relative)
        files.append(relative)
    commit_files(files, f"Save mobile evidence for {slug}")
    pin = git("rev-parse", "HEAD")
    if files:
        pin = git("log", "-1", "--format=%H", "--", *files)
    point_assets(data, f"{RAW}/{pin}/{dest.as_posix()}")
    save(ROOT / target, data)
    commit_files([target], f"Publish Floating review decisions for {slug}")
    git("push", "origin", f"HEAD:refs/heads/{branch}")
    if git("ls-remote", "origin", f"refs/heads/{branch}").split()[0] != git("rev-parse", "HEAD"):
        raise ValueError("Remote branch does not match the prepared review.")
    def check_asset(path):
        # Git can normalize Windows line endings. Verify the bytes from the pinned tree.
        blob = subprocess.check_output(["git","show",f"{pin}:{path.as_posix()}"],cwd=ROOT)
        expected = hashlib.sha256(blob).digest()
        published_bytes(f"{RAW}/{pin}/{path.as_posix()}", lambda body: hashlib.sha256(body).digest() == expected)
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(check_asset, files))

    stage_scored_review(app, site_files)
    subprocess.run([cli,"deploy","--prod","--yes","--scope",SCOPE,"--project",PROJECT,"--no-color"],
                   cwd=site_files, check=True)
    url = f"{LIVE}/runs/{slug}"
    published_bytes(f"{url}/manifest.json", lambda body: json.loads(body) == data)
    for view in ("scores", "review"):
        published_bytes(f"{url}/{view}", lambda body: f'id="{view}-view"'.encode() in body)
        print(f"{view}: {url}/{view}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    inputs = cli.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--report", type=Path, help="Verified existing stage report, with --queue")
    inputs.add_argument("--conditions", type=Path, help="Generate the stage report and queue first")
    inputs.add_argument("--fixture", type=Path, help="Compare normal local-grid outputs for a front-full or Combined V1 fixture")
    cli.add_argument("--queue",type=Path)
    cli.add_argument("--frames",type=Path)
    cli.add_argument("--run-root",type=Path)
    cli.add_argument("--evidence-root",type=Path,help="Candidate variant directory or paired evidence directory")
    cli.add_argument("--baseline-evidence-root",type=Path,help="Baseline variant directory for local-grid pairs")
    cli.add_argument("--variant",help="Source variant for this comparison (sky if available, otherwise original)")
    cli.add_argument("--case",action="append",default=[],help="Also include this case in a generated visual queue")
    cli.add_argument("--run",required=True)
    cli.add_argument("--title",required=True)
    cli.add_argument("--candidate",default="current")
    cli.add_argument("--baseline",default="split-border")
    cli.add_argument("--candidate-name")
    cli.add_argument("--baseline-name")
    cli.add_argument("--cohort-label")
    cli.add_argument("--visible-available",type=int)
    cli.add_argument("--perfect",type=Path)
    cli.add_argument("--perfect-inputs",type=Path)
    cli.add_argument("--labels",type=Path)
    cli.add_argument("--questions",type=Path,help="JSON array of unresolved decisions with visible cues and distinct next tests")
    cli.add_argument("--carry-forward",help="Next baseline, each candidate disposition, and what landed")
    cli.add_argument("--out",type=Path,required=True,help="Local output directory for this run")
    cli.add_argument("--publish",action="store_true")
    cli.add_argument("--app-root", type=Path, help="Review app checkout for publishing")
    cli.add_argument("--vercel",default=shutil.which("vercel"))
    args = cli.parse_args()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.run):
        cli.error("Use a lowercase hyphenated run ID.")
    if bool(args.report) != bool(args.queue):
        cli.error("Supply --report and --queue together, or use --conditions.")
    if args.conditions and (not args.cohort_label or not args.visible_available):
        cli.error("Supply the actual --cohort-label and --visible-available.")
    if args.fixture and not (args.evidence_root and args.baseline_evidence_root and args.candidate_name and args.baseline_name):
        cli.error("--fixture requires both variant evidence directories and both method names.")
    if not args.fixture and (not args.frames or (not args.run_root and not args.evidence_root)):
        cli.error("Supply --frames and a run or evidence root.")
    if args.conditions and not args.run_root:
        cli.error("--conditions requires --run-root for visual receipts.")
    if args.report and args.case:
        cli.error("Use --case when generating a queue from a fixture or conditions.")
    if args.fixture and (args.candidate != "current" or args.perfect or args.perfect_inputs):
        cli.error("Local-grid fixtures use the current candidate slot. Verified Perfect archives use --report or --conditions.")
    out = args.out.resolve()
    assets = out / "assets"
    out.mkdir(parents=True, exist_ok=True)
    report, queue, frames = args.report, args.queue, args.frames
    authored = read(args.questions) if args.questions else []
    if not isinstance(authored, list):
        cli.error("--questions must contain an array of decision proposals.")
    requested = list(dict.fromkeys(args.case + [q["caseId"] for q in authored]))
    if args.fixture:
        from benchmark.floating_local_review import prepare
        data, queue, frames = prepare(args.fixture, args.baseline_evidence_root, args.evidence_root,
            args.variant, out, args.cohort_label,
            args.visible_available, {"baseline":"Baseline — " + args.baseline_name,
            "current":"Candidate 1 — " + args.candidate_name},case_ids=requested)
        report = out / "stage/report.json"
        save(report, data)
    if args.conditions:
        cmd = [sys.executable, "-m", "benchmark.floating_stage_report", args.conditions, "--out", out / "stage",
               "--run-root", args.run_root, "--baseline", args.baseline, "--candidate", args.candidate,
               "--cohort-label", args.cohort_label, "--visible-available", str(args.visible_available)]
        for key in ("perfect", "perfect_inputs","baseline_name","candidate_name","carry_forward"):
            if value := getattr(args, key):
                cmd.extend(["--" + key.replace("_", "-"), value])
        subprocess.run(cmd, check=True)
        report = out / "stage/report.json"
        requested = list(dict.fromkeys([*requested, *(
            frame["id"] for row in (read(report).get("perfect_stage") or {}).get("rows", [])
            if row["stage"] in ("Surface Finder", "Surface and depth separation")
            for frame in surface_examples(row))]))
        queue = out / "queue"
        subprocess.run([sys.executable,"-m","benchmark.floating_visual_review",report,args.conditions,
                        "--run-root",args.run_root,"--frame-root",args.frames,"--candidate",args.candidate,
                        "--out",queue,*(arg for marker in requested for arg in ("--case",marker))], check=True)
    data = manifest(read(report),queue,frames,assets,"assets",args.run,args.title,args.candidate,
                    read(args.labels) if args.labels else {},args.run_root,
                    questions=authored,evidence_root=args.evidence_root,baseline_evidence_root=args.baseline_evidence_root,
                    carry_forward=args.carry_forward)
    save(out / "manifest.json", data)
    print(f"Prepared {len(data['cases'])} cases in {out}")
    if args.publish:
        with tempfile.TemporaryDirectory(prefix="floating-review-") as temp:
            deploy(data, assets, Path(temp), args.vercel, args.app_root)

if __name__ == "__main__":
    main()
