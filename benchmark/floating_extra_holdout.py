"""Replay the frozen floating baseline and face proposer on the fifth holdout."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from subprocess import check_output
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
GROUP = "holdout5"
FIXTURE = ROOT / "benchmark/reviews/floating-extra-audit-01/holdout5-frames.json"
OUTPUT = ROOT / "runs/floating-holdout5"
PROPOSALS = ROOT / "runs/face-proposals"
CHECKOUTS = {"baseline": ("floating-patch-validation", "7958722"),
             "proposer": ("floating-face-proposals", "e1f0218")}


def replay(arm, case=None):
    name, expected = CHECKOUTS[arm]
    checkout = ROOT.with_name(name)
    revision = check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if not revision.startswith(expected):
        raise ValueError(f"Expected {name} at {expected}, found {revision}")
    sys.path.insert(0, str(checkout))
    if arm == "baseline":
        from benchmark.floating_patch_validation import fresh, run
        from benchmark.expanded_baseline import save

        if case is not None:
            return run(case, output=OUTPUT)
        entries = list(fresh.cases(FIXTURE))
        save(OUTPUT / "manifest.json", dict(frozen_manifest_sha256=fresh.fixture_hash(FIXTURE), cases=entries))
        return entries

    if case is None:
        return json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))["cases"]

    from benchmark import floating_face_proposal_run as proposer

    source_inputs = proposer.source_inputs
    with patch.object(proposer, "OUT", PROPOSALS), patch.object(
        proposer, "source_inputs", lambda group, path: source_inputs("new6", path)
    ):
        return proposer.build((GROUP, case, str(OUTPUT)))


def run_case(arm, entry):
    if arm == "baseline":
        receipt = OUTPUT / "horizontal-cells/assembly/connected-atlas/sky" / entry["id"] / "score.json"
    else:
        receipt = PROPOSALS / GROUP / "assembly/connected-atlas/sky" / entry["id"] / "score.json"
    if receipt.is_file():
        return entry["id"], "already complete"
    return replay(arm, entry)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=CHECKOUTS)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    os.environ.update(MCIR_IMAGE_ONLY="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    entries = replay(args.arm)
    if len(entries) != 6:
        raise ValueError("Expected exactly six frozen frames")
    with ProcessPoolExecutor(args.workers) as pool:
        for result in as_completed(pool.submit(run_case, args.arm, entry) for entry in entries):
            print("COMPLETE", result.result(), flush=True)


if __name__ == "__main__":
    main()
