"""Decode the frozen round-two floating comparisons on the same 84 active conditions."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from benchmark import __path__ as benchmark_paths
from benchmark.expanded_baseline import WORKTREES, digest, read, save
from benchmark.floating_extra_budget import LOCATION, paths
from benchmark.floating_second_signoff import ISOLATION, PART, REFIT, RULER
from benchmark.strip_visibility_audit import horizontal_slivers


AUDIT = WORKTREES / "floating-signoff-audit"
BASELINE = AUDIT / "benchmark/reviews/floating-signoff-audit-01/audit.json.gz"
SEAM = RULER.parent.parent / "benchmark/reviews/floating-majority-edge/cohort.json.gz"
DEPTH = REFIT.parent.parent.parent / "runs/depth-refit"
REFIT_ARMS = ("source-depth", "source-refit", "source-refit-with-small-steps", "signed-boundary-guard")

# Use the same comparison of decoded blocks, saved registration and reference scores
# as the previous independent audit.
benchmark_paths.append(str(AUDIT / "benchmark"))
from benchmark.floating_signoff_audit import METRICS, paired, pool, verify_export


def verify_row(job):
    prior, depth, seam, case, reviewed = job
    name, panel = case["id"], prior["panel"]
    image = Path(case["variants"]["sky"]["image"])
    frame = Path(case["frame"])
    source_hash = digest(image)
    assert source_hash == prior["source_sha256"] == depth["source_sha256"] == seam["source_sha256"]
    assert source_hash == case["variants"]["sky"]["source_sha256"]
    reference_hash = {key: digest(frame / key) for key in prior["reference_sha256"]}
    assert reference_hash == prior["reference_sha256"]
    labels = read(frame / "labels.json")
    truth = {tuple(block["xyz_world"]): block["block_id"] for block in labels["blocks"]}
    with np.load(frame / "labels.npz", allow_pickle=False) as data:
        optional = horizontal_slivers(labels, data["face_id"])
    base = ISOLATION / panel / "large-contraction/assembly"
    depth_prior = PART / panel / "source/assembly"
    specs = {"large-contraction": (base, seam["scores"]["score"]["previous"], prior["pack_sha256"]["large-contraction"])}
    for arm in REFIT_ARMS:
        expected_hash = depth["pack_sha256"][arm]
        if expected_hash == prior["pack_sha256"]["large-contraction"]:
            root = base
        elif arm == "source-depth" or expected_hash == depth["pack_sha256"]["source-depth"]:
            root = depth_prior
        else:
            root = DEPTH / panel / arm / "assembly"
            if arm == "source-refit-with-small-steps" and not (root / LOCATION / name).is_dir():
                assert expected_hash == depth["pack_sha256"]["source-refit"]
                root = DEPTH / panel / "source-refit/assembly"
        specs[arm] = root, depth["score"][arm], expected_hash
    # The original run reuses the first earlier arm with identical chart inputs.
    rel = Path("charts/sky") / name / "charts.json"
    packets = {key: RULER / "cohort" / panel / key for key in ("prior", "majority", "seam")}
    packets = {"previous": base.parent, **packets}
    assert digest(packets["previous"] / rel) == seam["input_sha256"]
    wanted = read(packets["seam"] / rel)["charts"]
    chosen = next(root for root in packets.values() if read(root / rel)["charts"] == wanted)
    specs["majority-and-seam"] = chosen / "assembly", seam["scores"]["score"]["seam"], None
    source = paths(panel)[0][0]
    integration = source / ("relative-face-assembly" if panel == "sky" else "assembly")
    integrated = read(integration / LOCATION / name / "score.json")
    specs["connected-atlas"] = integration, integrated["score"]["fixed"] or integrated["score"]["relaxed_yaw"], prior["pack_sha256"]["connected-atlas"]
    results, aligned, hashes = {}, {}, {}
    for arm, (directory, expected_score, expected_hash) in specs.items():
        aligned[arm], receipt = verify_export(directory / LOCATION / name, expected_score, truth, optional,
                                              source_hash, reference_hash, expected_hash)
        results[arm] = {f"{field}_{kind}": expected_score[field][kind] for field, kind in METRICS}
        hashes[arm] = receipt["pack_sha256"]
    assert results["source-depth"] == prior["scores"]["source"]
    assert results["large-contraction"] == prior["scores"]["large-contraction"]
    assert results["connected-atlas"] == prior["scores"]["connected-atlas"]
    pairs = {arm: paired(aligned["source-depth" if arm in REFIT_ARMS else "large-contraction"], blocks, truth)
             for arm, blocks in aligned.items()}
    assert all(pairs[arm] == depth["paired"][arm] for arm in REFIT_ARMS)
    assert pairs["majority-and-seam"] == seam["scores"]["paired"]["seam"]
    return dict(panel=panel, id=name, dataset=case["dataset"], layout=case["layout"], split=case["split"],
                source_sha256=source_hash, reference_sha256=reference_hash, pack_sha256=hashes,
                reviewed=(case["dataset"], case["layout"]) in reviewed, scores=results, paired=pairs)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("feedback", type=Path)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--workers", type=int, default=4)
    args = cli.parse_args()
    depth_file = REFIT / "conditions.json.gz"
    prior_rows = read(BASELINE)["conditions"]
    depth = {(r["panel"], r["id"]): r for r in read(depth_file) if r["active"]}
    seam = {(r["panel"], r["scores"]["id"]): r for r in read(SEAM)["rows"]}
    cases = {(panel, r["id"]): r for panel in {r["panel"] for r in prior_rows} for r in paths(panel)[0][3]}
    reviewed_ids = {r["id"].rsplit("-", 1)[0] for r in read(args.feedback)["reviews"].values()}
    reviewed = {(r["dataset"], r["layout"]) for r in cases.values() if r["id"] in reviewed_ids}
    assert len(reviewed) == 8
    jobs = []
    for row in prior_rows:
        key = row["panel"], row["id"]
        jobs.append((row, depth[key], seam[key], cases[key], reviewed))
    assert len(jobs) == len(depth) == len(seam) == 84
    with ProcessPoolExecutor(args.workers) as pool_:
        rows = list(pool_.map(verify_row, jobs))
    groups = {"active": rows, "fixed_sky": [r for r in rows if r["panel"] == "sky"]}
    assert len(groups["fixed_sky"]) == 6
    for group, subset in list(groups.items()):
        groups[f"{group}_reviewed"] = [r for r in subset if r["reviewed"]]
        groups[f"{group}_transfer"] = [r for r in subset if not r["reviewed"]]
    save(args.out, dict(schema="floating-projection-controls-v1", feedback_sha256=digest(args.feedback),
                        archive_sha256={key: digest(path) for key, path in
                                        (("prior", BASELINE), ("depth", depth_file), ("seam", SEAM))},
                        reviewed_layouts=sorted(reviewed), groups={key: pool(g) for key, g in groups.items()},
                        available_active={"visible-builds-v2": 92, "visible-builds-v3": 108, "decorative-builds-v1": 40},
                        cases_by_dataset=Counter(r["dataset"] for r in rows), conditions=rows))


if __name__ == "__main__":
    main()
