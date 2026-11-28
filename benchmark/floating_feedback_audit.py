"""Regrade the 92 archived floating conditions and exact arms shown in the feedback review."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import benchmark

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.floating_extra_budget import LOCATION, paths
from benchmark.floating_extra_feedback import CASES
from benchmark.floating_one_face_control import PANELS

# Use the same reference and decoded-pack validation as the preceding audit.
AUDIT = WORKTREES / "floating-extra-audit"
benchmark.__path__.append(str(AUDIT / "benchmark"))
from benchmark.floating_extra_audit import load_reference, score_pack

ISOLATION = WORKTREES / "floating-extra-isolation"
SEAMS = WORKTREES / "floating-distant-seams"
REVIEW = ROOT / "benchmark/reviews/floating-feedback-audit-01"
GROUPS = (*PANELS, "holdout3", "holdout4", "holdout5")
METHODS = ("floating-default", "baseline", "source-one", "large-contraction")
SCORE_FIELDS = ("geometry", "scores")
TARGETS = ("strict", "facade")
COUNTS = ("correct", "missed", "extras")


def b_root(panel, ruler, gate):
    if panel == "holdout5":
        return gate.parent.parent / "sparse-mask-0/assembly"
    if panel in ("original", "holdout3"):
        return SEAMS / "runs/distant-seams" / panel / "assembly"
    return ruler


def changes(reference, before, after):
    gained = after.keys() - before.keys()
    lost = before.keys() - after.keys()
    before_exact = {p for p, name in before.items() if reference.truth.get(p) == name}
    after_exact = {p for p, name in after.items() if reference.truth.get(p) == name}
    return dict(
        correct_gained=len(gained & reference.truth.keys()),
        correct_lost=len(lost & reference.truth.keys()),
        extras_added=len(gained - reference.truth.keys()),
        extras_removed=len(lost - reference.truth.keys()),
        exact_ids_gained=len(after_exact - before_exact),
        exact_ids_lost=len(before_exact - after_exact),
    )


def verify(job):
    panel, case, methods, archived, feedback = job
    frame = case["id"]
    reference = load_reference(case)
    source = case["variants"]["sky"]
    source_sha = digest(Path(source["image"]))
    if source_sha != source["source_sha256"] or source_sha != archived["source_sha256"]:
        raise ValueError(f"Source image differs from frozen input: {panel}/{frame}")
    scores, aligned, evidence = {}, {}, {}
    for arm, root in methods.items():
        export = root / LOCATION / frame
        scores[arm], aligned[arm], evidence[arm] = score_pack(export, reference, source_sha)
        expected = archived["score"].get(arm)
        if expected is not None and scores[arm] != expected:
            raise ValueError(f"Archive and decoded score disagree: {panel}/{frame}/{arm}")
        old_export = archived["exports"].get(arm)
        if old_export and evidence[arm]["pack_sha256"] != old_export["pack_sha256"]:
            raise ValueError(f"Archive and decoded pack hashes disagree: {panel}/{frame}/{arm}")
    row = {key: case[key] for key in ("id", "dataset", "layout", "palette", "split", "rgb_sha256")}
    row.update(panel=panel, source_sha256=source_sha, reference_sha256=reference.hashes,
               dataset_manifest_sha256=case.get("dataset_manifest_sha256"),
               old_wall_packet_sha256=archived["old_packet_sha256"],
               proposal_packet_sha256=archived["proposal_packet_sha256"],
               contraction_charts=archived["selected"],
               scores={arm: {field: {mode: score[field][mode] for mode in TARGETS}
                             for field in SCORE_FIELDS} for arm, score in scores.items()},
               evidence={arm: {key: val for key, val in record.items() if key not in
                               ("source_sha256", "reference_sha256")} for arm, record in evidence.items()},
               paired={arm: changes(reference, aligned["baseline"], cells)
                       for arm, cells in aligned.items() if arm != "baseline"})
    if feedback:
        wall, preference, label = feedback
        _, seed_root, wall_root, _ = paths(panel)[0]
        charts_key = Path("sky") / frame / "charts.json"
        proposal_root = seed_root if label == "sparse-wall-bypass" else methods["source-one"].parent / "charts"
        chart_lengths = [len(next(chart["cells"] for chart in read(root / charts_key)["charts"]
                                  if chart["id"] == wall)) for root in (wall_root, proposal_root)]
        origins = {arm: list(read(methods[arm] / LOCATION / frame / "score.json")["selected_origins"])
                   for arm in ("baseline", "offered-b", "large-contraction")}
        row["feedback"] = dict(wall=wall, preferred=preference, b_method=label,
                               shown_wall_cells=dict(zip(("A", "B"), chart_lengths, strict=True)),
                               exported_charts=origins, b_matches_gate=aligned["offered-b"] == aligned["large-contraction"],
                               a_matches_gate=aligned["baseline"] == aligned["large-contraction"],
                               b_read_from=str(methods["offered-b"].relative_to(WORKTREES).as_posix()))
    return row


def pooled(rows, arm):
    result = {}
    for field in SCORE_FIELDS:
        result[field] = {}
        for mode in TARGETS:
            count = {key: sum(row["scores"][arm(row)][field][mode][key] for row in rows) for key in COUNTS}
            denominator = sum(count.values()) + count["correct"]
            result[field][mode] = dict(count, f1=2 * count["correct"] / denominator if denominator else None)
    return result


def perfect_choice(rows):
    """Try the 64 complete A/B combinations and maximize pooled strict placement F1."""
    frames = [(row["panel"], row["id"]) for row in rows]
    best = None
    for arms in product(("baseline", "offered-b"), repeat=len(rows)):
        choices = dict(zip(frames, arms, strict=True))
        scores = pooled(rows, lambda row: choices[row["panel"], row["id"]])
        if best is None or scores["geometry"]["strict"]["f1"] > best["geometry"]["strict"]["f1"]:
            best = scores
            selected = dict(zip((row["id"] for row in rows), arms, strict=True))
    return best, selected


def preferred(row):
    return {"A": "baseline", "B": "offered-b", "Neither": "large-contraction"}[row["feedback"]["preferred"]]


def load_jobs(feedback):
    expected = {frame for _, frame, _, _, _ in CASES}
    if expected != feedback.keys():
        raise ValueError("Feedback does not match the six published review cases")
    offered = {(panel, frame): wall for panel, frame, wall, _, _ in CASES}
    jobs = []
    for panel in GROUPS:
        (source, _, _, cases), old, ruler = paths(panel)
        gate = ISOLATION / "runs/extra-isolation" / panel / "large-contraction/assembly"
        ordinary = source / ("relative-face-assembly" if panel == "sky" else "assembly")
        archive = read(ISOLATION / "benchmark/reviews/floating-extra-isolation-01" / (panel + ".json.gz"))
        by_id = {row["id"]: row for row in archive}
        if by_id.keys() != {case["id"] for case in cases} or len(cases) != len(archive):
            raise ValueError(f"Archive and fixture cases disagree: {panel}")
        for case in cases:
            frame = case["id"]
            arms = dict(zip(METHODS, (ordinary, old, ruler, gate), strict=True))
            wall = offered.get((panel, frame))
            review = None
            if wall:
                answer = feedback[frame]
                if answer["wall"] != wall or answer["preferred"] not in ("A", "B", "Neither"):
                    raise ValueError(f"Invalid feedback entry: {frame}")
                method = {"holdout5": "sparse-wall-bypass", "original": "distant-seams",
                          "holdout3": "distant-seams"}.get(panel, "source-one")
                arms["offered-b"] = b_root(panel, ruler, gate)
                if method == "distant-seams":
                    seam_rows = read(SEAMS / "benchmark/reviews/floating-distant-seams-01" / (panel + ".json.gz"))
                    seam = next(row for row in seam_rows["rows"] if row["id"] == frame)["score"]["distant"]
                    by_id[frame]["score"]["offered-b"] = seam
                else:
                    key = "sparse-mask-0" if method == "sparse-wall-bypass" else "source-one"
                    by_id[frame]["score"]["offered-b"] = by_id[frame]["score"][key]
                    by_id[frame]["exports"]["offered-b"] = by_id[frame]["exports"][key]
                review = (wall, answer["preferred"], method)
            jobs.append((panel, case, arms, by_id[frame], review))
    if len(jobs) != 92 or len({(panel, case["id"]) for panel, case, *_ in jobs}) != 92:
        raise ValueError("Expected exactly 92 unique panel/frame pairs")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feedback", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    jobs = load_jobs(read(args.feedback))
    with ProcessPoolExecutor(args.workers) as workers:
        rows = list(workers.map(verify, jobs))
    reviewed = [row for row in rows if "feedback" in row]
    layouts = {row["layout"] for row in reviewed}
    for row in rows:
        row["feedback_split"] = "reviewed-layout" if row["layout"] in layouts else "other-archived-layout"
    regression = [row for row in rows if row["layout"] in layouts]
    comparison = [row for row in rows if row["layout"] not in layouts]
    decisive = [row for row in reviewed if row["feedback"]["preferred"] != "Neither"]
    cohorts = {
        "all92": rows,
        "reviewed-layouts": regression,
        "other-archived-layouts": comparison,
        "previous86": [row for row in rows if row["panel"] != "holdout5"],
        **{panel: [row for row in rows if row["panel"] == panel] for panel in GROUPS},
        **{dataset: [row for row in rows if row["dataset"] == dataset]
           for dataset in sorted({row["dataset"] for row in rows})},
    }
    totals = {}
    for name, cohort in cohorts.items():
        totals[name] = dict(
            conditions=len(cohort), unique_frames=len({row["id"] for row in cohort}),
            layouts=len({row["layout"] for row in cohort}),
            scores={arm: pooled(cohort, lambda row, arm=arm: arm) for arm in METHODS},
            gains={arm: dict(sum((Counter(row["paired"][arm]) for row in cohort), Counter()))
                   for arm in METHODS if arm != "baseline"},
        )
        print(name, len(cohort), {arm: f"{100 * s['geometry']['strict']['f1']:.4f}"
                                 for arm, s in totals[name]["scores"].items()}, flush=True)
    best, selection = perfect_choice(reviewed)
    choices = {
        "five-decisive": {**{arm: pooled(decisive, lambda row, arm=arm: arm)
                            for arm in ("baseline", "offered-b", "large-contraction")},
                          "feedback-choice": pooled(decisive, preferred)},
        "six-with-abstention": {"feedback-choice-else-gate": pooled(reviewed, preferred),
                                "large-contraction": pooled(reviewed, lambda row: "large-contraction"),
                                "best-offered-by-strict-f1": best},
    }
    revisions = {
        "feedback-review": "0d2eb20d4341736436fc9504631c5b66edf821e6",
        "integration-runtime": "52de2c9b0d9f80c1c008fb2a4a10073be948658f",
        "archive-audit": "c4d58a2f765d7a6e3258bc2dd2e773bba518a8df",
        "isolation": "e97dd44f84051c644fed5769d0697aaa7a08581b",
        "distant-seams": "28f9bbe8bb0366fef1cb175304ff8d3732b1c360",
    }
    save(REVIEW / "decoded.json.gz", dict(revisions=revisions, rows=rows, totals=totals, choices=choices,
                                        reference_selection=selection))
    print("preferences", Counter(row["feedback"]["preferred"] for row in reviewed), flush=True)


if __name__ == "__main__":
    main()
