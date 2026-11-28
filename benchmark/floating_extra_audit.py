"""Decode the floating placement exports and audit their precision and recall tradeoff."""

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from subprocess import check_output

import numpy as np

from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.expanded_scores import FrameReference
from benchmark.floating_one_face_control import PANELS, base, inputs
from benchmark.floating_patch_report import baseline
from benchmark.mask_blocks import decoded_pack
from benchmark.relative_face_scoring import evaluate_relative_voxels
from benchmark.strip_visibility_audit import horizontal_slivers


REVIEW = ROOT / "benchmark/reviews/floating-extra-audit-01"
PROPOSER = WORKTREES / "floating-face-proposals"
INTEGRATION = WORKTREES / "floating-visible-builds"
MATCHED_SCORE = ("geometry", "scores")
KINDS = ("strict", "facade")


def panel_inputs(panel):
    if panel == "holdout5":
        source = ROOT / "runs/floating-holdout5"
        cases = read(source / "manifest.json")["cases"]
        return cases, {
            "horizontal": source / "horizontal-cells/assembly",
            "proposed": ROOT / "runs/face-proposals/holdout5/assembly",
            "whole-wall": source / "source_material/assembly",
        }
    case_root = {"holdout3": "new6", "holdout4": "new7"}.get(panel, panel)
    roots = {"horizontal": base(panel), "proposed": PROPOSER / "runs/face-proposals" / case_root / "assembly"}
    if panel == "sky":
        roots["connected-atlas"] = INTEGRATION / "runs/floating-cloud-gate-01/relative-face-assembly"
        roots["whole-wall"] = baseline(panel)
    return inputs(panel)[3], roots


def load_reference(case):
    folder = Path(case["frame"])
    labels = read(folder / "labels.json")
    faces = {tuple(row["xyz_world"]) for row in labels["faces"]}
    truth = {tuple(row["xyz_world"]): row["block_id"] for row in labels["blocks"]}
    if faces != truth.keys():
        raise ValueError(f"Visible block and face coordinate sets differ: {folder}")
    with np.load(folder / "labels.npz", allow_pickle=False) as regions:
        optional = horizontal_slivers(labels, regions["face_id"])
    hashes = {name: digest(folder / name) for name in ("camera.json", "labels.json", "labels.npz")}
    return FrameReference(truth, optional, {}, {}, hashes)


def score_pack(path, reference, source_sha):
    receipt = read(path / "score.json")
    pack = path / "reconstruction.zip"
    if (receipt["pack_sha256"] != digest(pack) or receipt["source_sha256"] != source_sha
            or receipt["reference_sha256"] != reference.hashes):
        raise ValueError(f"Pack, source or visible reference differs from assembly receipt: {path}")
    predicted = decoded_pack(pack)
    saved = read(path / "placements.json")
    if predicted != {tuple(p): m for p, m in saved["local"]}:
        raise ValueError(f"Decoded cells differ from recorded coordinates and IDs: {path}")
    fixed = receipt["score"]["fixed"]
    grades, blocks = evaluate_relative_voxels(
        predicted, reference, yaw_quarters=fixed["yaw_quarters"] if fixed else None, relaxed_yaw=not fixed
    )
    audited = grades["fixed"] or grades["relaxed_yaw"]
    original = fixed or receipt["score"]["relaxed_yaw"]
    if (blocks != {tuple(p): m for p, m in saved["aligned"]}
            or any(audited[field] != original[field] for field in ("yaw_quarters", "translation", "geometry"))
            or any(audited["scores"][kind] != original["scores"][kind] for kind in KINDS)):
        raise ValueError(f"Recomputed scoring differs from the archived score: {path}")
    evidence = {key: receipt[key] for key in ("source_sha256", "reference_sha256", "pack_sha256", "chart_sha256")}
    return original, blocks, dict(evidence, compass_fixed=fixed is not None)


def compare(case, roots):
    reference = load_reference(case)
    source = case["variants"]["sky"]
    source_sha = digest(Path(source["image"]))
    if source_sha != source["source_sha256"]:
        raise ValueError(f"Image diverges from benchmark fixture: {source['image']}")
    scores, positions, evidence = {}, {}, {}
    for arm, root in roots.items():
        scores[arm], positions[arm], evidence[arm] = score_pack(
            root / "connected-atlas/sky" / case["id"], reference, source_sha
        )
    truth = reference.truth
    base_pos, new_pos = (positions[arm].keys() for arm in ("horizontal", "proposed"))
    gained, lost = new_pos - base_pos, base_pos - new_pos
    old_ids, new_ids = ({p for p, value in positions[arm].items() if truth.get(p) == value}
                        for arm in ("horizontal", "proposed"))
    return dict(id=case["id"], dataset=case["dataset"], scores=scores, evidence=evidence,
                paired=dict(gained=len(gained & truth.keys()), lost=len(lost & truth.keys()),
                            added=len(gained - truth.keys()), removed=len(lost - truth.keys()),
                            exact_ids_gained=len(new_ids - old_ids), exact_ids_lost=len(old_ids - new_ids)))


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def rates(counts):
    true, missed, extra = (counts[k] for k in ("correct", "missed", "extras"))
    return dict(counts, precision=safe_ratio(true, true + extra), recall=safe_ratio(true, true + missed),
                f1=safe_ratio(2 * true, 2 * true + missed + extra))


def compare_counts(after, before, metric):
    def rate(counts):
        hit, miss, extra = (counts[k] for k in ("correct", "missed", "extras"))
        if metric == "precision":
            return Fraction(hit, hit + extra) if hit + extra else None
        if metric == "recall":
            return Fraction(hit, hit + miss) if hit + miss else None
        return Fraction(2 * hit, 2 * hit + miss + extra) if hit + miss + extra else None

    a, b = rate(after), rate(before)
    if a is None or b is None:
        return "tied" if a == b else "undefined"
    return "better" if a > b else "worse" if a < b else "tied"


def pool(rows):
    arms = {}
    common = set.intersection(*(set(row["scores"]) for row in rows))
    for arm in rows[0]["scores"]:
        if arm not in common:
            continue
        counts = {field: {kind: Counter() for kind in KINDS} for field in MATCHED_SCORE}
        for row in rows:
            score = row["scores"][arm]
            for field in MATCHED_SCORE:
                for kind in KINDS:
                    counts[field][kind].update({key: score[field][kind][key] for key in
                                               ("correct", "missed", "extras")})
        arms[arm] = {field: {kind: rates(counts[field][kind]) for kind in KINDS} for field in MATCHED_SCORE}
        arms[arm]["id_accuracy"] = safe_ratio(arms[arm]["scores"]["strict"]["correct"],
                                             arms[arm]["geometry"]["strict"]["correct"])
    paired = Counter()
    directions = {field: {kind: {metric: Counter() for metric in ("precision", "recall", "f1")}
                          for kind in KINDS} for field in MATCHED_SCORE}
    pareto = Counter()
    for row in rows:
        paired.update(row["paired"])
        after, before = (row["scores"][arm] for arm in ("proposed", "horizontal"))
        for field in MATCHED_SCORE:
            for kind in KINDS:
                for metric, tally in directions[field][kind].items():
                    tally[compare_counts(after[field][kind], before[field][kind], metric)] += 1
        delta_hits = after["geometry"]["strict"]["correct"] - before["geometry"]["strict"]["correct"]
        delta_extras = after["geometry"]["strict"]["extras"] - before["geometry"]["strict"]["extras"]
        if delta_hits >= 0 and delta_extras <= 0 and (delta_hits or delta_extras):
            pareto["better"] += 1
        elif delta_hits <= 0 and delta_extras >= 0 and (delta_hits or delta_extras):
            pareto["worse"] += 1
        else:
            pareto["tradeoff" if delta_hits or delta_extras else "tied"] += 1
    hit_loss = paired["lost"] - paired["gained"]
    extras_saved = paired["removed"] - paired["added"]
    before, after = (arms[arm]["geometry"]["strict"] for arm in ("horizontal", "proposed"))
    if (before["correct"] - after["correct"], before["extras"] - after["extras"]) != (hit_loss, extras_saved):
        raise ValueError("Gross paired changes do not reconcile with pooled counts")
    return dict(views=len(rows), arms=arms, paired=dict(paired), view_directions=directions, pareto=pareto,
                extras_saved=extras_saved, hit_loss=hit_loss,
                extras_saved_per_hit_lost=safe_ratio(extras_saved, hit_loss) if hit_loss > 0 and extras_saved > 0 else None,
                extra_penalty_breakeven=safe_ratio(hit_loss, extras_saved) if hit_loss > 0 and extras_saved > 0 else None)


def verify_saved(rows, archived):
    saved = {item["id"]: item for item in archived["rows"]}
    if saved.keys() != {row["id"] for row in rows}:
        raise ValueError("Archived and decoded frame identities disagree")
    for row in rows:
        target = saved[row["id"]]
        if (any(row["scores"][arm] != score for arm, score in target["score"].items())
                or any(row["paired"][key] != val for key, val in target["paired"]["proposed"].items())):
            raise ValueError(f"Decoded audit differs from published paired result: {row['id']}")


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--workers", type=int, default=3)
    cli.add_argument("--holdout5", action="store_true")
    args = cli.parse_args()
    archived = read(PROPOSER / "benchmark/reviews/floating-face-proposals-01/placement.json.gz")
    groups = [*PANELS, "holdout3", "holdout4"]
    if args.holdout5:
        groups.append("holdout5")
    panels = {}
    with ProcessPoolExecutor(args.workers) as workers:
        for group in groups:
            cases, roots = panel_inputs(group)
            rows = list(workers.map(compare, cases, [roots] * len(cases)))
            if group != "holdout5":
                verify_saved(rows, archived[{"holdout3": "new6", "holdout4": "new7"}.get(group, group)])
            panels[group] = dict(rows=rows, total=pool(rows))
            print(group, panels[group]["total"]["paired"], flush=True)
    sky_ids = {"validation-hillside-chapel-1-warm-low-left"}
    mill_ids = {"train-riverside-mill-1-warm-high-right"}
    subsets = {
        "original86": [row for group in groups if group != "holdout5" for row in panels[group]["rows"]],
        "sky_without_chapel": [r for r in panels["sky"]["rows"] if r["id"] not in sky_ids],
        "holdout4_without_mill_high_right": [r for r in panels["holdout4"]["rows"] if r["id"] not in mill_ids],
    }
    subsets["original86_without_both_outliers"] = [r for r in subsets["original86"]
                                                   if r["id"] not in sky_ids | mill_ids]
    if args.holdout5:
        subsets["all92"] = [row for group in groups for row in panels[group]["rows"]]
    for dataset in ("visible-builds-v2", "visible-builds-v3", "decorative-builds-v1"):
        subsets[dataset] = [r for r in subsets["original86"] if r["dataset"] == dataset]
    for subset, rows in subsets.items():
        panels[subset] = dict(total=pool(rows))
    revision = check_output(["git", "-C", str(INTEGRATION), "rev-parse", "HEAD"], text=True).strip()
    runtime_revision = check_output(
        ["git", "-C", str(INTEGRATION), "log", "-1", "--format=%H", "--", "engine", "reconstruct.py",
         "benchmark/floating_build_local_run.py", "benchmark/shared_corner_rows.py",
         "benchmark/native_surface_counter.py", "benchmark/native_corner_registration.py",
         "benchmark/garden_anchor_extension.py", "benchmark/relative_face_round.py"], text=True
    ).strip()
    save(REVIEW / "paired-audit.json.gz", dict(integration_revision=revision, runtime_revision=runtime_revision,
                                               panels=panels))


if __name__ == "__main__":
    main()
