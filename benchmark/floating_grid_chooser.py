"""Offline, one-local-grid-decision-per-view pilot using verified paired export receipts."""
import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

FEATURES = ("edge_alignment", "atlas_agreement", "visible_support")
METRICS = ("geometry_strict", "geometry_facade", "scores_strict", "scores_facade")
COUNTS = ("correct", "missed", "extras")
SIMPLE = {"rank": "edge_alignment", "gate": None}
KEEP = {"rank": None, "gate": None}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(row, expected):
    require(set(row) == set(expected), f"Expected fields {expected}, received {list(row)}")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(base, ref):
    keys(ref, ("path", "sha256"))
    require(digest(base / ref["path"]) == ref["sha256"], f"Hash mismatch for {ref['path']}")
    return ref["sha256"]


def read_json(path):
    def unique(pairs):
        obj = dict(pairs)
        require(len(obj) == len(pairs), f"Duplicate JSON key in {path}")
        return obj
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def source_views(path):
    data = read_json(path)
    keys(data, ("cohort", "revision", "views"))
    require(data["cohort"] and data["revision"], "Identify cohort and downstream revision")
    families, screenshots, ids = {}, set(), set()
    for view in data["views"]:
        keys(view, ("id", "family", "split", "decision", "screenshot", "surroundings", "incumbent", "options"))
        require(all(isinstance(view[k], str) and view[k].strip() for k in ("id", "family", "decision")),
                "Missing view, building family, or decision identity")
        split = view["split"]
        require(split in ("development", "holdout"), "Unknown split")
        require(families.setdefault(view["family"], split) == split, "Building family crosses splits")
        image = artifact(path.parent, view["screenshot"])
        artifact(path.parent, view["surroundings"])
        require(image not in screenshots and view["id"] not in ids, "Duplicate screenshot or decision view")
        screenshots.add(image)
        ids.add(view["id"])
        options = view["options"]
        require(view["incumbent"] in options and len(options) >= 2, "Need incumbent and an alternative")
        for proposal in options.values():
            keys(proposal, ("grid", "features"))
            artifact(path.parent, proposal["grid"])
            features = proposal["features"]
            keys(features, FEATURES)
            require(all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1
                        for v in features.values()), "Features must be finite source fractions in [0,1]")
    return data


def scored_exports(path, packet_hash, views):
    receipt = read_json(path)
    keys(receipt, ("source_sha256", "evaluator", "views"))
    evaluator = artifact(path.parent, receipt["evaluator"])
    require(receipt["source_sha256"] == packet_hash, "Scores belong to another source packet")
    require(set(receipt["views"]) == {v["id"] for v in views}, "Missing or unexpected scored view")
    scores = {}
    for v in views:
        record = receipt["views"][v["id"]]
        keys(record, ("visible_reference", "options"))
        artifact(path.parent, record["visible_reference"])
        labels = record["options"]
        require(set(labels) == set(v["options"]), "Missing or unexpected scored option")
        by_export = {}
        targets = None
        for name, label in labels.items():
            keys(label, ("surroundings_sha256", "grid_sha256", "export", "scores"))
            require(label["grid_sha256"] == v["options"][name]["grid"]["sha256"], "Scored a different grid option")
            require(label["surroundings_sha256"] == v["surroundings"]["sha256"], "Changed surroundings")
            pack = artifact(path.parent, label["export"])
            score = label["scores"]
            keys(score, METRICS)
            for row in score.values():
                require(all(type(row.get(k)) is int and row[k] >= 0 for k in COUNTS), "Invalid score counts")
            totals = tuple(score[k]["correct"] + score[k]["missed"] for k in METRICS)
            require(targets in (None, totals), "Options use different visible targets")
            targets = totals
            counts = [[score[k][c] for c in COUNTS] for k in METRICS]
            require(by_export.setdefault(pack, counts) == counts, "Identical export scored differently")
        scores[v["id"]] = {opt: label["scores"] for opt, label in labels.items()}
    return scores, evaluator


def choose(options, incumbent, rule):
    """Inference receives source fractions only. Equal rankings keep the incumbent."""
    rank = rule["rank"]
    if rank is None:
        return incumbent
    features = {name: proposal["features"] for name, proposal in options.items()}
    peak = max(x[rank] for x in features.values())
    best = [name for name, x in features.items() if x[rank] == peak]
    if len(best) != 1 or not rule["gate"]:
        return best[0] if len(best) == 1 else incumbent
    gate, margin = rule["gate"]
    winner = best[0]
    return winner if features[winner][gate] - features[incumbent][gate] >= margin else incumbent


def f1(counts):
    hit, miss, extra = (counts[k] for k in COUNTS)
    return Fraction(2 * hit, max(1, 2 * hit + miss + extra))


def objective(scores, name):
    return f1(scores[name]["geometry_strict"])


def oracle(view, scores):
    best = max(objective(scores, name) for name in scores)
    return next(name for name in [view["incumbent"], *sorted(scores)] if objective(scores, name) == best)


def fit(views, scores):
    """Learn an interpretable rank feature and optional independent evidence gate."""
    rules = [KEEP] + [{"rank": rank, "gate": None} for rank in FEATURES]
    rules += [{"rank": rank, "gate": [gate, margin]}
              for rank in FEATURES for gate in FEATURES if rank != gate for margin in (0, .125, .25)]
    def utility(rule):
        return sum(objective(scores[v["id"]], choose(v["options"], v["incumbent"], rule)) for v in views)
    return max(rules, key=utility)


def compare(views, scores, model):
    arms = {"incumbent": KEEP, "edge_alignment": SIMPLE, "development_rule": model, "reference_oracle": None}
    result = {}
    for arm, rule in arms.items():
        totals = {metric: dict.fromkeys(COUNTS, 0) for metric in METRICS}
        decisions = []
        for v in views:
            options = scores[v["id"]]
            selected = oracle(v, options) if rule is None else choose(v["options"], v["incumbent"], rule)
            gain = objective(options, selected) - objective(options, v["incumbent"])
            decisions.append(dict(view=v["id"], family=v["family"], selected=selected, incumbent=v["incumbent"],
                                  strict_f1_delta=float(gain)))
            for metric, counts in totals.items():
                for k in COUNTS:
                    counts[k] += options[selected][metric][k]
        for metric, counts in totals.items():
            counts["f1"] = float(f1(counts))
        base = result.get("incumbent", {}).get("scores", totals)
        delta = {m: {k: totals[m][k] - base[m][k] for k in totals[m]} for m in METRICS}
        gains = [d["strict_f1_delta"] for d in decisions]
        result[arm] = dict(scores=totals, delta_from_incumbent=delta, decisions=decisions,
                           wins=sum(x > 0 for x in gains), losses=sum(x < 0 for x in gains),
                           ties=gains.count(0), mean_view_strict_f1_delta=sum(gains)/len(gains))
    return result


def run(source, development, holdout):
    data = source_views(source)
    packet_hash = digest(source)
    dev, test = ([v for v in data["views"] if v["split"] == split] for split in ("development", "holdout"))
    report = dict(cohort=data["cohort"], revision=data["revision"], source_sha256=packet_hash,
                  development_views=len(dev), holdout_views=len(test))
    if len({v["family"] for v in dev}) < 2 or not test:
        return dict(report, status="insufficient_building_families")
    train_scores, evaluator = scored_exports(development, packet_hash, dev)
    model = fit(dev, train_scores)
    report.update(model=model, development=compare(dev, train_scores, model), development_sha256=digest(development))
    if not report["development"]["reference_oracle"]["wins"]:
        return dict(report, status="no_development_headroom")
    test_scores, test_evaluator = scored_exports(holdout, packet_hash, test)
    require(evaluator == test_evaluator, "Evaluator differs between splits")
    report.update(holdout=compare(test, test_scores, model), holdout_sha256=digest(holdout))
    return dict(report, status="evaluated" if report["holdout"]["reference_oracle"]["wins"] else "no_holdout_headroom")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "development", "holdout", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run(args.source, args.development, args.holdout)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
