"""Re-score archived round 2 material rankings by the native width of the sampled face."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from engine.face_evidence import MIN_FACE_WIDTH, effective_width


def counts(rows):
    total = len(rows)
    eligible = sum(row["effective_width_px"] >= MIN_FACE_WIDTH for row in rows)
    return dict(total=total, eligible=eligible, under_minimum=total-eligible)


def counts_by_kind(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["kind"], []).append(row)
    return {kind: counts(group) for kind, group in groups.items()}


def rescore(path):
    archived = json.loads(gzip.decompress(path.read_bytes()))
    cohorts = {"fixed": [], "paired_known": [], "predicted": []}
    for row in archived["rows"]:
        row["effective_width_px"] = effective_width(row["quad"])
        if row["fixed"]:
            cohorts["fixed"].append(row)
        if row["paired"]:
            cohorts["paired_known"].append(row)
        if row["group"] == "predicted":
            cohorts["predicted"].append(row)
    scores = {}
    for kind, rows in cohorts.items():
        score = counts(rows)
        for name, field in (("exposure_shadow", "block"), ("shape_shadow", "shape_block")):
            score[name] = {
                "correct_eligible": sum(r[field] == r["truth"] for r in rows if r["effective_width_px"] >= MIN_FACE_WIDTH),
                "correct_under_minimum_in_archive": sum(r[field] == r["truth"] for r in rows if r["effective_width_px"] < MIN_FACE_WIDTH),
            }
        scores[kind] = score
    return dict(source=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), scores=scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    archive = Path(__file__).resolve().parent / "reviews/blocker-picker-02"
    report = dict(min_face_width_px=MIN_FACE_WIDTH,
                  archives=[rescore(archive / name) for name in ("shadow-final.json.gz", "plane-shadow-final.json.gz")])
    args.out.write_text(json.dumps(report, indent=2) + "\n")
