"""Verify and summarize the frozen V1 stage interventions using the shared pack decoder."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import gzip
import json
from pathlib import Path
from zipfile import ZipFile

from benchmark.mask_blocks import decoded_pack


REVIEW = Path(__file__).resolve().parents[1] / "benchmark/reviews/floating-perfect-stage-inputs-01"
STAGES = ("surface-mask", "surface-plane", "wall-grid")


def verify(stage, directory):
    rows = json.loads(gzip.decompress((directory / f"{stage}-conditions.json.gz").read_bytes()))
    summary = json.loads((directory / f"{stage}-summary.json").read_bytes())
    totals = {}
    with ZipFile(directory / f"{stage}-inputs-and-exports.zip") as archive:
        expected_members = {f"{r['id']}/{name}" for r in rows for name in r["artifacts"]}
        assert set(archive.namelist()) == expected_members
        assert len(archive.namelist()) == len(rows) * 4
        assert len({r["id"] for r in rows}) == len(rows)
        for row in rows:
            case = row["id"]
            config = row["config"]
            canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            assert sha256(canonical).hexdigest() == row["cache_key"]
            assert config["comparison_arm"] == "current" and config["arm"] == stage
            assert config["comparison_sha256"] == row["base"]["pack_sha256"]
            files = {name: archive.read(f"{case}/{name}") for name in row["artifacts"]}
            assert {n: sha256(data).hexdigest() for n, data in files.items()} == row["artifacts"]
            handoff = json.loads(gzip.decompress(files["input.json.gz"]))
            assert (handoff["id"], handoff["arm"]) == (case, stage)
            score = json.loads(files["score.json"])
            assert score["pack_sha256"] == row["output"]["pack_sha256"] == row["artifacts"]["reconstruction.zip"]
            assert score["source_sha256"] == config["source_sha256"]
            assert score["reference_sha256"] == config["references"]
            blocks = decoded_pack(BytesIO(files["reconstruction.zip"]))
            coordinates = json.loads(files["placements.json"])
            assert blocks == {tuple(xyz): material for xyz, material in coordinates["local"]}
            fixed = score["score"]["fixed"]
            measured = fixed or score["score"]["relaxed_yaw"]
            for family in ("geometry", "scores"):
                for mode in ("strict", "facade"):
                    metric = f"{family}_{mode}"
                    counts = {k: measured[family][mode][k] for k in ("correct", "missed", "extras")}
                    if blocks and not fixed:
                        counts = dict(correct=0, missed=counts["correct"] + counts["missed"], extras=len(blocks))
                    assert all(row["output"]["metrics"][metric][k] == v for k, v in counts.items())
                    for scope in ("all", row["split"]):
                        for arm in ("base", "output"):
                            key = (scope, arm, metric)
                            totals.setdefault(key, Counter()).update({k: row[arm]["metrics"][metric][k] for k in counts})
    for (scope, arm, metric), counts in totals.items():
        c, m, e = (counts[k] for k in ("correct", "missed", "extras"))
        assert summary[scope][arm][metric] == dict(counts, f1=2*c/max(1, 2*c+m+e))
    return {scope: dict(cases=sum(scope in ("all", row["split"]) for row in rows),
                        metrics=stats["output"]) for scope, stats in summary.items()}


if __name__ == "__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--directory", type=Path, default=REVIEW)
    cli.add_argument("--stage", choices=STAGES)
    args = cli.parse_args()
    print(json.dumps({s: verify(s, args.directory) for s in ([args.stage] if args.stage else STAGES)}, indent=2))
