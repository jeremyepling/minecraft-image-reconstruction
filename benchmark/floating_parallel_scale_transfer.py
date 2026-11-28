"""Replay the frozen parallel wall rule on the active floating conditions."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark import floating_parallel_scale as rule
from benchmark.expanded_baseline import ROOT, WORKTREES, digest, read, save
from benchmark.floating_bridges_report import read_arm
from benchmark.floating_extra_budget import LOCATION, combined, paths
from benchmark.relative_face_scoring import _yaw
from benchmark.strip_visibility_audit import horizontal_slivers
from benchmark.typed_structures import geometry_score, score as typed_score


ARCHIVE = WORKTREES / "floating-depth-refit/benchmark/reviews/floating-depth-refit-01/conditions.json.gz"
LARGE_CONTRACTION = WORKTREES / "floating-extra-isolation/runs/extra-isolation"
OUT = ROOT / "runs/parallel-scale-broad"


def source_proposals(packet, planes, rgb):
    texture_score = rule.texture_score
    source_charts = packet["charts"]
    brick_matches = {chart["id"]: rule.matches(chart, "bricks") for chart in source_charts}
    informed = [c for c in source_charts if brick_matches[c["id"]] >= 4]
    context = []
    eligible = []
    changes = []
    charts = []
    for chart in source_charts:
        # Record the brick evidence prerequisites, independently of the frozen gate.
        h = np.asarray(chart["H"])
        neighbors = [c for c in informed if c is not chart]
        corners = [c["id"] for c in neighbors if
                   np.allclose(np.asarray(c["H"])[:, 1:] / c["H"][2][2], h[:, 1:] / h[2, 2], atol=1.e-5)]
        donors = [c["id"] for c in neighbors if brick_matches[c["id"]] >= .4 * len(c["cells"]) and
                  np.dot(c["normal_camera"], chart["normal_camera"]) > .99]
        if corners and donors and brick_matches[chart["id"]] < .4 * len(chart["cells"]):
            context.append(dict(chart=chart["id"], corners=corners, donors=donors))
        correlations = []

        def measured_texture(*args):
            value = texture_score(*args)
            correlations.append(value)
            return value

        with patch.object(rule, "texture_score", measured_texture):
            result = rule.proposals(chart, source_charts, planes[chart["id"]], rgb)
        if correlations:
            eligible.append(dict(chart=chart["id"], scores=correlations))
        if result is not None:
            chart, change = result
            changes.append(change)
        charts.append(chart)
    assert {c["chart"] for c in eligible} <= {c["chart"] for c in context}
    return charts, brick_matches, context, eligible, changes


def verify_export(folder, chart_hash, source_hash, reference_hashes, truth, optional):
    receipt = read(folder / "score.json")
    score, aligned = read_arm(folder)
    assert receipt["chart_sha256"] == chart_hash
    assert receipt["source_sha256"] == source_hash
    assert receipt["reference_sha256"] == reference_hashes
    assert receipt["pack_sha256"] == digest(folder / "reconstruction.zip")
    yaw, shift = score["yaw_quarters"], score["translation"]
    local = read(folder / "placements.json")["local"]
    assert aligned == {tuple(a + b for a, b in zip(_yaw(p, yaw), shift)): m for p, m in local}
    correct = aligned.keys() & truth.keys()
    exact = {p for p in correct if aligned[p] == truth[p]}
    extras = aligned.keys() - truth.keys()
    assert score["geometry"] == geometry_score(aligned, truth, optional)
    typed = typed_score(aligned, truth, optional, set())
    assert all(score["scores"][k] == typed[k] for k in ("strict", "facade", "material"))
    return score, dict(correct=correct, facade_correct=correct - optional,
                       exact=exact, facade_exact=exact - optional, extra=extras), receipt["pack_sha256"]


def run(job):
    archived, case, reviewed = job
    cv2.setNumThreads(1)
    panel, name = archived["panel"], case["id"]
    inputs = paths(panel)[0]
    baseline = rule.BASE / panel if reviewed else LARGE_CONTRACTION / panel / "large-contraction"
    rel = Path("sky") / name / "charts.json"
    chartpath = baseline / "charts" / rel
    packet = read(chartpath)
    source = Path(case["variants"]["sky"]["image"])
    source_hash = digest(source)
    assert source_hash == case["variants"]["sky"]["source_sha256"] == packet["source_sha256"] == archived["source_sha256"]
    surfaces = inputs[0] / "sky" / name / "source-handoff/surfaces.json"
    planes = {p["id"]: p for p in read(surfaces)["planes"]}
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    charts, brick_matches, context, eligible, changes = source_proposals(packet, planes, rgb)
    dest = OUT / ("reviewed" if reviewed else "transfer") / panel
    previous_chart = digest(chartpath)
    previous_folder = baseline / "assembly" / LOCATION / name
    candidate_folder = previous_folder
    candidate_chart = previous_chart
    if changes:
        updated = dest / "charts" / rel
        save(updated, dict(packet, charts=charts))
        with patch.object(combined, "inputs", lambda _: inputs):
            combined.run((panel, case, dest / "charts", dest / "assembly"))
        candidate_folder = dest / "assembly" / LOCATION / name
        candidate_chart = digest(updated)

    frame = Path(case["frame"])
    reference_hashes = {p: digest(frame / p) for p in ("camera.json", "labels.json", "labels.npz")}
    labels = read(frame / "labels.json")
    truth = {tuple(b["xyz_world"]): b["block_id"] for b in labels["blocks"]}
    with np.load(frame / "labels.npz", allow_pickle=False) as masks:
        optional = horizontal_slivers(labels, masks["face_id"])
    before, old, previous_pack = verify_export(previous_folder, previous_chart, source_hash,
                                                reference_hashes, truth, optional)
    if changes:
        after, new, candidate_pack = verify_export(candidate_folder, candidate_chart, source_hash,
                                                   reference_hashes, truth, optional)
    else:
        after, new, candidate_pack = before, old, previous_pack
    coordinates = {kind: dict(gained=sorted(new[kind] - old[kind]), lost=sorted(old[kind] - new[kind]))
                   for kind in old}
    paired = {kind: {direction: len(points) for direction, points in moves.items()}
              for kind, moves in coordinates.items()}
    row = dict(id=name, panel=panel, layout=archived["layout"], split=case["split"], dataset=case["dataset"],
               reviewed=reviewed, source_sha256=source_hash, surfaces_sha256=digest(surfaces),
               reference_sha256=reference_hashes, charts_seen=len(charts), brick_matches=brick_matches,
               context=context, eligible=eligible, changed=changes,
               baseline_chart_sha256=previous_chart, candidate_chart_sha256=candidate_chart,
               baseline_pack_sha256=previous_pack, candidate_pack_sha256=candidate_pack,
               baseline=before, candidate=after, paired=paired, coordinates=coordinates)
    save(dest / "cases" / (name + ".json.gz"), row)
    return row


def totals(rows):
    result = dict(conditions=len(rows), layouts=len({r["layout"] for r in rows}),
                  charts_seen=sum(r["charts_seen"] for r in rows))
    for key in ("context", "eligible", "changed"):
        selected = [r for r in rows if r[key]]
        result[key] = dict(conditions=len(selected), charts=sum(len(r[key]) for r in selected),
                           layouts=sorted({r["layout"] for r in selected}))
    result["paired"] = {kind: {direction: sum(r["paired"][kind][direction] for r in rows)
                              for direction in ("gained", "lost")}
                        for kind in ("correct", "facade_correct", "exact", "facade_exact", "extra")}
    for arm in ("baseline", "candidate"):
        scores = {}
        for field in ("geometry", "scores"):
            for mode in ("strict", "facade"):
                counts = Counter()
                for row in rows:
                    counts.update({k: row[arm][field][mode][k] for k in ("correct", "missed", "extras")})
                correct, missed, extra = (counts[k] for k in ("correct", "missed", "extras"))
                scores[field + "_" + mode] = dict(**counts, f1=2 * correct / (2 * correct + missed + extra))
        result[arm] = scores
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    archived = [r for r in read(ARCHIVE) if r["active"]]
    assert len(archived) == 84 and sum(r["panel"] == "sky" for r in archived) == 6
    panels = {panel: {c["id"]: c for c in paths(panel)[0][3]} for panel in {r["panel"] for r in archived}}
    jobs = [(r, panels[r["panel"]][r["id"]], False) for r in archived]
    reviewed = [r for r in archived if r["panel"] == "original" and
                r["id"] in {"test-twinspire2-earth-high-right", "test-twinspire2-earth-low-left"}]
    assert len(reviewed) == 2
    jobs.extend((r, panels[r["panel"]][r["id"]], True) for r in reviewed)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for status in as_completed([pool.submit(run, job) for job in jobs]):
            row = status.result()
            rows.append(row)
            if row["eligible"] or len(rows) % 10 == 0:
                print(len(rows), row["panel"], row["id"], "reviewed" if row["reviewed"] else "transfer",
                      len(row["eligible"]), len(row["changed"]), row["paired"], flush=True)
    rows.sort(key=lambda r: (r["reviewed"], r["panel"], r["id"]))
    transfer = [r for r in rows if not r["reviewed"]]
    sky = [r for r in transfer if r["panel"] == "sky"]
    summary = dict(condition_archive_sha256=digest(ARCHIVE),
                   rule_sha256=digest(ROOT / "benchmark/floating_parallel_scale.py"),
                   active=totals(transfer), sky=totals(sky), reviewed=totals([r for r in rows if r["reviewed"]]),
                   datasets={key: totals([r for r in transfer if r["dataset"] == key])
                             for key in sorted({r["dataset"] for r in transfer})})
    save(OUT / "transfer-cases.json.gz", rows)
    save(OUT / "transfer-summary.json", summary)


if __name__ == "__main__":
    main()
