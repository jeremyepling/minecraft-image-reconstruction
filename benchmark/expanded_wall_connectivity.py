"""Measure reference-only join coverage on the published floating evaluation frames."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.expanded_baseline import SOURCE
from benchmark.relative_oracle_charts import ENDS, UNIT, owner_edge_samples, unit_projection


def largest(nodes, links, wall_cells):
    graph = defaultdict(set)
    for a, b in links:
        graph[a].add(b)
        graph[b].add(a)
    pending = set(nodes)
    best = set()
    while pending:
        queue = [pending.pop()]
        cells = set()
        for node in queue:
            cells.update(wall_cells.get(node, ()))
            neighbors = graph[node] & pending
            pending.difference_update(neighbors)
            queue.extend(neighbors)
        if len(cells) > len(best):
            best = cells
    return best


def measure(case):
    frame = Path(case["frame"])
    labels = json.loads((frame / "labels.json").read_text())
    with np.load(frame / "labels.npz") as arrays:
        owners = arrays["face_id"]
    counts = np.bincount(owners.ravel().astype(int))
    blocks = {b["id"]: b for b in labels["blocks"]}
    targets, nodes = set(), set()
    wall_cells, edges = defaultdict(set), defaultdict(list)
    for face in labels["faces"]:
        assert counts[face["id"]] == face["visible_pixels"] > 0
        xyz = tuple(face["xyz_world"])
        targets.add(xyz)
        if blocks[face["block"]].get("geometry_kind", "full_cube") != "full_cube":
            continue
        sid = face["surface_id"]
        nodes.add(sid)
        normal = np.asarray(face["normal_world"])
        axis = int(np.argmax(abs(normal)))
        if axis != 1:
            wall_cells[sid].add(xyz)
        axes = tuple(i for i in range(3) if i != axis)
        px = unit_projection(face, axes)
        samples = owner_edge_samples(px, face["id"], owners)
        corners = np.asarray(xyz) + np.maximum(0, normal) + UNIT @ np.eye(3)[list(axes)]
        for (a, b), bits in zip(ENDS, samples):
            if bits:
                edges[tuple(corners[a]), tuple(corners[b])].append(
                    (sid, axis, bits, np.linalg.norm(px[a] - px[b])))
    lengths = defaultdict(list)
    for candidates in edges.values():
        for (a, ax, abits, length), (b, bx, bbits, _) in combinations(candidates, 2):
            if ax != bx and abits & bbits:
                lengths[tuple(sorted((a, b)))].append(length)
    links = set(lengths)
    stronger = [pair for pair, values in lengths.items() if sum(d >= 4 for d in values) >= 2]
    wall_links = [(a,b) for a,b in links if a in wall_cells and b in wall_cells]
    direct = largest(wall_cells, wall_links, wall_cells)
    bridged = largest(nodes, links, wall_cells)
    if case["id"] == "test-walled-garden-2-warm-low-left":
        source = Image.open(frame / "rgb.png").convert("RGB")
        sheet = Image.new("RGB", (1200, 264), "#162130")
        pen = ImageDraw.Draw(sheet)
        panels = [(None, "Source image"), (direct, f"Wall joins   |   {len(direct)} blocks"),
                  (bridged, f"Joins via tops and undersides   |   {len(bridged)} blocks")]
        for i, (selected, title) in enumerate(panels):
            panel = source.copy()
            if selected is not None:
                ids = [f["id"] for f in labels["faces"] if f["surface_id"] in wall_cells
                       and tuple(f["xyz_world"]) in selected]
                mask = Image.fromarray(np.isin(owners, ids).astype("uint8") * 155)
                panel.paste(Image.new("RGB", panel.size, "#2aff8d"), mask=mask)
            panel.thumbnail((396, 228), Image.Resampling.LANCZOS)
            sheet.paste(panel, (i*400, 30))
            pen.text((i*400+8, 8), title, fill="white")
        dest = Path(__file__).parent / "reviews/expanded-wall-connectivity-01"
        dest.mkdir(parents=True, exist_ok=True)
        sheet.crop((0, 0, 1200, 30 + panel.height)).save(dest / "garden-wall-joins.png")
    return dict(id=case["id"], dataset=case["dataset"], targets=len(targets),
                direct=len(direct),
                via_horizontal=len(bridged),
                two_long_edges=len(largest(nodes, stronger, wall_cells)),
                all_wall=len(set().union(*wall_cells.values())))


if __name__ == "__main__":
    cohorts = ("visible-builds-v2", "visible-builds-v3", "decorative-builds-v1")
    cases = []
    for cohort in cohorts:
        base = SOURCE / cohort
        for row in json.loads((base / "manifest.json").read_text())["frames"]:
            if row["split"] == "test":
                cases.append(dict(row, dataset=cohort, frame=str(base / "data/test/frames" / row["id"])))
    assert len(cases) == 38
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(measure, cases))
    assert all(r["direct"] <= r["via_horizontal"] <= r["all_wall"] <= r["targets"] for r in rows)
    dest = Path(__file__).parent / "reviews/expanded-wall-connectivity-01"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "coverage.json").write_text(json.dumps(rows, indent=2) + "\n")
    for cohort in (*cohorts, "all"):
        group = [r for r in rows if cohort == "all" or r["dataset"] == cohort]
        print(cohort, {k: sum(r[k] for r in group) for k in ("targets","direct","via_horizontal","all_wall")})
