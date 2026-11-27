"""Score oracle-edge connectivity with registration confined to this diagnostic."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from benchmark.relative_oracle_charts import ENDS, UNIT, dump, unit_projection


def fingerprint(vertices):
    return tuple(np.rint(vertices.ravel()*1000).astype(int))


def view(case, output, solver_checkout):
    native = json.loads((Path(case["frame"])/"labels.json").read_text())
    rows = json.loads((output/"sky"/case["id"]/"charts.json").read_text())["charts"]
    blocks = {b["id"]: b for b in native["blocks"]}
    lookup = {}
    for face in native["faces"]:
        if blocks[face["block"]].get("geometry_kind", "full_cube") != "full_cube":
            continue
        axes = tuple(i for i in range(3) if not face["normal_world"][i])
        key = fingerprint(unit_projection(face, axes))
        if key in lookup:
            raise ValueError("Nonunique measured corners")
        lookup[key] = face
    edge_owners, worlds, cells_by_chart = defaultdict(list), [], []
    for i, chart in enumerate(rows):
        h = np.array(chart["H"])
        u, v = (np.eye(3, dtype=int)[axis-1] for axis in chart["axes"])
        observed = {(c["u"], c["v"]): c for c in chart["cells"]}
        origins, owned = set(), set()
        for (x, y), c in observed.items():
            q = np.column_stack((UNIT + [x, y], np.ones(4))) @ h.T
            px = q[:, :2]/q[:, 2:]
            face = lookup[fingerprint(px)]
            block = np.array(face["xyz_world"])
            owned.add(tuple(block))
            start = block + np.maximum(0, face["normal_world"])
            origins.add(tuple(start - x*u - y*v))
            lattice = start + UNIT.astype(int) @ np.stack((u,v))
            neighbors = [(x, y-1), (x+1, y), (x,y+1), (x-1,y)]
            for edge, (a,b) in enumerate(ENDS):
                if neighbors[edge] in observed or not c["edge_samples"][edge]:
                    continue
                key = (tuple(lattice[a]), tuple(lattice[b]))
                edge_owners[key].append((i, c["edge_samples"][edge], float(np.linalg.norm(px[a]-px[b]))))
        if len(origins) != 1:
            raise ValueError("Chart is not a single physical plane")
        worlds.append(np.array(origins.pop()))
        cells_by_chart.append(owned)
    evidence = defaultdict(list)
    for candidates in edge_owners.values():
        for (i, b, length), (j, c, _) in combinations(candidates, 2):
            if i != j and abs(rows[i]["normal_axis"]) != abs(rows[j]["normal_axis"]) and b & c:
                evidence[tuple(sorted((i,j)))].append(length)
    def component_stats(min_edges, min_px, joins):
        graph = defaultdict(set)
        for i,j in joins:
            graph[i].add(j)
            graph[j].add(i)
        remaining = set(range(len(rows)))
        sizes = []
        while remaining:
            queue = [remaining.pop()]
            for i in queue:
                for j in graph[i] & remaining:
                    queue.append(j)
                    remaining.remove(j)
            sizes.append(len(set.union(*(cells_by_chart[k] for k in queue))))
        return dict(min_edges=min_edges, min_projected_px=min_px, components=len(sizes),
                    largest_true_blocks=max(sizes), joined_pairs=len(joins))
    controls = []
    for minimum, min_px in ((1,0), (1,4), (2,4)):
        links = [ij for ij, lengths in evidence.items() if sum(d>=min_px for d in lengths)>=minimum]
        controls.append(component_stats(minimum,min_px,links))
    solver = []
    if solver_checkout:
        import importlib.util
        spec = importlib.util.spec_from_file_location("audited_solver", solver_checkout/"benchmark"/"relative_face_assembly.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for minimum in (1,2):
            assembled = mod.assemble(rows, min_edges=minimum)
            joins = assembled["joins"]
            wrong = [e for e in joins if tuple(worlds[e["j"]]-worlds[e["i"]]) != tuple(e["shift"])]
            unsupported = [e for e in joins if sum(d >= 4 for d in evidence[tuple(sorted((e["i"],e["j"])))]) < minimum]
            solver.append(dict(min_edges=minimum, accepted=len(joins), wrong_physical_offset=len(wrong),
                               wrong_offsets=[dict(chart_ids=e["chart_ids"], edges=e["edges"],
                                                   offset_error_xyz=(e["shift"] - (worlds[e["j"]] - worlds[e["i"]])).tolist(),
                                                   mean_endpoint_error_px=e["error"]) for e in wrong],
                               insufficient_true_coincident_edges=len(unsupported),
                               cycle_conflicts=len(assembled["conflicts"]),
                               largest_emitted=len(assembled["groups"][0]["blocks"])))
    return dict(id=case["id"], controls=controls, solver=solver)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--solver-checkout", type=Path)
    args = parser.parse_args()
    from benchmark.relative_oracle_charts import digest
    report = [view(c, args.output, args.solver_checkout) for c in json.loads(args.fixtures.read_text())["cases"]]
    solver_digest = digest(args.solver_checkout/"benchmark"/"relative_face_assembly.py") if args.solver_checkout else None
    dump(args.output/"oracle-connectivity.json", dict(views=report, solver_sha256=solver_digest))
    for row in report:
        print(row["id"], [r["largest_true_blocks"] for r in row["controls"]], row["solver"])


if __name__ == "__main__":
    main()
