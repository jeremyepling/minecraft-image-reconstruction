"""Verify one observed atlas tile before extending its wall chart across the surface."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from benchmark.expanded_baseline import ROOT, digest, read, save
from benchmark.expanded_wall_source_resample import TEXELS
from benchmark import floating_patch_combined
from benchmark.floating_bridges_report import summarize
from benchmark.floating_patch_controls import WORKTREES, inputs, paths
from benchmark.floating_patch_report import COHORTS, baseline, collect
from benchmark.full_block_lattice import atlas
from engine.atlas_count import _shortlist
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

OUT = ROOT / "runs/face-verifier"
REVIEW = ROOT / "benchmark/reviews/floating-face-verifier-01"
GROUPS = (*COHORTS, "holdout2")


def panel(group):
    if group == "holdout3":
        root = WORKTREES/"floating-face-audit/runs/floating-holdout3"
        return root/"ordinary", root/"ordinary/atlas/connected", root/"source_material/charts", read(root/"manifest.json")["cases"]
    return inputs(group)


def placement_scores(group):
    if group == "holdout3":
        root = panel(group)[0].parent
        return summarize(panel(group)[3], {"horizontal":root/"horizontal-cells/assembly",
                         "verified":OUT/group/"assembly"}, "connected-atlas/sky")
    return collect(group,{"horizontal":baseline(group,True),"verified":OUT/group/"assembly"})


def correlation(patterns, observed):
    patterns = patterns - patterns.mean(axis=-1, keepdims=True)
    observed = observed - observed.mean()
    return (patterns * observed).sum(axis=-1) / np.maximum(
        1e-6, np.linalg.norm(patterns, axis=-1) * np.linalg.norm(observed))


def sample(h, origin, scale, rgb, mask):
    coords = np.floor(project(h, (TEXELS - .5) * scale + .5 + origin)).astype(int)
    x, y = coords[..., 0], coords[..., 1]
    height, width = mask.shape
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x.clip(0, width-1), y.clip(0, height-1)
    # Split by native pixel so minification cannot copy a training sample into validation.
    train = ((3*x + 5*y + (x // 3)) % 7) < 4
    return rgb[y, x], valid & mask[y, x], train, y*width+x


def verify(h, cell, rgb, mask):
    origin = np.array([cell["u"], cell["v"]])
    values, valid, train, pixels = sample(h, origin, 1., rgb, mask)
    fit = valid & train
    test = valid & ~train
    coverage = float(valid.mean())
    record = dict(u=cell["u"], v=cell["v"], coverage=coverage, accept=False)
    if coverage < .84 or min(len(np.unique(pixels[m])) for m in (fit, test)) < 16:
        return record
    gray = values.mean(-1)
    contrast = float(gray[test].std())
    if contrast < 2:
        return dict(record, contrast=contrast)
    library = atlas()
    indices, penalties = _shortlist(values, fit, library)
    patterns = library.patterns[indices]
    scores = correlation(patterns[:,:,fit], gray[fit]) - penalties[indices,None]/45
    material, orientation = np.unravel_index(scores.argmax(), scores.shape)
    template = patterns[material,orientation]
    heldout = float(correlation(template[test], gray[test]))
    # Recheck the chosen boundaries against compressed and expanded tiles at the same center.
    alternatives = []
    for scale in (.75, 1.25):
        alt, supported, old_train, _ = sample(h, origin, scale, rgb, mask)
        support = supported & ~old_train
        if support.sum() >= 32:
            alternatives.append(float(correlation(template[support], alt.mean(-1)[support])))
    boundary_margin = heldout - max(alternatives, default=1.)
    neighbor = -1.
    for step in ((1,0),(-1,0),(0,1),(0,-1)):
        adj, supported, _, adj_pixels = sample(h, origin+step, 1., rgb, mask)
        # Adjacent witness must contribute genuinely distinct source pixels.
        supported &= ~np.isin(adj_pixels, pixels[valid])
        if supported.mean() >= .84 and len(np.unique(adj_pixels[supported])) >= 32:
            neighbor = max(neighbor, float(correlation(template[supported], adj.mean(-1)[supported])))
    qualified = heldout >= .46 and boundary_margin >= .07 and (neighbor >= .29 or heldout >= .68)
    return dict(record, accept=qualified, contrast=contrast, heldout=heldout,
                boundary_margin=boundary_margin, neighbor=neighbor, texture=library.labels[indices[material]],
                train_score=float(scores.max()))


def build(job):
    group, case, source, ordinary, whole = job
    cv2.setNumThreads(1)
    name = case["id"]
    rel = Path("sky")/name/"charts.json"
    image = Path(case["variants"]["sky"]["image"])
    rgb = np.asarray(Image.open(image).convert("RGB"))
    sha = digest(image)
    charts = read(whole/rel)
    packet = read(ordinary/rel)
    handoff = source/"sky"/name/"source-handoff"
    for item in (charts, packet):
        paths.check_handoff(item, handoff, sha)
    planes = {p["id"]:p for p in read(handoff/"surfaces.json")["planes"]}
    by_id = {c["id"]:c for c in packet["charts"]}
    result, evidence = [], []
    for chart in charts["charts"]:
        seed = by_id[chart["id"]]
        if not seed["cells"] or abs(seed["normal_camera"][1]) >= .67:
            result.append(chart)
            continue
        assert seed["H"] == chart["H"]
        alpha, (x,y,x1,y1) = _native_alpha(planes[chart["id"]]["support"],rgb.shape[1::-1])
        mask = np.zeros(rgb.shape[:2], bool)
        mask[y:y1,x:x1] = np.asarray(alpha)>0
        rows = [verify(np.asarray(seed["H"]), c, rgb, mask) for c in seed["cells"]]
        accepted = any(r["accept"] for r in rows)
        evidence.append(dict(id=chart["id"],seeds=rows,extend=accepted))
        # Use the full original ordinary chart on abstention, including confidence and material IDs.
        result.append(chart if accepted else seed)
    dest = OUT/group
    save(dest/"charts"/rel,dict(charts,charts=result))
    save(dest/"evidence"/(name+".json"),dict(id=name,source_sha256=sha,ordinary_sha256=digest(ordinary/rel),
                                              expanded_sha256=digest(whole/rel),charts=evidence))
    with patch.object(floating_patch_combined,"inputs",panel):
        floating_patch_combined.run((group,case,dest/"charts",dest/"assembly"))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("groups",nargs="*",default=GROUPS)
    args=p.parse_args()
    jobs=[]
    for g in args.groups:
        source, ordinary, whole, cases=panel(g)
        jobs.extend((g,c,source,ordinary,whole) for c in cases)
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(build,jobs))
    save(REVIEW/("placements-"+"-".join(args.groups)+".json.gz"),
         {g:placement_scores(g) for g in args.groups})

