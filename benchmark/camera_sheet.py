"""Render the Stage 1 review sheet from frozen predictions and evaluation labels."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.camera_metrics import observation, project_visible, registration
from engine.geometry import project


PAPER, INK, MUTED, RULE = "#f8f8f4", "#182429", "#4b6064", "#bac8c7"
COLORS = {"truth": "#ffffff", "pavement": "#ff68b2", "seam-fit": "#ffc444", "tile-lines": "#40e9ee"}
NAMES = {"pavement": "Pavement", "seam-fit": "Seam fit", "tile-lines": "Tile lines"}
PAGE = 1760
MARGIN, COL, GUTTER = 40, 405, 20


@lru_cache(maxsize=None)
def font(size, bold=False):
    for name in (("segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def label(draw, xy, message, size=20, fill=INK, bold=False):
    draw.text(xy, message, font=font(size, bold), fill=fill, anchor="lt")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def scene(fixture, out, frame, methods):
    with Image.open(fixture / frame["image"]) as opened:
        source = opened.convert("RGB")
    faces = read(fixture / frame["labels"])["faces"]
    pose = read(fixture / frame["pose"])
    vertices, _, _ = observation(faces)
    edges = set()
    for face in faces:
        if face["visible_fraction"] < .98:
            continue
        q, v = face["world_corners"], face["corner_visible"]
        for i, a in enumerate(q):
            j = (i + 1) % 4
            if v[i] and v[j]:
                edges.add(tuple(sorted((tuple(a), tuple(q[j])))))
    keys = sorted(vertices)
    index = {p: i for i, p in enumerate(keys)}
    actual = np.array([vertices[p] for p in keys])
    projections = {"truth": actual}
    predictions = {}
    for method in methods:
        pred = read(out / "predictions" / method / (frame["id"] + ".json"))
        predictions[method] = pred
        if pred["P"] is None:
            projections[method] = None
        else:
            P = np.asarray(pred["P"])
            projections[method] = project_visible(P @ np.linalg.inv(registration(P, pose)), keys)
    return dict(source=source, points=projections, predictions=predictions, actual=actual,
                edges=sorted((index[a], index[b]) for a, b in edges))


def crop_box(center, width, height, image_size):
    w, h = image_size
    width, height = min(w, int(np.ceil(width))), min(h, int(np.ceil(height)))
    x = int(np.clip(round(center[0] - width/2), 0, w-width))
    y = int(np.clip(round(center[1] - height/2), 0, h-height))
    return (x, y, x+width, y+height)


def overview_box(sc, ratio):
    pts = sc["actual"]
    lo, hi = pts.min(0), pts.max(0)
    extent = hi - lo + 52
    width = max(extent[0], extent[1] * ratio)
    return crop_box((lo+hi)/2, width, width/ratio, sc["source"].size)


def reference_vertex(sc, methods):
    # Highlight the greatest baseline displacement, or the largest returned one
    # when pavement abstains. Exclude vertices without any drawable visible edge.
    ids = np.unique(sc["edges"])
    if not len(ids):
        ids = np.arange(len(sc["actual"]))
    for method in methods:
        points = sc["points"][method]
        if points is None:
            continue
        errors = np.linalg.norm(sc["actual"][ids] - points[ids], axis=1)
        good = np.isfinite(errors)
        if good.any():
            return int(ids[good][np.argmax(errors[good])])
    return int(ids[len(ids)//2])


def line_on(draw, points, color, width):
    if not np.isfinite(points).all() or np.max(np.abs(points)) > 1e6:
        return
    xy = [tuple(p) for p in points]
    draw.line(xy, fill="#202020", width=width+2)
    draw.line(xy, fill=color, width=width)


def proposal_lines(pred):
    if not pred or pred["P"] is None:
        return
    for plane in pred["planes"]:
        P = np.array(pred["P"])
        free = plane["axes"]
        low, high = plane["low"], plane["high"]
        for axis in (0, 1):
            other = 1-axis
            for k in range(low[axis], high[axis]+1):
                xyz = np.full((2, 3), plane["offset"])
                xyz[:, free[axis]] = k
                xyz[:, free[other]] = [low[other], high[other]]
                if np.all(xyz @ P[2, :3] + P[2, 3] > 1e-6):
                    yield project(P, xyz), k in (low[axis], high[axis])


def panel(sc, box, size, method, mode="transfer", focus=None):
    # Preserve square source pixels even if the source clips an overview crop.
    region = sc["source"].crop(box)
    scale = min(size[0] / region.width, size[1] / region.height)
    fitted = tuple(round(v*scale) for v in region.size)
    pic = Image.new("RGB", size, "#e4e9e5")
    inset = np.array([(n-v)//2 for n, v in zip(size, fitted)])
    pic.paste(region.resize(fitted, Image.Resampling.NEAREST if scale > 1 else Image.Resampling.LANCZOS), tuple(inset))
    conv = lambda xy: (np.asarray(xy) - box[:2]) * scale + inset
    draw = ImageDraw.Draw(pic)
    if mode == "overview":
        pass
    elif mode == "auto":
        for xy, boundary in proposal_lines(sc["predictions"][method]):
            line_on(draw, conv(xy), COLORS[method], 3 if boundary else 1)
    elif mode == "combined":
        # Interleave dashes so coincident grid lines from different fits are all legible.
        for j, (name, points) in enumerate(sc["points"].items()):
            if name == "truth" or points is None:
                continue
            screen = conv(points)
            for a, b in sc["edges"]:
                pair = screen[[a, b]]
                if (not np.isfinite(pair).all() or np.any(pair.max(0) < 0)
                    or np.any(pair.min(0) > size)):
                    continue
                delta = pair[1] - pair[0]
                length = np.linalg.norm(delta)
                if length < 1:
                    continue
                for start in np.arange(6*(j-1), length, 18):
                    ends = np.array([start, min(start+5, length)])[:, None]
                    line_on(draw, pair[0] + ends*delta/length, COLORS[name], 2)
        for name, points in sc["points"].items():
            if points is None or not np.isfinite(points[focus]).all():
                continue
            x, y = conv(points[focus])
            r = {"truth": 13, "pavement": 9, "seam-fit": 6, "tile-lines": 3}[name]
            draw.ellipse((x-r, y-r, x+r, y+r), outline="#202020", width=4)
            draw.ellipse((x-r, y-r, x+r, y+r), outline=COLORS[name], width=2)
    else:
        points = sc["points"][method]
        if points is not None:
            points = conv(points)
            for a, b in sc["edges"]:
                line_on(draw, points[[a, b]], COLORS[method], 2 if scale > 1 else 1)
    if isinstance(focus, tuple):
        mark = conv(np.array(focus).reshape(2, 2))
        draw.rectangle([tuple(v) for v in mark], outline="#161f24", width=4)
        draw.rectangle([tuple(v) for v in mark], outline="#ffffff", width=2)
    elif focus is not None and mode == "transfer":
        x, y = conv(sc["actual"][focus])
        for p, q in (((x-7, y), (x+7, y)), ((x, y-7), (x, y+7))):
            line_on(draw, np.array([p, q]), "#ffffff", 2)
    return pic


def sheet(fixture: Path, out: Path, run_id: str, frames: list, rows: list, summary: dict, winner: str):
    methods = tuple(summary)
    by_id = {f["id"]: f for f in frames}
    rr = {(r["id"], r["method"]): r for r in rows}
    normal = sorted({r["id"] for r in rows if r["condition"] == "ground-visible"})
    baseline = "pavement"
    registered = [i for i in normal if rr[i, baseline]["mean_px"] is not None]
    primary = max(registered or normal, key=lambda i: rr[i, baseline]["mean_px"] or 0.)
    others = [i for i in normal if i != primary] or normal
    secondary = max(others, key=lambda i: (not rr[i, baseline]["metric"], rr[i, winner]["metric"],
                                            by_id[i]["layout"] != by_id[primary]["layout"],
                                            rr[i, winner]["hit_rate"]))
    controls = [i for i in by_id if (i, winner) in rr and by_id[i]["condition"] == "limited-ground"]
    control = max(controls, key=lambda i: (by_id[i]["palette"] == "plaster", i))
    a, b, c = (scene(fixture, out, by_id[i], methods) for i in (primary, secondary, control))
    detail = reference_vertex(a, methods)
    crop = crop_box(a["actual"][detail], 80, 44, a["source"].size)
    size = (COL, 216)
    abox, bbox = overview_box(a, COL/216), overview_box(b, COL/216)
    regions = []
    def region(frame_id, name, coords):
        regions.append(dict(image=frame_id, stage="camera", name=name, box=list(coords)))
    region(primary, "registered-grid", abox)
    region(primary, "corner-detail", crop)
    region(secondary, "automatic-plane-proposals", bbox)
    cbox = (0, 0, *c["source"].size)
    region(control, "limited-ground", cbox)

    canvas = Image.new("RGB", (PAGE, 1690), PAPER)
    d = ImageDraw.Draw(canvas)
    xs = [MARGIN + j*(COL+GUTTER) for j in range(4)]
    def rule(y):
        d.line((MARGIN, y, PAGE-MARGIN, y), fill=RULE, width=2)
    def marker(x, y, name):
        d.rounded_rectangle((x, y+3, x+17, y+20), radius=4, fill=COLORS[name], outline="#46595b")
    def method_header(j, y, name, title):
        marker(xs[j], y, name)
        label(d, (xs[j]+29, y), title, 23, bold=True)
    def num(v, places=2):
        return "—" if v is None else f"{v:.{places}f}"

    split = by_id[primary]["split"].capitalize()
    label(d, (MARGIN, 28), f"{split.upper()} REVIEW     /     JAVA 26.3", 19, MUTED, True)
    label(d, (MARGIN, 66), "Camera and block grid", 50, bold=True)
    label(d, (MARGIN, 132), f"{run_id}    ·    Development selection: {NAMES[winner]}    ·    {summary[winner]['layouts']} layouts, {summary[winner]['images']} single-image predictions per method", 22, MUTED)
    headers = ((40, "Method"), (345, "Corners ≤ 2 px"), (635, "Metric frames"), (890, "Mean error¹"), (1145, "Span MAE¹"), (1400, "Exact planes²"))
    d.rectangle((MARGIN, 179, PAGE-MARGIN, 218), fill="#e6ede9")
    for x, txt in headers:
        label(d, (x+8, 187), txt, 20, MUTED, True)
    for j, m in enumerate(methods):
        s, y = summary[m], 231+j*35
        marker(48, y-1, m)
        label(d, (76, y), NAMES[m], 21, bold=True)
        vals = (f"{100*s['layout_hit_rate']:.1f}%", f"{s['metric_images']} / {s['images']}",
                f"{num(s['layout_mean_px'])} px", f"{num(s['layout_span_mae_blocks'],3)} blocks",
                f"{s['exact_chart_planes']} / {s['charts']}")
        for ((x, _), val) in zip(headers[1:], vals):
            label(d, (x+8, y), val, 21)
    label(d, (MARGIN, 348), "¹ Errors average layouts on returned estimates. Missing projections count as misses in the 2 px rate.     ² Exact depth of the proposed wall plane / eligible wall planes.", 18, MUTED)
    rule(387)
    label(d, (MARGIN, 411), "Grid alignment on known surfaces", 32, bold=True)
    label(d, (MARGIN, 456), f"{primary}    ·    Largest mean baseline drift among returned ground views. White box marks the enlarged corner.", 20, MUTED)
    for j, m in enumerate(("truth", *methods)):
        method_header(j, 498, m, "Recorded grid" if j == 0 else NAMES[m])
        canvas.paste(panel(a, abox, size, m, focus=crop), (xs[j], 534))
    for j, m in enumerate(("truth", *methods)):
        y = 765
        if m == "truth":
            label(d, (xs[j], y), "Same corner, all methods", 21, bold=True)
        else:
            p = a["points"][m]
            delta = np.linalg.norm(p[detail]-a["actual"][detail]) if p is not None else None
            label(d, (xs[j], y), "Scale uncertain" if delta is None else f"Corner error   {delta:.2f} px", 21, bold=True)
        pic = panel(a, crop, (COL, 223), m, mode="combined" if j == 0 else "transfer", focus=detail)
        canvas.paste(pic, (xs[j], 800))
    label(d, (MARGIN, 1038), f"5× source pixels   ·   White ring or cross = recorded corner. Colored rings = predicted centers. Lines project the same visible 3D edges through each estimated camera.", 19, MUTED)
    rule(1080)
    label(d, (MARGIN, 1102), "Automatic wall planes", 32, bold=True)
    label(d, (MARGIN, 1145), f"{secondary}    ·    The left panel uses the known surfaces. Colored rectangles show the planes each method proposed from the screenshot.", 20, MUTED)
    for j, m in enumerate(("truth", *methods)):
        if m == "truth":
            title = "Recorded visible edges"
        else:
            n = len(b["predictions"][m]["planes"])
            title = f"{NAMES[m]}   ·   {n} plane" + ("" if n == 1 else "s")
        method_header(j, 1185, m, title)
        pic = panel(b, bbox, size, m, mode="transfer" if j == 0 else "auto")
        canvas.paste(pic, (xs[j], 1220))
    rule(1461)
    label(d, (MARGIN, 1482), "Facade-only scale control", 30, bold=True)
    label(d, (MARGIN, 1525), control, 20, MUTED)
    canvas.paste(panel(c, cbox, (270, 111), "truth", mode="overview"), (MARGIN, 1560))
    metric = [NAMES[m] for m in methods if rr[control, m]["metric"]]
    label(d, (337, 1565), f"{len(metric)} of {len(methods)} methods returned absolute block scale.", 24, bold=True)
    label(d, (337, 1606), "This native frame contains only the facade. A near-uniform surface provides little evidence for the one-block period.", 21, MUTED)
    canvas.save(out / "comparison.png")
    return regions
