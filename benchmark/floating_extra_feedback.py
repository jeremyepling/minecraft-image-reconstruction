"""Build a portable feedback viewer from the paired source images and wall charts."""
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark.floating_extra_budget import paths, LOCATION, REVIEW, WORKTREES
from benchmark.expanded_baseline import read
from benchmark.floating_bridges_report import read_arm
from engine.geometry import project
from engine.surface_perimeter import _native_alpha

CASES = [
    ("holdout5", "train-amphitheater-01-orchid-low-left", "mask-0", "Amphitheater", "Two seed cells grew into 1,548 cells. What visible boundary or second block would you require before allowing that expansion?"),
    ("sky", "validation-hillside-chapel-1-warm-low-left", "mask-0", "Hillside Chapel", "Where is the first feature that tells you to stop extending the stone-brick ruler? Which region can share its spacing?"),
    ("holdout4", "train-riverside-mill-1-warm-high-right", "mask-0", "Riverside Mill, high right", "Which two visible edges or block corners establish the spacing you trust?"),
    ("holdout4", "train-riverside-mill-1-warm-low-left", "mask-0", "Riverside Mill, low left", "What would make you keep the original grid in this view of the same building?"),
    ("holdout3", "train-sawtooth2-cool-high-right", "mask-0-wall-1", "Sawtooth", "Which joints or changes in material confirm that the proposed grid is still aligned away from its starting point?"),
    ("original", "test-twinspire2-cool-low-left", "mask-2", "Twinspire", "The distant seams favored the replacement. Where does that evidence first become misleading, or does the local grid look right?"),
]


def picture(im):
    out = io.BytesIO()
    im.convert("RGB").save(out, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def grid(image, chart):
    copy = image.copy()
    pen = ImageDraw.Draw(copy)
    tile = np.array([[0,0], [1,0], [1,1], [0,1], [0,0]])
    for cell in chart["cells"]:
        corners = project(np.asarray(chart["H"]), tile + [cell["u"], cell["v"]])
        if np.isfinite(corners).all() and np.max(abs(corners)) < 100000:
            pen.line(list(map(tuple, corners)), fill="#00fff1", width=1)
    return copy


def case_data(panel, name, wall, title, question):
    (src, seeds, walls, cases), old, new = paths(panel)
    case = next(c for c in cases if c["id"] == name)
    rel = Path("sky") / name / "charts.json"
    def chart(folder):
        return next(c for c in read(folder / rel)["charts"] if c["id"] == wall)

    with Image.open(case["variants"]["sky"]["image"]) as im:
        source = im.convert("RGB")
    planes = read(src / "sky" / name / "source-handoff/surfaces.json")["planes"]
    plane = next(p for p in planes if p["id"] == wall)
    alpha, box = _native_alpha(plane["support"], source.size)
    x,y,r,b = box
    box = (max(0,x-10), max(0,y-10), min(source.width,r+10), min(source.height,b+10))
    mask = Image.new("L", source.size)
    mask.paste(alpha, (x,y))
    dim = Image.blend(source, Image.new("RGB",source.size), .76)
    isolated = Image.composite(source, dim, mask)
    context = source.copy()
    ImageDraw.Draw(context).rectangle(box, outline="gold", width=3)
    if panel == "holdout5":
        new = WORKTREES / "floating-extra-isolation/runs/extra-isolation/holdout5/sparse-mask-0/assembly"
        proposal = chart(seeds)
        alternative = "B · Original sparse atlas cells"
    else:
        proposal = chart(new.parent / "charts")
        alternative = "B · Proposed full-face ruler"
    output = new
    if title in {"Twinspire", "Sawtooth"}:
        # This is the measured seam-rule export, which can accept a subset of chart changes.
        audit = read(WORKTREES / "floating-distant-seams/benchmark/reviews/floating-distant-seams-01" / (panel+".json.gz"))
        selected = next(r for r in audit["rows"] if r["id"] == name)["score"]["distant"]
    else:
        selected = read_arm(output / LOCATION / name)[0]
    previous = read_arm(old / LOCATION / name)[0]
    with np.load(Path(case["frame"]) / "labels.npz") as labels:
        ids = labels["face_id"]
    edges = np.zeros(ids.shape, bool)
    edges[1:] |= ids[1:] != ids[:-1]
    edges[:,1:] |= ids[:,1:] != ids[:,:-1]
    pixels = np.array(source)
    pixels[edges & (ids != 0)] = (255, 206, 0)
    panels = [(label,picture(img.crop(box))) for label,img in [
        ("Source crop",source), ("Detected surface · surroundings dimmed",isolated),
        ("A · Previous expanded grid",grid(source,chart(walls))), (alternative,grid(source,proposal)),
        ("Reference · visible face boundaries",Image.fromarray(pixels))]]
    stats = []
    for score in (previous,selected):
        g = score["geometry"]["strict"]
        stats.append([g["correct"], g["extras"], score["scores"]["strict"]["correct"], round(100*g["f1"],2)])
    context.thumbnail((760,425))
    return dict(id=name,title=title,wall=wall,question=question,context=picture(context),panels=panels,stats=stats)


def main():
    cases = [case_data(*c) for c in CASES]
    template = Path(__file__).with_name("floating_extra_review_template.html").read_text(encoding="utf-8")
    (REVIEW/"review.html").write_text(template.replace("__CASES__",json.dumps(cases)),encoding="utf-8")


if __name__ == "__main__":
    main()
