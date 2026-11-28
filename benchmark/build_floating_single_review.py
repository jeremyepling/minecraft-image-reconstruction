"""Package a protected-preview review from saved Expanded Split-Border evidence."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from benchmark.expanded_baseline import digest, read
from benchmark.floating_review_app import SINGLE_TEMPLATE, app_checkout
from benchmark.mask_blocks import decoded_pack
from benchmark.mobile_surface_evidence import add_grid, packet_grids, parent, source_surfaces


MEASURES = (
    ("geometry_strict", "Strict placement F1"),
    ("geometry_facade", "Facade placement F1"),
    ("scores_strict", "Strict position + material F1"),
    ("scores_facade", "Facade position + material F1"),
)


def run_evidence(folder, image, source_sha, label):
    handoff = folder / "source-handoff"
    masks = handoff / "surfaces.json"
    counts = handoff / "counts.json"
    mask_doc = read(masks)
    manifest = read(handoff / "build-manifest.json")
    if manifest["image_sha256"] != source_sha or mask_doc["image_size"] != list(image.size):
        raise ValueError(f"Surface handoff differs from the supplied source in {folder}")
    surfaces = source_surfaces(mask_doc["planes"], read(counts), image)
    proof = dict(sourceSha=source_sha, surfaceSha=digest(masks), counts=digest(counts))
    charts, _ = packet_grids(folder / "charts.json", proof, {})
    oriented = read(folder / "oriented-charts.json")["charts"]
    assembly = read(folder / "assembly.json")
    selected = set(assembly["selected"])
    blocks = decoded_pack(folder / "reconstruction.zip")
    if assembly["size"] != len(blocks) or not selected <= {chart["id"] for chart in oriented}:
        raise ValueError(f"Assembly receipt differs from the saved grid or pack in {folder}")
    for chart in charts:
        add_grid(surfaces[parent(chart["id"])], "Charts before assembly", chart)
    for chart in oriented:
        surface = surfaces[parent(chart["id"])]
        add_grid(surface, "Oriented charts", chart)
        if chart["id"] in selected:
            add_grid(surface, "Selected grids", chart, ["selected"])
    counted = sum(all(value is not None for value in surface["count"]["blocks"])
                  for surface in surfaces.values())
    stages = [len(surfaces), counted, len(charts), len(oriented), len(selected), len(assembly["joins"]), len(blocks)]
    return dict(label=label, stages=stages, surfaces=surfaces)


def scorecard(label, path):
    report = read(path)
    cohort = report["cohorts"]["Visible Builds"]
    scores = []
    for key, arm in cohort["scores"].items():
        strict = arm["geometry_strict"]
        scores.append(dict(name=report["names"][key],
                           f1=[arm[metric]["f1_percent"] for metric, _ in MEASURES],
                           counts=[strict[key] for key in ("correct", "missed", "extras")]))
    return dict(label=label, views=cohort["views"], scores=scores)


def prepare(args):
    with Image.open(args.source) as opened:
        image = opened.convert("RGB")
    source_sha = digest(args.source)
    arms = [run_evidence(args.baseline, image, source_sha, args.baseline_name)]
    if args.candidate:
        arms.append(run_evidence(args.candidate, image, source_sha, args.candidate_name))
    rendered = BytesIO()
    image.save(rendered, "WEBP", quality=91, method=5)
    blob = dict(title=args.title, revision=args.revision, credit=args.credit, size=image.size,
                image="data:image/webp;base64," + base64.b64encode(rendered.getvalue()).decode("ascii"),
                sourceSha256=source_sha, arms=arms, measures=[label for _, label in MEASURES],
                scorecards=[scorecard(label, Path(path)) for label, path in args.scorecard])
    score_blob = {key: blob[key] for key in ("title", "revision", "measures", "scorecards")}
    score_blob["arms"] = [{key: arm[key] for key in ("label", "stages")} for arm in arms]
    template = (app_checkout(args.app_root) / SINGLE_TEMPLATE).read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(path.name not in {"index.html", "scores.html", "vercel.json"} or not path.is_file()
           for path in args.out.iterdir()):
        raise ValueError("Choose a directory reserved for generated review files")
    for name, view, data in (("index.html", "review", blob), ("scores.html", "scores", score_blob)):
        encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", r"<\/")
        page = template.replace("%%VIEW%%", view).replace("%%EVIDENCE%%", encoded)
        (args.out / name).write_text(page, encoding="utf-8")
    config = dict(cleanUrls=True, rewrites=[dict(source="/review", destination="/index")],
                  headers=[dict(source="/(.*)", headers=[{"key": "X-Robots-Tag", "value": "noindex, nofollow"}])])
    (args.out / "vercel.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "baseline", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--app-root", type=Path, help="Review app checkout")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-name", default="Default")
    parser.add_argument("--candidate-name", default="Candidate")
    parser.add_argument("--title", default="Floating reconstruction")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--credit")
    parser.add_argument("--scorecard", action="append", nargs=2, metavar=("LABEL", "STAGE_REPORT"), default=[])
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
