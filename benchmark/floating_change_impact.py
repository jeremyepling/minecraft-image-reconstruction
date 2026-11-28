"""Identify Floating review products affected by edits, including unknown pipeline inputs."""
import argparse
from fnmatch import fnmatchcase
import json
from pathlib import Path
import subprocess

# Order is upstream first. Each entry declares the inputs it reads and artifacts it owns.
STAGES = {
    "fixtures": ((), ("training/floating*","training/*manifest*","training/*cohort*"), "frozen view IDs and visible references"),
    "fields": (("fixtures",), ("engine/surface_fields*","benchmark/floating_v1_fields*"), "normal and depth fields"),
    "masks": (("fields",), ("engine/surface_model*","engine/floating_sky*","engine/surface_mask*"), "visible surface masks"),
    "grid": (("masks",), ("engine/atlas*","calibration/*","benchmark/*ruler*","benchmark/*counter*","benchmark/*corner*","benchmark/full_block_lattice*","benchmark/floating_build_local_grid*","benchmark/floating_v1_expanded*"), "face corners, grid attempts and block IDs"),
    "parts": (("grid","masks"), ("benchmark/floating_pregrid*","benchmark/floating_depth*"), "depth partitions and inherited/refit grids"),
    "assembly": (("grid","parts"), ("benchmark/relative_face_assembly*","benchmark/floating_patch_combined*","benchmark/floating_horizontal*","engine/export*"), "joined charts, chart ownership and local exported blocks"),
    "score": (("assembly",), ("benchmark/relative_face_scoring*","benchmark/expanded_scores*","benchmark/mask_blocks*"), "decoded placement and material scores"),
    "surfaces": (("grid","parts","assembly"), ("benchmark/mobile_surface_evidence*","benchmark/floating_surface_evidence*","benchmark/floating_surface_loss_trace*"), "surface, grid and export overlays"),
    "perfect": (("score",), ("benchmark/floating_perfect*","benchmark/reviews/floating-perfect*"), "matched Perfect Stage diagnostic receipts"),
    "report": (("perfect","score"), ("benchmark/floating_stage_report*","benchmark/floating_visual_review*","benchmark/floating_local_review*"), "stage report and visual queue"),
    "analysis": (("report","surfaces"), ("benchmark/floating_review_decisions*","benchmark/floating_review_intake*","benchmark/floating_grid_chooser*"), "priorities, source-only tests and pending decisions"),
    "site": (("analysis","surfaces"), ("review-site/*","benchmark/build_floating_mobile_review*","benchmark/publish_floating_review*","benchmark/reviews/mobile-review-assets/*"), "published review cards, overlays and links"),
}
RUNNERS = {"benchmark/floating_v1_run.py","benchmark/floating_build_local_run.py",
           "benchmark/floating_selective_replay.py","benchmark/floating_stage_cache.py"}

def plan(paths):
    direct = {}
    unknown = []
    for path in sorted(set(paths)):
        path = path.replace(chr(92), "/")
        owners = [key for key, (_, patterns, _) in STAGES.items()
                  if any(fnmatchcase(path, pattern) for pattern in patterns)]
        if path in RUNNERS:
            owners = ["fields"]  # A runner can change the complete stage order.
        if not owners and path.startswith(("engine/", "benchmark/", "runtime/", "training/", "calibration/")):
            if Path(path).suffix.lower() not in {".md",".html",".jpg",".png",".webp"} and not Path(path).name.startswith("test_"):
                unknown.append(path)
        for owner in owners:
            direct.setdefault(owner, []).append(path)
    affected = {}
    for stage, (parents, _, artifact) in STAGES.items():
        changed_parents = [p for p in parents if p in affected]
        if stage in direct or changed_parents:
            affected[stage] = dict(artifact=artifact, files=direct.get(stage, []), upstream=changed_parents)
    return dict(refresh=affected, needsClassification=unknown)

def changed(base, head):
    data = subprocess.check_output(["git","diff","--name-only","--no-renames","-z",base,head,"--"])
    return [p.decode("utf-8") for p in data.split(bytes([0])) if p]

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("paths", nargs="*")
    cli.add_argument("--base")
    cli.add_argument("--head", default="HEAD")
    cli.add_argument("--json", action="store_true")
    options = cli.parse_args()
    if not options.paths and not options.base:
        cli.error("Supply paths or --base.")
    impact = plan([*options.paths, *(changed(options.base, options.head) if options.base else [])])
    if options.json:
        print(json.dumps(impact, indent=2))
        return
    print("## Floating evidence affected by these edits\n")
    for stage, record in impact["refresh"].items():
        print(f"- **{stage}** — check {record['artifact']} ({', '.join(record['files'] or record['upstream'])})")
    if not impact["refresh"]:
        print("No declared outputs affected.")
    if impact["needsClassification"]:
        print("\nCheck these unclassified dependencies before reusing a measurement:")
        for path in impact["needsClassification"]:
            print(f"- `{path}`")

if __name__ == "__main__":
    main()
