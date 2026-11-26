"""Collect paired public benchmark exports, metrics and part recoveries."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark.blind_shape_results import measure
from benchmark.texture_camera_eval import checked, digest, read, save


# The two challenger adapters live in this fixture branch. Their models come
# from separate checkouts, so the adapter revision alone cannot identify them.
OVERLAY_REVISION = {
    "nearby": "545252331519566327e86cec0801b3c8e57cd76c",
    "thin": "3b7b69c04bd9a992417fb5f6e69b3d4411b2e17f",
}


def archive(root, runs, names, dest, prefix="blind-"):
    dest.mkdir(exist_ok=True, parents=True)
    methods = {}
    with ZipFile(dest / "scores.zip", "w", ZIP_DEFLATED) as bundle:
        for name in names:
            inputs = runs / (prefix + name)
            manifest = read(inputs / "manifest.json")
            checked(root / "screen.json", manifest["selection_sha256"])
            cases = read(checked(inputs / "cases.json", manifest["cases_sha256"]))
            measure(root, inputs, inputs / "parts.json")
            records = ["manifest.json", "cases.json", "parts.json"]
            for row in cases:
                for filename in ("reconstruction.zip", "cells.json"):
                    records.append(f"{row['id']}/{filename}")
            for relative in records:
                bundle.write(inputs / relative, name + "/" + relative)

            methods[name] = {
                "adapter": manifest["adapter"],
                "revision": manifest["revision"],
                "rgb_sha256": [row["rgb_sha256"] for row in cases],
                "relative_geometry": manifest["summary"]["relative_geometry"],
                "cases_sha256": manifest["cases_sha256"],
            }
            if name in OVERLAY_REVISION:
                methods[name]["overlay_revision"] = OVERLAY_REVISION[name]
    save(dest / "summary.json", {
        "scores_sha256": digest(dest / "scores.zip"),
        "methods": methods,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    archive(args.root, args.runs, args.names, args.out)
