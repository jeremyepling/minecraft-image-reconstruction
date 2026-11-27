"""Build the cross-arm placement report from independently decoded export receipts."""
import argparse
import gzip
import json
from pathlib import Path


def report(folder):
    scores = {}
    cohorts = ("visible-builds-v2", "visible-builds-v3", "decorative-builds-v1", "all")
    for receipt in sorted(folder.glob("*audit*.json.gz")):
        with gzip.open(receipt) as stream:
            data = json.load(stream)["cohorts"]
        for arm, pooled in data["all"].items():
            if pooled["views"] != 38:
                continue
            values = [data[c][arm]["metrics"] for c in cohorts]
            if arm in scores:
                assert values == scores[arm]
            scores[arm] = values
    parts = ["# Floating wall experiments across 38 held-out images",
             "All values are pooled block F1 percentages. Each export receives one whole-build translation. "
             "The cohorts contain 12 Twin Spire, 18 Castle and eight Walled Garden images, with eight distinct layouts across the three families. "
             "The fixed 104 production trend uses a different pipeline and cohort.",
             "The source baseline uses the ordinary surface finder. The native baseline supplies exact frontmost surface masks. "
             "The rulers/noop export uses those masks and the existing predicted wall grid, sampling cells over the entire mask. "
             "Spacing then supplies the true local pitch along both predicted axes. Origin also snaps one central grid corner to the reference lattice. "
             "The complete-grid exports additionally supply the individual visible cells and projected corners of full-cube walls, including walls originally missed by the counter. "
             "Source-normal estimates face directions from the source image. Signed supplies the recorded axis and facing. "
             "Both leave inter-wall offsets and depth to the assembler and estimate materials from source pixels. "
             "Two-edge variants require two matching segments for a perpendicular join. The count-only arm replaces the numeric dimensions at the counter handoff. "
             "The source_material arm retains original IDs for existing cells and uses the masked atlas matcher on newly sampled cells. "
             "The ordinary masks and grids are the only geometry inputs to both source resampling variants.",
             "Green highlights visible positions recovered after one whole-build translation. Extra placements are included in F1 and omitted from the overlays.\n\n"
             "![Recovered positions on three examples](placement.jpg)"]
    sections = (("geometry_strict", "Visible position"), ("geometry_facade", "Facade-priority position"),
                ("scores_strict", "Position and material"), ("scores_facade", "Facade-priority position and material"))
    order = ["source", "rulers/source_resample", "rulers/source_material", "native", "rulers/counts", "rulers/noop", "rulers/spacing", "rulers/origin",
             "grids/source-normal", "grids/signed", "grids/source-normal-two-edges", "grids/signed-two-edges"]
    for field, title in sections:
        lines = [f"## {title}", "", "| Export | Visible v2 | Visible v3 | Decorative v1 | All 38 |",
                 "|---|---:|---:|---:|---:|"]
        for arm in dict.fromkeys(order + sorted(scores)):
            if arm in scores:
                cells = [f'{100 * result[field]["f1"]:.2f}' for result in scores[arm]]
                lines.append("| " + " | ".join([arm, *cells]) + " |")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    args = parser.parse_args()
    target = Path(__file__).parent / "reviews/expanded-wall-connectivity-01/placement.md"
    target.write_text(report(args.receipts), encoding="utf-8")
