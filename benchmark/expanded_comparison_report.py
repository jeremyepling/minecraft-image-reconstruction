"""Compile the paired final scores from the independently replayable experiment artifacts."""
from hashlib import sha256
import json
from pathlib import Path


REVIEWS = Path(__file__).resolve().parent / "reviews"
DATASETS = ("fixed-104", "visible-builds-v3", "decorative-builds-v1")


def read(path):
    return json.loads(path.read_bytes())


def arm_metrics(row):
    return {"geometry": {key: row["geometry"][key] for key in ("strict", "facade")},
            "scores": {key: row["scores"][key] for key in ("strict", "facade", "material")}}


def check_control(results, released):
    for section in ("geometry", "scores"):
        for metric in ("strict", "facade"):
            for count in ("correct", "missed", "extras"):
                if results[section][metric][count] != released[section][metric][count]:
                    raise ValueError(f"Control mismatch in {section}/{metric}/{count}")


def method_rows(method, title, artifact, cohorts, source, released, note, oracle="oracle"):
    rows = {}
    for dataset in DATASETS:
        measured = cohorts[dataset]
        baseline = measured["baseline"]
        check_control(baseline, released[dataset])
        arms = {"baseline": arm_metrics(baseline), "source": arm_metrics(measured[source])}
        if oracle is not None:
            arms["oracle"] = arm_metrics(measured[oracle])
        rows[dataset] = {"views": released[dataset]["tested"], "note": note, "arms": arms}
    return {"id": method, "name": title, "artifact": str(artifact.relative_to(REVIEWS).as_posix()),
            "sha256": sha256(artifact.read_bytes()).hexdigest(), "cohorts": rows}


def main():
    baseline = read(REVIEWS / "expanded-baseline-01/summary.json")["cohorts"]
    experiments = []
    for slug, title, note in [
        ("visible-boundary-direction", "Surface boundaries and face directions",
         "Compare the original surfaces with source-derived alternatives at a visible seam. Each option runs through placement and final material assignment."),
        ("placement-grid-choice", "Placement grid chooser",
         "Keep the measured block counts and evaluate which of the offered grid corners best fits the source image."),
        ("full-block-lattice", "3D block grid from structure textures",
         "Use visible full-block texture matches to estimate a shared 3D grid and propose occupied cells."),
    ]:
        artifact = REVIEWS / (slug + "-01") / "summary.json"
        cohorts = read(artifact)
        experiments.append(method_rows(slug, title, artifact, cohorts, "source", baseline, note))
        if slug == "full-block-lattice":
            experiments.append(method_rows(slug + "-rescue", "3D block grid after camera failure", artifact,
                                           cohorts, "rescue_only", baseline,
                                           "Apply the structure-derived grid only when the ordinary camera failed. Camera-supported builds keep their original exports.", oracle=None))
    material_path = REVIEWS / "view-conditioned-atlas-01/summary.json"
    material = read(material_path)
    material_arms = {
        key: {arm: {"geometry": baseline[key]["geometry"], "scores": material[key][arm]}
              for arm in ("baseline", "source", "oracle")}
        for key in DATASETS
    }
    experiments.append(method_rows(
        "view-atlas", "View-adjusted atlas on ordinary geometry", material_path, material_arms, "source", baseline,
        "Reidentify materials at the same released coordinates using an atlas sampled over each source pixel. The reference chooser may keep the old ID or choose the candidate ID for each individual cell."))
    combined_path = REVIEWS / "view-atlas-lattice-01/summary.json"
    combined = read(combined_path)
    experiments.append(method_rows(
        "view-atlas-grid", "View-adjusted atlas and structure grid", combined_path, combined, "source", baseline,
        "Apply the view-adjusted texture matches during anchor and subface proposal generation, then grade the decoded build using the same final view-adjusted material matcher. Compare with the ordinary export on the same images."))
    out = REVIEWS / "expanded-round-01" / "summary.json"
    out.write_text(json.dumps({"experiments": experiments}, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
