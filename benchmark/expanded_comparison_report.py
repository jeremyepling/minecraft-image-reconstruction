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
    for slug, title, note in [
        ("camera-horizon-floor", "Floor-based camera recovery",
         "Fit two independent repeats of the whole block texture on the floor when the ordinary camera fails. Test candidate horizons and require atlas verification on separated floor patches. Accepted cameras run through the ordinary build."),
        ("camera-structure-scaffold", "Camera from long structure edges",
         "Fit a scaffold from visible roof, wall and base edges. Attempt to establish block scale and ground phase before sending a camera to the ordinary build."),
    ]:
        artifact = REVIEWS / (slug + "-01") / "summary.json"
        report = read(artifact)
        cohorts = report["datasets"] if slug == "camera-horizon-floor" else report
        experiments.append(method_rows(slug, title, artifact, cohorts, "source", baseline, note, oracle=None))
    camera_path = REVIEWS / "camera-geocalib-01/summary.json"
    camera = read(camera_path)['arms']
    for key, title in (("matched", "Default camera: repeats and wider image-only search"),
                       ("geocalib", "GeoCalib after whole-tile repeats")):
        cohorts = {dataset: {'baseline': camera['ordinary'][dataset], 'source': camera[key][dataset]}
                   for dataset in DATASETS}
        experiments.append(method_rows('camera-model-' + key, title, camera_path, cohorts, 'source', baseline,
            'Both policies preserve ordinary and original whole-tile repeat successes. The wider image-only control '
            'recovers the same three Harbor validation views as GeoCalib. The learned model adds coverage only on '
            'two training palettes of one viewpoint. Compare their rows to separate the earlier repeat gain.', oracle=None))
    moge_path = REVIEWS / 'camera-moge-surface-first-01/scores.json'
    moge = read(moge_path)
    cohorts = {dataset: {'baseline': moge['ordinary'][dataset], 'source': moge['repeat_then_moge'][dataset]}
               for dataset in DATASETS}
    experiments.append(method_rows('camera-moge', 'MoGe surface-first after whole-tile repeats', moge_path,
        cohorts, 'source', baseline, 'Experimental: whole-build translation gives partial placement credit, but '
        'native corner projection and material errors expose physically incorrect cameras. Existing supported builds are kept.', oracle=None))
    joint_path = REVIEWS / 'camera-grid-joint-01/summary.json'
    joint = read(joint_path)
    cohorts = {dataset: {'baseline': joint['incumbent'][dataset], 'source': joint['source'][dataset],
                        'oracle': joint['joint-oracle'][dataset]} for dataset in DATASETS}
    experiments.append(method_rows('camera-grid-joint', 'Joint camera and wall-grid refinement', joint_path,
        cohorts, 'source', joint['incumbent'], 'Control is the new image-only default. The source verifier rejects '
        'all 70 residual views. The reference-only chooser includes available raw candidates and four coupled '
        'exports on two source-selected frames; this is only a finite candidate-set diagnostic. Its winning joint '
        'cameras remain physically incorrect.'))
    out = REVIEWS / "expanded-round-01" / "summary.json"
    out.write_text(json.dumps({"experiments": experiments}, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
