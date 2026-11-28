# Run the local grid chooser pilot

This offline pilot tests whether source-image evidence can select a better grid for one photographed surface before the ordinary joins, component selection, material matching and export. It needs a completed set of paired exports. The included tests use invented counts and features. There is no measured chooser gain or reconstruction default change from those tests.

## Collect a paired packet

1. Freeze the cohort and list of frames before inspecting scores. Assign all angles, geometry variants and lighting conditions of one building family to one split. Include at least two development families and one untouched holdout family. For each frame preselect one decision using source information, for example the first incumbent wall with a bounded set of alternate phases in a fixed chart order. Publish the eligibility rule, excluded frames and their reasons alongside the packet. Scores describe this eligible subset.
2. Save the screenshot and the source-derived local grid handoff for **each** option, including the exact incumbent. For example, keep chart identity, ordered projected corners, U/V period and phase, dimensions and supported cell mask. Save one surroundings manifest per frame containing frame ID, frozen settings, code and asset revision, random seeds, and hashes of every other handoff and chart. Keep the state entering the local decision identical. Rerun downstream consumers with each offered grid, saving the whole exported pack including any changed joins or selected components. Never sum independent per-surface gains.
3. Before scoring, compute three source-only values per option. Use the same masks, edge detector, native resolution, atlas and feature definitions throughout the packet. Freeze these definitions and measurement settings in the surroundings manifest. Suggested measurements for this protocol are:

   | Field | Measurement on the supplied source screenshot |
   | --- | --- |
   | `edge_alignment` | Fraction of sampled, projected grid-boundary pixels within a fixed pixel radius of a detected image edge inside the chart. |
   | `atlas_agreement` | Fraction of supported cell crops passing a fixed texture agreement threshold against the frozen atlas. |
   | `visible_support` | Fraction of sampled pixels inside the proposed grid footprint covered by the source-derived surface mask. |

   Each field takes a finite value in [0, 1]. Preserve numerator, denominator and extractor revision in the producer's audit files. An unavailable measurement makes the decision ineligible. Keep values derived from recorded corners, camera pose, reference masks, other views, scorer outcomes and review answers out of all three fields.
4. Decode and score each whole exported pack using the same visible-only scorer and the geometry-selected alignment for material. Write the three count fields from each of the four standard Floating stage-report metrics. Apply the stage-report unresolved and empty-export conventions. Save the scorer revision and exact settings in a common evaluator manifest, and the hashed visible reference separately for each frame. The pilot consumes these scores and checks artifact identities. It relies on the producer for the score computation and for truthful enumeration of unchanged surroundings.

## Files

All paths in `{"path":"...", "sha256":"..."}` are relative to the JSON file that contains them. Use the SHA-256 of the actual bytes. The following Python objects show one view, with `ref(path)` building that two-field object. Replace the example paths and counts with the producer's measured inputs. Repeat the two option entries for any additional bounded proposals.

```python
import hashlib
from pathlib import Path

def ref(path):
    return {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}

view = {
    "id": "frame-017", "family": "building-family-A", "split": "development",
    "decision": "chart-3-grid", "screenshot": ref("frame-017/source.png"),
    "surroundings": ref("frame-017/fixed-inputs.json"), "incumbent": "keep",
    "options": {
        "keep": {"grid": ref("frame-017/keep/grid.json"),
                 "features": {"edge_alignment": .61, "atlas_agreement": .75, "visible_support": .91}},
        "phase-2": {"grid": ref("frame-017/phase-2/grid.json"),
                    "features": {"edge_alignment": .81, "atlas_agreement": .70, "visible_support": .90}},
    },
}
source = {"cohort": "Visible Builds, frozen eligible subset", "revision": "full revision",
          "views": [view]}  # include every frozen development and holdout view
```

Write `source.json` first. Produce separate `development.json` and `holdout.json` with the following structure. Each file includes precisely the views assigned to its split. The four `scores` entries below come from one decoded export. All options use the same visible reference for that view.

```python
def option_receipt(name, scores):
    return {
        "surroundings_sha256": view["surroundings"]["sha256"],
        "grid_sha256": view["options"][name]["grid"]["sha256"],
        "export": ref(f"frame-017/{name}/reconstruction.zip"), "scores": scores,
    }

# Each value has integer correct, missed and extras fields.
metrics = ("geometry_strict", "geometry_facade", "scores_strict", "scores_facade")
split_receipt = {
    "source_sha256": ref("source.json")["sha256"], "evaluator": ref("scorer-settings.json"),
    "views": {
        "frame-017": {
            "visible_reference": ref("frame-017/visible-reference.json"),
            "options": {name: option_receipt(name, scored_whole_export[name]) for name in view["options"]},
        },
    },
}
```

## Run and interpret

```powershell
python -m benchmark.floating_grid_chooser --source packet/source.json --development packet/development.json --holdout packet/holdout.json --out results/chooser.json
python -m unittest discover -s tests -p test_floating_grid_chooser.py
```

The fixed comparator ranks by edge alignment. The learned comparator ranks by one allowed feature and can require agreement from another feature before switching. Its optional gate requires the second feature to improve by at least 0, 0.125 or 0.25. Development fitting maximizes mean per-view strict placement F1 over a fixed grid of rules. Equal utility favors keeping the incumbent, then the simpler earlier rule. A tied top source ranking keeps the incumbent. The fitted rule is frozen before opening holdout scores.

Read the holdout section for each arm's four pooled F1 scores, count totals, paired differences, per-view selections, and wins/losses/ties. F1 and its deltas are fractions. The reference-only oracle maximizes each view's strict placement F1 among the same exported options, preferring the incumbent on a best-score tie. Its mean per-view strict gain bounds this objective. Its pooled and secondary scores describe those same choices. A zero development ceiling stops before opening the holdout receipt. Insufficient family coverage yields an explicit status. Review measured source-only gains and losses by building before considering a reconstruction change.
