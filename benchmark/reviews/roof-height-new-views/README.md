# Native roof-height confirmation

Two newly authored geometries and four Java 26.3 screenshots were frozen in recipe 5bef426 before any predictions. The roof block course is measured above the first occupied block at local y=0. A masonry awning has four short piers under one course at 3. A wooden covered walkway has five tall posts and a projecting section at course 8. Both use the same ordinary vanilla smooth stone superflat and two oblique views. Each photograph is an independent input. All four pass the whole-structure framing audit, with at least 20.2% projected clearance at every edge.

The public [fixture archive](https://github.com/jeremyepling/minecraft-image-reconstruction/releases/download/roof-height-confirmation-v1/roof-height-confirmation-v1.zip) holds the original pixels, per-frame world readback and camera pose, view-masked labels, image-only metric camera result, actual ordinary CLI surface/count/placement manifests and decoded-pack inputs. Its SHA-256 is 513bc90233c22545843a5260eb3fd0a240f226def9de99d192f5946e449dc4d6. The native label audit checked 2,712 server readbacks and 567 visible faces. The full source geometry and recorded pose are opened after prediction for scoring.

## Four-view result

Each candidate produced the same four pack hashes as the 54c9fee ordinary CLI. Relative visible-block F1 was 86.46% with one global integer translation per photo. At the true post separation, 223 of 307 visible roof blocks were recovered. There were seven extra placements, zero hidden-block placements and zero fills on 85 observed clear rays.

| View | True course | Current highest | Atlas primary | Correct / placed / visible | F1 | Anchored roof / visible roof |
|---|---:|---:|---:|---:|---:|---:|
| Masonry low left | 3 | 3 | 2 | 27 / 27 / 47 | 72.97% | 15 / 35 |
| Masonry high right | 3 | — | — | 0 / 0 / 61 | 0.00% | 0 / 54 |
| Timber low left | 8 | 8 | — | 134 / 138 / 144 | 95.04% | 94 / 104 |
| Timber high right | 8 | 8 | 10 | 152 / 155 / 152 | 99.02% | 114 / 114 |

All three camera-supported photos already had the correct tallest and modal occupied course. The high-right masonry view failed image-only metric calibration. The loose atlas proposal on the low masonry roof chose the course below it, with 520 of 784 support pixels on the actual roof. Its high timber choice had zero of 125 support pixels on the roof. The contact-gated early model, early consensus, and hybrid all abstained on primary height in these four images. None gained or lost a correct block or changed an extra or hidden placement. The leftmost adjacent front-post pair defines the anchored bay for both geometries.

The new cases support retaining the conservative fallback. They provide no independent placement gain for promoting a roof-height method. They are four views of two geometries in daytime vanilla lighting.

## Replay

Extract the release archive to runs/new-height-input. The four inputs in selection.json pair with builds.json. Run each candidate in its own process to isolate module paths and use the verified 26.3 atlas. The baseline code in this branch is 54c9fee. The saved algorithm revisions are 5435770 for loose atlas and 3381ec7 for the early/consensus/hybrid trials. Existing camera/projection, material assets and local model dependencies follow the checkout guide.

~~~powershell
python -m benchmark.roof_height_native_panel --sources runs/new-height-input --data runs/new-height-input --out runs/height-prior
python -m benchmark.roof_height_native_panel --sources runs/new-height-input --data runs/new-height-input --out runs/height-atlas --candidate-root ../roof-height-atlas --candidate engine.roof_height_atlas:recover
python -m benchmark.roof_height_native_panel --sources runs/new-height-input --data runs/new-height-input --out runs/height-early --candidate-root ../roof-height-round --candidate benchmark.roof_height_early:supported
python -m benchmark.roof_height_native_panel --sources runs/new-height-input --data runs/new-height-input --out runs/height-consensus --candidate-root ../roof-height-round --candidate benchmark.roof_height_consensus_trial:consensus
python -m benchmark.roof_height_native_panel --sources runs/new-height-input --data runs/new-height-input --out runs/height-hybrid --candidate-root ../roof-height-round --candidate benchmark.roof_height_consensus_trial:hybrid
~~~

For a new native capture use the private scripted-client runner with python -m benchmark.roof_height_fixture --out runs/new-height-input/native. Run benchmark.roof_height_inputs with stages prepare, build and package, passing --root runs/new-height-input --out benchmark/reviews/roof-height-new-views. The release is the exact saved input for comparisons.
