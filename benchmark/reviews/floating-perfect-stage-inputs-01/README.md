# Frozen V1 reference interventions

Each row follows one released V1 RGB frame to a decoded Minecraft pack. The control is the guarded-pregrid `current` arm at `01d51c03929340d97c17befb76505e151e754bda`. Wall charts cover 144 frames from regression, train, validation and test. Both Surface Finder arms cover all 12 held-out Twinspire test frames. Each frame uses its own original image, native frontmost owner buffer, visible-face labels and recorded camera.

| Cohort and intervention | Correct / missed / extra | Strict / facade placement F1 | Strict / facade position + ID F1 |
| --- | ---: | ---: | ---: |
| All 144, current | 10,156 / 24,298 / 6,974 | 39.38% / 41.37% | 17.06% / 18.54% |
| All 144, exact local wall charts | 24,263 / 10,191 / 427 | 82.05% / 85.41% | 63.89% / 70.41% |
| Test 12, current | 1,143 / 2,271 / 376 | 46.34% / 48.77% | 23.03% / 24.24% |
| Test 12, exact surface masks | 1,269 / 2,145 / 537 | 48.62% / 51.00% | 27.62% / 29.04% |
| Test 12, exact masks and camera-space plane measurements | 1,178 / 2,236 / 58 | 50.67% / 53.50% | 38.67% / 40.90% |
| Test 12, exact local wall charts | 2,632 / 782 / 8 | 86.95% / 90.18% | 76.97% / 80.10% |

## Intervention contracts

- **Surface masks.** Replace the kept Surface Finder proposals with one native-resolution row-run mask per visible labeled surface. Each mask is the union of its first-hit face-owner pixels. The ordinary model normal and relative inverse-depth fields supply median values under that mask. Source contour fitting, unit counting, tile assignment, wall expansion, guarded pregrid and assembly run downstream. Both surface arms pass all 216 test surfaces into the adapter. The emitted handoffs preserve each injected mask, camera normal and scalar depth.
- **Masks and camera-space plane measurements.** In addition to each exact mask, supply the exact optical-frame unit normal and median inverse optical depth at its owned pixel centers. The existing scalar adapter receives a monotonic min–max transform across that frame’s surface medians. The handoff retains the measured medians in reciprocal blocks. This arm measures plane attributes at retained proposals. The downstream dense field consumers continue to read the image model fields.
- **Local wall charts.** Replace the vertical full-cube chart collection with its exact visible unit membership, scale, phase and projected shape. A face qualifies only when its first-hit pixel count is positive. Each chart has an independent zero origin. The ordinary atlas tile ranker reads RGB samples at owned visible texels. Its no-evidence fallback supplied 194 of 29,578 units. Source normals feed the ordinary axis estimator and the original surface handoff supplies any horizontal chart proposals. The pack writer receives the ordinary winning component and material selections. This intervention bundles scale, phase, chart coverage and face membership. Source materials, horizontal proposals and world registration can each limit its final score.

For all frames, the native owner histograms matched the recorded visible-pixel counts. Projected camera corners differed by at most `5.1e-13` pixel. Every supplied wall unit agreed with all four native face corners to within `7.3e-11` pixel. The wall-chart arm gained 15,255 correct positions and lost 1,148 against the paired control. Seven control exports and two wall-chart exports lacked an identified yaw. V1 charges every exported block as extra and every required block as missed in that case.

## Reuse and verification

Each stage has a conditions file, an aggregate summary and a ZIP containing its actual input, placements, score and serialized reconstruction for each frame. Per-frame records pin the native RGB and reference hashes, baseline pack and chart digests, image field and model hashes, atlas, environment, source generator and downstream code. `cache_key` is the SHA-256 of the canonical configuration JSON. The `base` and `output` records contain strict and facade placement and position-plus-ID scores. The `paired` record compares scored registered coordinate and exact-ID sets. A nonempty unresolved export has no registered site attribution.

Verify a published archive from either the pinned experiment or a Floating checkout with the shared pack decoder. Run the command from the repository root with Python and the repository’s NumPy and Pillow dependencies.

```powershell
python -m tools.verify_floating_stage_inputs
```

The verifier reads all 168 reconstruction packs, checks structure templates against the saved local placements, validates input/score/pack digests and recomputes the reported counts and aggregate F1. A verified archive can be used without the source model cache or Minecraft. ZIP headers in the original pack writer include creation times. Semantic replication compares decoded blocks and counts, while byte hashes identify the specific frozen export.

To regenerate on the measured consumer revision, use the worktree for `experiment/floating-perfect-stage-inputs` beside `floating-visible-builds-v1`, with its archived `runs/floating-v1` outputs and `runs/v1-fields` source model products. Set `FLOATING_V1_RELEASE` to the restored V1 release root containing `data/<split>/frames/<id>`. The pinned package used NumPy 2.3.5, Pillow 13.0.0 and opencv-python 4.12.0.88 with the repository's texture atlas. Replay uses the existing local cache when all input and consumer hashes agree.

```powershell
python -m benchmark.floating_perfect_stages --stage surface-mask --split test
python -m benchmark.floating_perfect_stages --stage surface-plane --split test
python -m benchmark.floating_perfect_stages --stage wall-grid --split all
```

These measurements belong in the Floating diagnostic ledger. They support development of a source-only local wall chart predictor. The frozen packs exercise the existing source assembler, and the interventions require visible reference labels. Production and fixed-104 scores are separate.
