# Additional camera-supported native views

The frozen integrated placer at `eea027b` raised relative geometry F1 from **79.58% to 81.94%** against `3585aa5`. It recovered 188 correct visible blocks and lost one. It also added 71 extras and removed 17. The result supports using the integrated placer for another camera view of a known structure. These 20 screenshots cover 16 previously evaluated occupancy shapes. Generalization to new structures remains a separate test.

| Decoded geometry | Previous | Integrated |
|---|---:|---:|
| Correct of 4,098 visible blocks | 2,986 | 3,173 |
| Missed visible blocks | 1,112 | 925 |
| Extras | 420 | 474 |
| Extras coinciding with hidden source blocks | 54 | 56 |
| Precision | 87.67% | 87.00% |
| Recall | 72.86% | 77.43% |

Twelve views improved, five tied and three declined. The largest loss was 0.35 F1 points on Corner 03 from a single extra block. The largest gain was 9.60 points on Canopy 09 from 14 recovered blocks. Grouping alternate views by occupancy shape gave eight improvements, five ties and three declines.

Eight views depict the exact same world-coordinate canopy structures as existing roof-to-post probes. With both front posts defining a shared origin, the previous placer aligned 112 of 398 roof positions and the integrated placer aligned 131. Correct roof-and-post triples rose from 575 to 670 of 2,005. Both measured six exact gap widths and left two unresolved. Across all 20 views the original camera estimates projected all 5,708 checked reference corners within two native pixels.

## Selection and replay

Commit `c025a2a` froze the selection before placement scoring. The earlier screen had 140 RGB views and 40 distinct images with supported cameras. The previously selected 12 reviewed images and eight additional native confirmation images account for 20 of those 40. We used every remaining supported image and checked its hash against all 112 evaluation rows. Four pairs of fixed-suite screenshots have identical RGB bytes across their two floor settings, so those 112 rows contain 108 unique images. None appears among these 20.

Both revisions consumed the same frozen estimated camera, original screenshot and newly generated ordinary surface masks and counts. Their RGB, surface and count hashes match per image. The scorer decoded the real packs and allowed one integer translation for the whole structure. Camera truth and complete source readbacks supplied evaluation labels. [Selection with frame and source hashes](screen.json) · [Per-frame changes](results.json) · [Eight roof probes](roofs.json) · [Decoded packs, coordinates, camera audit and manifests](scores.zip)

The release includes the 20 original screenshots, camera and block readbacks, frozen model-input records and surface/count manifests. Its archive is 29,331,160 bytes with SHA-256 `fcb5187313ddeccde01263757ac1a8e275ccbb2fed96fd60937b14c0f4ef1c3e`. [Portable fixture](https://github.com/jeremyepling/minecraft-image-reconstruction/releases/tag/visible-recovery-extra-views-v1)

```powershell
gh release download visible-recovery-extra-views-v1 --repo jeremyepling/minecraft-image-reconstruction --pattern extra-views-v1.zip --dir runs
Expand-Archive runs/extra-views-v1.zip runs/extra-views-fixture
$env:MCIR_NATIVE_TRANSFER_FIXTURES = (Resolve-Path runs/extra-views-fixture/fixtures).Path
$inputs = (Resolve-Path runs/extra-views-fixture/confirmation).Path
python -m benchmark.canopy_compare --inputs $inputs --candidate-root PATH_TO_3585AA5 --candidate benchmark.visible_depth_score:integrated --out runs/extra-baseline
python -m benchmark.canopy_compare --inputs $inputs --baseline runs/extra-baseline --candidate-root PATH_TO_EEA027B --candidate benchmark.visible_depth_score:integrated --out runs/extra-candidate
```
