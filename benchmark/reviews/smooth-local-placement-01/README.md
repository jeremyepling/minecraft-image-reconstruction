# Local placement from smooth wall patches

The placement helper starts from the ordinary visible build and tests unassigned pixels connected to a nearly uniform patch within a measured surface. A temporary expanded mask provides a second fit. A candidate block is retained only if it has at least 16 frontmost samples at two-pixel spacing, at least 60% of its visible samples land inside the newly observed region, and those samples show the face axis of the supporting surface. Every original block is preserved. The exported Surface Finder masks and Block Counter measurements continue to use their original observed support.

The fixed 104 whole-structure source images gain 77 required blocks. All eight changed datapacks decode to strict supersets of the prior builds. No extra block or hidden block was added, and the 100 development and transfer views tie in all four additional cohorts. The comparison starts from `a5a88ad`, which includes the eave-pinhole and repeated-texture outline changes. Its placement scores match the previous pavement release on every cohort.

| Fixed 104-view relative placement | Before | With local additions | Correct / missed / extra |
|---|---:|---:|---|
| Strict F1 | 90.8126% | 90.9974% | 20,696 / 3,006 / 1,089 |
| Facade-priority F1 | 92.9145% | 93.1179% | 18,875 / 1,701 / 1,089 |

Registry plaster high-right gained 43 blocks. Shopfront plaster low-left gained two on the gray-concrete floor and one on the smooth-stone floor. Portico plaster high-right and Crossing plaster low-left each gained five. Three Wing views contributed 21. The historical 112-view strict score including retired wall close-ups is 90.2357%. Accuracy uses one whole-structure translation for each source image. The saved production cameras gave the same pooled strict score without translation. [Metrics and archive SHA-256](summary.json) · [Per-view metrics and previous pack hashes](cases.json) · [Changed datapacks](default-exports.zip)

The helper requires the ordinary Java 26.3 joint placer, texture-guided counter and usable camera. Images involving accepted inset or pavement-opening events keep their dedicated placement path. It builds a second candidate only for the 21 views with qualifying patches in the 204-view replay. With five concurrent workers and machine load from ordinary builds, the initial scan took a median 0.20 s on the 162 views that reached it and had no proposal. The 21 positive scans with candidate evaluation took a median 14.35 s, including the recount, placement and source reprojection. A visibly similar background area can produce a temporary proposal. The helper never publishes that candidate as a measured surface.

The ordinary CLI reproduced the released coordinates and materials on Registry (+43), Portico (+5), Shopfront (+2) and Wing (+12). Release pack digests refer to the scored archive entries. Separate ordinary-run package digests and equality checks are recorded as well. [Ordinary-build verification](ordinary.json)

## Replay

Run with the registered frozen fixtures and prepared Java 26.3 atlas.

```powershell
python -m benchmark.smooth_local_release --workers 5
python -m unittest tests.test_smooth_extent_placement
```

For each frame ID in `ordinary.json`, run `Invoke-WallRecovery.ps1` on its original `runs/visible-builds-v2/data/regression/frames/<id>/rgb.png` from the stable checkout. Use `-Through placement`, the stable checkout's `runs/26.3/atlas` and output `runs/smooth-cli/<id>` in this checkout. Then run `python -m benchmark.smooth_local_release --cli`. The verifier compares every original block coordinate and material in those four generated datapacks with the corresponding release pack.
