# Provisional placement from visible surfaces

This controlled experiment keeps the frozen sky-gap masks and existing decoded placements. It also tests all masks for additional visible cells, including masks that already emitted some cells. The fallback selects axis-aligned planes from the masks' measured camera normals. It estimates plane offsets using the observed silhouette's ground contact where available, and otherwise uses the predicted relative inverse depth. Where the original camera was uncertain, it estimates orientation from the surface normals, uses chart-based lens estimation, and infers a rough viewing distance from counted texture repeats. When there is no repeat measurement it assumes one block spans one twenty-fourth of the frame height. The lowest observed structure pixels supply an approximate ground anchor. This assumes some observed pixels reach the ground, which can fail on cropped views. Depth remains ambiguous.

Each extra cell needs at least four source samples on its predicted face and 50% visible mask coverage on the clipped face footprint. After 3D assembly, the runner drops new cells with fewer than four nearest samples in any of the original supporting masks. Each case writes and decodes a Minecraft structure pack before evaluation. Pose and frame labels enter only after export. Surface masks, per-frame camera estimates, and the original placement replay receive no scoring inputs.

| Metric | Frozen placement | Provisional placement |
| --- | ---: | ---: |
| Frames with blocks | 9 / 56 | 56 / 56 |
| Correct visible coordinates | 344 | 3,408 |
| Extra coordinates | 18 | 15,079 |
| Registered silhouette IoU | 2.75% | 43.61% |

IoU is total intersection divided by total union. Source-camera silhouettes are rendered from decoded block coordinates on stride-four pixel centers. The main table uses the recorded reference camera and the same integer registration as the placement scorer. In the fallback's own inferred camera, silhouette IoU is 92.89%. The separation measures how much apparent shape depends on the chosen camera. On the 16 camera-supported views the fallback recovers 2,024 locations with 621 extras. On the other 40 views it recovers 1,384 locations with 14,458 extras.

To reproduce with the frozen source checkout, archived baseline packs, and per-frame visible targets from the baseline checker:

```powershell
python -m benchmark.local_visible_placement --root C:/code/minecraft/minecraft-image-reconstruction --baseline C:/code/minecraft/workspace/builds/end-to-end-placement --out C:/code/minecraft/workspace/builds/placement-rigidity-local
python -m benchmark.local_visible_scoring --root C:/code/minecraft/minecraft-image-reconstruction --baseline C:/code/minecraft/workspace/builds/placement-baseline-check --out C:/code/minecraft/workspace/builds/placement-rigidity-local
```

The output contains the fitted camera and per-frame source fingerprints, exported packs, pose-registered coordinates, binary source-camera footprints, registered footprints and per-case metrics. No material matcher ran in this placement experiment. Surface masks came from the pinned frozen suite; the predictor uses their observed-camera-angle geometry and image-based depth fields. Keep this branch experimental while uncertain-view registration and depth produce substantial excess placements.

