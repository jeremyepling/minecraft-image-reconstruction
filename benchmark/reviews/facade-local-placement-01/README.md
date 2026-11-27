# Place the photographed back of a recess locally

**All 28 photographed tuff back blocks in the three marked recesses now occupy their correct relative positions.** On the named Facade, the local placement recovers 12/12 in D, 10/10 in B and 6/6 in C. It also removes 15 false foreground blocks. Whole-image geometry improves from 277 correct, 84 missing and 32 extra (82.69% F1) to 305 correct, 56 missing and 17 extra (89.31%). The marked letters, original masks and recorded return directions are shown in the accompanying baseline. [Original photo and masks](../facade-insets-01/README.md) · [All seven recessed walls and the flat panel](named-openings.json)

| Cohort | Images | Changed | Correct gained | Extras removed | Strict F1 before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed structure views | 104 | 7 | 84 | 49 | 90.50% → 90.80% |
| Development training views | 72 | 6 | 46 | 20 | 91.21% → 91.41% |
| Extra reference views | 20 | 4 | 65 | 42 | 88.67% → 90.02% |
| Four new-building views | 4 | 0 | 0 | 0 | 83.58% → 83.58% |
| Four independent shelter/gate views | 4 | 0 | 0 | 0 | 83.47% → 83.47% |

The same 204 saved views produced zero correct positions lost, zero new extras and zero new hidden blocks. All 84 fixed-suite gains are required by the local 4×4 rule. Facade-priority F1 improves from 92.56% to 92.90% (18,798 correct, 1,778 required misses, 1,097 extras). The historical comparison with eight retired texture-only images is 90.04% versus 89.74% strict. The fitted rule was compared on these known views and remains untested on a new independent building that actually triggers it. [Frozen per-view manifests](cases.json) · [Per-split strict counts](summary.json) · [Facade-priority regrade](priority.json)

## Placement and material handling

The host wall’s integer grid plane, rim rectangle and signed setback come from the strict image-only detector. The back must have an observed mask backed by the photographed side and sill. For each pixel in the back mask, intersect the camera ray with the measured back plane. Count support only inside the measured rim and away from the nearest grid line by at least 0.015 blocks on each tangent axis. Four native pixels in the same face interior qualify a cell. This preserves partially visible edge columns in the named Facade.

Start with the existing ordinary reconstruction so other walls retain their previous positions and states. Test only original foreground cells inside this rim and in front of its measured back. Render the original visible cells and require at least four supporting rays, with at least 90% of that cell’s first-hit rays projecting onto the photographed side, sill or back. This permits a small raster boundary error. Cells outside the measured grid region and completely occluded original cells stay untouched.

The stricter 100%, 95% and selected 90% ray gates all recover the same 84 fixed target positions and remove 32, 37 and 49 extras respectively. The more permissive 80% gate removed 62 extras on the fixed views and 20 instead of 15 on the named image. All four avoided losses here. Keep 80% for another building with photographed relief. Adding only the visible back cells yields 90.70% strict and 92.78% facade-priority. [Threshold results by split](sensitivity.json)

The fixture uses the ordinary configured geometry material, `minecraft:smooth_stone`, for all coordinates. The verifier decoded each changed datapack, checked every retained material against the original map, and used that configured geometry material for new coordinates. Source truth identifies all 28 newly placed Facade back blocks as tuff. This comparison measures geometry; it provides no material recognition gain. [Actual changed exports, sparse coordinates and frozen input hashes](exports.zip)

## Reproduce

The source revision for this experiment is `c589f1a` and the baseline engine for each image is its ordinary pre-inset reconstruction. Reuse the captured 204-frame strict detector run from `facade-inset-round/runs/returns` and the previously verified production packs. The candidate manifest records the SHA-256 of those detector cases. The local module applies no source labels or recorded camera information. The evaluator uses them only after writing and decoding the pack.

```powershell
python -m benchmark.inset_local_placement --workers 4
```

The helper `engine.visible_inset_cells.inset_edits(size, camera, parts, events, original)` takes the photograph size, its estimated camera, detected surfaces, strict detector events and ordinary local coordinate set. Apply the returned additions and removals to that ordinary set. Empty events produce empty edits. When the detector yields an event for only part of a build, recomputing unrelated ordinary placement from the split masks can affect unrelated chart choices. Preserve the pre-inset surfaces and counts for the baseline placement before applying the photographed patch.

