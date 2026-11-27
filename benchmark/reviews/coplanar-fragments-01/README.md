# How often one physical plane is divided into several masks

On the **104 whole-structure photographs at edb10a3**, 56 of 1,462 recorded visible surface observations (3.83%) had at least two substantial neighboring predicted fragments. Those 56 surfaces appeared in 42 views across 20 of the 40 layout identities. Together they accounted for 117 qualifying mask pieces. A surface observed on two photographs contributes two observations. [Three example photographs and mask comparisons](split-surfaces.jpg)

Among 3,167 missing visible blocks, **89 (2.81%)** have a photographed face somewhere on one of those surfaces. **57 (1.80%)** have actual face pixels covered by the identified fragments, and **15 (0.47%)** have face pixels at the seam between them. Each count deduplicates block coordinates within a photograph and uses the ordinary single shared translation when identifying misses. The numbers are nested and do not establish that joining the masks recovers the blocks.

| Definition | Split surface observations | Affected views | Missed on the surface | Missed within the pieces | Missed at the seam |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inclusive | 77 | 56 | 226 | 151 | 27 |
| Standard | 56 | 42 | 89 | 57 | 15 |
| Strict | 44 | 36 | 24 | 11 | 6 |

The captured surface IDs connect visible block faces that share an edge in the world, have the same signed direction and lie on the same plane. They extend across a change of block material. The standard measure requires each original mask to place at least 80% of its native pixels on the same recorded surface and to cover at least 5% of that surface and at least 64 pixels. Two qualifying masks may overlap on at most 10% of the smaller supported area. At least eight distinct pixels must lie within four native pixels of the other mask along a horizontal or vertical image line. Adjacency groups are counted once per surface per photograph. The inclusive sensitivity uses 70%, 2% and 32 pixels; the strict sensitivity uses 90%, 10% and 128 pixels. Neither gives a universal upper or lower bound on split-related failures: an impure mask involving several surfaces or a larger gap can escape all three definitions. [Full census and each source mask ID](cases.json) · [Counts by cohort](summary.json)

The right Facade has a false diagonal between two pieces of the same wall. Zero currently missed blocks lie on that specific plane. The pictured plaster Shopfront and Canopy 04 each contain a qualifying split with zero misses on its associated plane. The photographed Dev Relief 00, Relief 00 and Colonnade 01 have no qualifying split under this definition. On the masonry Portico high-right view, 25 misses lie on the fragmented roof plane. Most of the thin far roof remains outside its two detected pieces.

## What happens after a merge

The image-only trial fits a 16×16 texel map to each adjacent, similarly directed mask at each plausible integer plane depth. Repeating blocks in alternating checkerboard cells must predict one another, and the winning levels must agree independently with a margin over alternative depths. To join, both sides also need at least 28 common well-supported texel bins, cross-boundary texture correlation at least 0.72 and similar mean intensity. It can detect a false diagonal across repeating stone while retaining a seam between two posts at competing depths.

A separate scoring-only intervention supplies the exact standard census groups from recorded labels. Both trials union the original mask pixels without filling any gap, recalculate their sampling charts and block counts with the same calibrated atlas and run the complete ordinary placer. Neither provides a correct plane depth, outline, material or block count to the placer. The label-assisted run tests the specific effect of combining existing fragments and recounting them. It is not an upper bound on a future joint depth, extent and surface estimator.

### Measured changes

Each column counts individual gains and losses against the same frozen export. “Added / removed” counts incorrect placed blocks. Surface matches use one-to-one assignment at native IoU ≥0.5. Paired pixels measure correctly assigned surface area. Surface-stage improvements can occur when the original wall already passed the match threshold. [Frame-level matches and mask IDs](merge-detail.json) · [Placement metrics](placement.json) · [Hash-checked comparisons and changed exports](exports.zip)

| Fixed 104 intervention | Same-plane groups | Matches gained / lost | Unmatched masks Δ | Paired pixels Δ | Correct blocks gained / lost | Extras added / removed | Geometry F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | — | — | — | — | — | — | 90.4964% |
| Repeating texture on both sides | 7/7 | 2 / 0 | −10 | +168,373 | 0 / 0 | 0 / 0 | 90.4964% |
| Recorded fragment assignments | 56/56 | 5 / 0 | −66 | +556,981 | 18 / 0 | 2 / 15 | 90.5658% |
| Color change with shared projected depth | 8/8 | 1 / 0 | −10 | +84,659 | 18 / 0 | 2 / 15 | 90.5658% |
| Diagonal wall cut with matching registered texture | 1/1 | 0 / 0 | −1 | +31,234 | 0 / 0 | 0 / 0 | 90.4964% |

| Cohort | Repeating texture | Recorded fragment assignments | Color change with shared depth | Guarded diagonal |
| --- | --- | --- | --- | --- |
| Development, 72 | +2/−0 correct, +1/−1 extras | +36/−30 correct, +21/−18 extras | +3/−0 correct, +0/−1 extras | All tied |
| Alternate, 20 | All tied | 1 extra added | All tied | All tied |
| New switchback and portico, 4 | All exports tied; one false merge lost three matched surfaces | +5/−2 correct, +5/−2 extras | −1 correct, +3 extras on the brick switchback | All tied |
| Independent shelter and gate, 4 | +4/−15 correct, +7/−3 extras; two false merges lost three matches | No label-assisted replay | 8 correct lost, 73 extras added, 125 extras removed; one correct surface join | All tied |

**Shopfront.** The recorded assignment joins the gray side of a five-block-high post with its wider brown cap. Both pieces already contributed zero missed blocks on that reference surface. The original placer projects 790 of 797 stem samples and 554 of 588 cap samples onto the same plane at X=5. The stem's texture independently favors X=5. The cap weakly favors X=13. Their original vertical count estimates are 4.40 and 4.15; the merged T-shaped mask has no resolved dimensions. Suppressing the original stem count, the cap count, or both without merging the masks reproduced the original export in all three tests. The new shape and sampling chart influence placement elsewhere. The view goes from 103/169 correct with 21 extras to 121/169 with 8 extras, 70.31% to 81.21% F1. These 18 blocks account for the entire fixed-set gain. Joining the two roof pieces on the masonry Portico does not recover any of its 25 associated misses. [Shopfront chart and count ablations](shopfront-probe.json)

The follow-up for a material change was designed after seeing Shopfront. Both touching masks must have the same signed direction, a chromaticity difference of at least 0.08, at least 90% of mask samples on one common first-hit plane and at least two distinct projected blocks with 12 samples each. At least one side independently identifies that depth from texture. Shopfront is the only fixed photo with a changed export. On the new Switchback, the surface join itself is correct but placement worsens. A local surface improvement can change global chart placement. The independent shelter/gate F1 also increases from 83.47% to 85.61% despite losing eight true blocks and moving many extras. Keep this rule experimental.

The guarded wall alternative uses the current image-estimated metric camera, the two observed masks and their RGB pixels. It searches up to 49 integer planes around the photograph's ground anchor independently of any placement. A straight seam tilted ≥12° away from both rectified block axes and extending at least two blocks is inconsistent with a full block edge. Both sides must independently favor the same level and repeat the same texels. Missing or invalid camera evidence, insufficient texture, physical corners and ambiguous chains retain the original masks. Its only edit across 204 photos was the previously identified Facade diagonal. Every decoded output contains the same block positions as the baseline. The 200 original serialized packs also match byte for byte. It improved the actual wall support, with 31,234 more correctly paired pixels and one fewer stray mask. No independent photo supplied a second positive example. The helper is suitable as a narrow option if the normal Surface Finder path can obtain a calibrated image camera economically. Compare it to the independently tested same-parent outline rule before choosing a default. [Production-capable helper](../../../engine/diagonal_texture_join.py) · [Timed 204-frame replay](timing.json)

Treat fragmentation as a dedicated surface-stage diagnostic and a small component of the geometry error budget. Preserve observed-only mask unions. For another experiment, target pixels beyond the two detected Portico pieces and compare the depth and extent of the uncovered visible roof through reprojection. Merging the already detected areas leaves that portion uncovered.

## Replay

The following commands use the shared frozen prior (ordinary placer at edb10a3) and its saved complete-structure captures. Run the adapters through the evaluator from the strip-orientation-audit checkout at 59ac564. The original texture method was replayed first on the 72 development photos. The material change and diagonal follow-ups were informed by specific inspected cases. The diagonal rule was frozen after the development replay and before the final 204-frame run. The timed replay includes branch decisions and creating any accepted mask union, with inputs in RAM. Camera inference, model inference, block recount and placement are excluded.

```powershell
python -m benchmark.coplanar_fragment_audit --workers 5
cd ../strip-orientation-audit
python -m benchmark.strip_orientation_eval --candidate-root ../coplanar-fragments --candidate benchmark.coplanar_fragment_join:photograph --out ../coplanar-fragments/runs/photo --workers 5
python -m benchmark.strip_orientation_eval --candidate-root ../coplanar-fragments --candidate benchmark.coplanar_fragment_intervention:ideal --out ../coplanar-fragments/runs/ideal --workers 5
python -m benchmark.strip_orientation_eval --candidate-root ../coplanar-fragments --candidate benchmark.coplanar_fragment_join:material_transition --out ../coplanar-fragments/runs/transition --workers 5
python -m benchmark.strip_orientation_eval --candidate-root ../coplanar-fragments --candidate benchmark.coplanar_diagonal_trial:predict --out ../coplanar-fragments/runs/diagonal --workers 5
$adapters = @{photo='benchmark.coplanar_fragment_join:photograph'; transition='benchmark.coplanar_fragment_join:material_transition'; diagonal='benchmark.coplanar_diagonal_trial:predict'}
foreach ($name in $adapters.Keys) {
    python -m benchmark.strip_orientation_eval --transfer-inputs ../framed-depth-transfer/runs/transfer-input --candidate-root ../coplanar-fragments --candidate $adapters[$name] --out "../coplanar-fragments/runs/transfer-$name"
}
cd ../coplanar-fragments
python -m benchmark.coplanar_fragment_report --workers 5
python -m benchmark.coplanar_shopfront_probe
python -m benchmark.coplanar_fragment_timing --workers 5
```
