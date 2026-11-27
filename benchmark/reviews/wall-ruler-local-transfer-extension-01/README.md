# Frozen wall-ruler extension, November 27, 2026

**Keep the local transfer rule experimental.** On eight additional screenshots, the local rules reproduced the ordinary block coordinates and IDs exactly. Four views offered certified cells and four abstained. On Hillside Chapel the whole-wall arm recovered two correctly typed positions and added zero extras. No other whole-wall export changed. See the [original 13-view analysis and selection history](../wall-ruler-local-transfer-01/README.md).

The pairs below are pooled strict / facade F1. Position-plus-ID uses the same geometry-selected translation. All eight empty-hook controls matched the released decoded inventories and scores. Only one camera abstained. Castle 03 rear oblique exported zero blocks, with 4,822 visible misses counted in every arm. Every supported camera exported blocks. Across the seven camera-supported views ordinary and all local methods scored 57.21 / 57.75 placement F1.

| Coverage | Ordinary placement | Whole wall placement | Ordinary position + ID | Whole wall position + ID |
|---|---|---|---|---|
| Fixed 2 / 104 | 92.50 / 97.15 | same | 87.28 / 94.84 | same |
| Visible Builds v3 4 / 144 | 30.84 / 30.79 | same | 25.02 / 25.06 | same |
| Decorative Builds v1 2 / 40 | 56.53 / 59.02 | 56.71 / 59.21 | 43.53 / 45.74 | 43.73 / 45.95 |
| All 8 | 38.66 / 38.75 | 38.68 / 38.77 | 32.13 / 32.51 | 32.15 / 32.54 |

Ordinary, measured tiles, connected supplementation and connected overlap replacement each produced 2,875 correct positions, 7,804 misses and 1,318 extras. Exact IDs matched on 2,389 correct positions (83.10%), with 4,193 assigned IDs and zero fallbacks. Whole wall produced 2,877 correct positions, 7,802 misses and 1,318 extras. Exact IDs matched on 2,391 correct positions, with 4,195 assigned IDs and zero fallbacks.

| Screenshot | Accepted rulers | Native connected footprints | Whole-wall correct gained / lost, extras added / removed | All local coordinate deltas |
|---|---:|---:|---|---|
| Twin Spire 3 cool high right | 6 | 48 | 0/0/0/0 | 0/0/0/0 |
| Galleries plaster high right | 4 | 7 | 0/0/0/0 | 0/0/0/0 |
| Harbor 01 kiln low left | 0 | 0 | 0/0/0/0 | 0/0/0/0 |
| Harbor 01 seafoam low left | 0 | 0 | 0/0/0/0 | 0/0/0/0 |
| Castle 03 ashlar low left | 0 | 0 | 0/0/0/0 | 0/0/0/0 |
| Castle 03 ashlar rear oblique | 0 | 0 | 0/0/0/0 | 0/0/0/0 |
| Hillside Chapel 2 cool high right | 4 | 63 | 2/0/0/0 | 0/0/0/0 |
| Walled Garden 2 cool low left | 2 | 2 | 0/0/0/0 | 0/0/0/0 |

The original native-single ruler scanned the same finite candidates on these eight predeclared inputs. The local admission rules and scope of replacement were frozen before inspecting the extension scores. No old strict vote on any of the four active views lay wholly within new connected support. The overlap arm reused the connected pack after seeing all its strict handoff events with identical supplemental-cell counts and zero removals. The other four receipts point to the validated no-op; their inputs had no accepted carrier. [Hillside Chapel source](validation-hillside-chapel-2-cool-high-right.jpg) · [Twin Spire source](test-twinspire3-cool-high-right.jpg) · [Per-frame receipts, references and hashes](exports.json)

## Reproduce

Use the extension manifest from the sibling `benchmark/wall-ruler-transfer-controls` checkout. The four commands after `regions` each write to private arm directories; run overlap after the connected exports are available.

```powershell
$env:MCIR_LOCAL_RULER_EXTENSION='1'
foreach ($stage in 'candidates','regions','export','whole','noop','castle-overlap') {
    python -m benchmark.wall_ruler_local_transfer $stage --split transfer
}
python -m benchmark.wall_ruler_local_summary
Remove-Item Env:MCIR_LOCAL_RULER_EXTENSION
```
