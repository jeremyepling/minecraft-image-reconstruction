# Shared Floating datasets

Use two active replay choices on the Floating integration branch. Both use the same sparse worktree and shared Java 26.3 input directory.

| Launcher choice | Inputs |
| --- | --- |
| `visible` | The 200 active Visible Builds views in Floating Combined V1, with 92 from V2 and 108 from V3 |
| `decorative` | The 40 active Decorative Builds V1 views, reported as their own cohort |

The checked-in [Combined V1 index](../training/FLOATING_DATASETS.md) supplies the frame IDs, splits, RGB hashes, visible-reference hashes and framing decisions. The launcher checks that the chosen cohort is available before running. Original V1, historical placements, lighting pairs and synthetic backgrounds use their historical entry points and pinned archives.

## Create a sparse replay worktree

Run from the reconstruction repository. Choose an unused worktree path. A review session can use a full checkout to access its saved review artifacts.

```powershell
$checkout = 'C:/code/minecraft/minecraft-image-reconstruction-worktrees/floating-replay'
git worktree add --no-checkout --detach $checkout experiment/floating-visible-builds
git -C $checkout sparse-checkout set --no-cone -- '/**' '!/benchmark/reviews/**'
git -C $checkout restore --source=HEAD --staged --worktree -- .
```

## Prepare shared inputs once

On Jeremy's PC, use `C:/code/minecraft/workspace/artifacts/reconstruction-fixtures/floating-combined-v1-32d151f5-java26.3`. Its `atlas` folder uses the prepared Java 26.3 atlas. Pass the pinned dataset release parent as `--archives-root`. Each dataset directory there must contain its original `manifest.json` and shard ZIPs.

```powershell
$shared = 'C:/code/minecraft/workspace/artifacts/reconstruction-fixtures'
$fixture = "$shared/floating-combined-v1-32d151f5-java26.3"
foreach ($dataset in @('visible-builds-v2', 'visible-builds-v3', 'decorative-builds-v1')) {
    python -B -m training.floating_shared_inputs --archives-root $archives --fixture-root $fixture --dataset $dataset
    if ($LASTEXITCODE) { throw "Fixture setup failed for $dataset" }
}
```

The installer verifies each release and each consumed frame file against the index, then publishes a read-only directory under `frames/<release>`. Its receipt pins the index and release versions. Source RGB goes to reconstruction. The scorer loads camera and visible-label files after pack decoding. Missing release inputs stop the chosen run before it creates job outputs.

## Replay either cohort

Use a different `-RunDirectory` per job. Omit `-Case` for the chosen full cohort. Earlier source fields can be reused when the RGB, model, preprocessing and field hashes match. New fields are generated inside the run with the pinned normal and depth models. Provide the model directory using `-ModelRoot` if the models are outside the checkout. On this PC, add `C:/code/minecraft/workspace/artifacts/reconstruction-fixtures/runtime/opencv-4.12.0.88` to `PYTHONPATH` when using the bundled Codex Python.

```powershell
$fields = "$shared/visible-builds-v1-a702f1d1-java26.3/fields"
.\Invoke-FloatingReplay.ps1 -Cohort visible -FixtureRoot $fixture `
    -FieldArchive $fields -ModelRoot $models -RunDirectory $visibleOutput -Python $benchmarkPython
.\Invoke-FloatingReplay.ps1 -Cohort decorative -FixtureRoot $fixture `
    -ModelRoot $models -RunDirectory $decorativeOutput -Python $benchmarkPython
```

Each run saves its selected fixture cases beside the scores and packs. The same manifest and run directory can be passed to the [Floating review publisher](MOBILE_REVIEW.md).
