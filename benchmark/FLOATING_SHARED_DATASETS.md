# Shared Floating datasets

Use Floating Front 100 for new sparse replays. Each of its 100 layouts has a full-block noon, morning and afternoon capture and a decorative noon capture. All four captures of a layout share its native camera geometry and frozen train, validation or test assignment. The clean lighting variants share visible geometry and materials. Java can render the same RGB at two recorded times. Each capture retains its own frame ID and camera record.

| Launcher choice | Views |
| --- | --- |
| `front-full` (default) | 100 full-block noon views by default. Select `morning`, `afternoon` or `all` for the paired captures. |
| `front-decorative` | 100 decorative noon views, scored separately. |
| `visible` | The historical 200 active Visible Builds views in Floating Combined V1. |
| `decorative` | The historical 40 active Decorative Builds V1 views. |

Original V1 and the archived historical placements, lighting pairs and synthetic backgrounds retain their pinned releases and entry points.

## Create a sparse replay worktree

Run from the reconstruction repository. Point `$revision` at the commit or branch containing the Front 100 release pin. Use a new checkout path. A full checkout can also access the archived review artifacts.

```powershell
$checkout = 'C:/code/minecraft/minecraft-image-reconstruction-worktrees/floating-replay'
git worktree add --no-checkout --detach $checkout $revision
git -C $checkout sparse-checkout set --no-cone -- '/**' '!/benchmark/reviews/**'
git -C $checkout reset --hard HEAD
Set-Location $checkout
```

## Restore Front 100 once

The dataset uses the stock Java 26.3 atlas. The shared root on Jeremy's PC already contains the prepared atlas under `atlas`. A different prepared atlas location can be configured with `MCIR_PREPARED_ATLAS_26_3`.

```powershell
$shared = 'C:/code/minecraft/workspace/artifacts/reconstruction-fixtures'
$archives = "$shared/releases"
$dataset = 'floating-front-100-v1'
$fixture = "$shared/floating-front-100-v1-java26.3"
gh release download $dataset --repo jeremyepling/minecraft-image-reconstruction `
    --dir "$archives/$dataset" --skip-existing --pattern manifest.json --pattern "${dataset}-*.zip"
if ($LASTEXITCODE) { throw "Release download failed for $dataset" }
python -B -m training.floating_shared_inputs --archives-root $archives --fixture-root $fixture --dataset $dataset
if ($LASTEXITCODE) { throw "Fixture setup failed for $dataset" }
```

The installer verifies the manifest against the checkout's SHA-256 pin, verifies all four capture shards, frame hashes and the Java structure archive, then publishes a read-only `frames/floating-front-100-v1` directory. It verifies an existing installation before reusing it. The capture shards retain the server readback and any additional records listed for a frame. The staged fixture holds RGB, the scorer's camera and first-hit visible labels, and `floating-front-100-v1-structures.zip` with 200 Java structure templates. A replay verifies each selected frame before creating job outputs. Reconstruction receives the capture RGB and game version. The evaluator uses the visible labels and camera after it decodes the resulting pack.

## Replay Front 100

Use a private `-RunDirectory` per job. Omit `-Case` for the entire selection. Use `-Case` with one or more exact frame IDs from the release manifest to freeze a focused comparison. Separate runs may share the staged atlas and captures. When an archived model field has the same RGB, models, preprocessing and field hashes, pass it with `-FieldArchive` to reuse it read-only. New fields are generated inside the private job.

On this PC the prepared model and Python add-on directories are under `$shared/runtime`.

```powershell
$runtime = "$shared/runtime"
$models = "$runtime/surface-models"
$benchmarkPython = 'C:/Users/jerem/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$env:PYTHONPATH = "$runtime/onnxruntime-1.23.2;$runtime/opencv-4.12.0.88"
.\Invoke-FloatingReplay.ps1 -FixtureRoot $fixture -ModelRoot $models `
    -RunDirectory $fullNoonOutput -Python $benchmarkPython
.\Invoke-FloatingReplay.ps1 -FixtureRoot $fixture -Cohort front-full -Lighting all -ModelRoot $models `
    -RunDirectory $lightingOutput -Python $benchmarkPython
.\Invoke-FloatingReplay.ps1 -FixtureRoot $fixture -Cohort front-decorative -ModelRoot $models `
    -RunDirectory $decorativeOutput -Python $benchmarkPython
```

Each run saves its exact selection in `fixtures.json` beside the case scores, datapack exports and `summary.json`. The summary includes per-lighting scores for Front 100. Pass a pair of full-block runs and their fixture to the [Floating review publisher](MOBILE_REVIEW.md) for a paired review. Decorative scores are available from the separate replay summary.

## Prepare a release from native captures

Use the finalized private capture directories as input. Repeat `--captures-root` for disjoint batches. The packager requires all 100 layout pairs and writes `manifest.json`, four capture shards and the structure archive to a new directory. It verifies the camera pairs, clean lighting labels, split assignments and capture checksums. Keep the release directory name `floating-front-100-v1` under the staging archive parent.

```powershell
python -B -m training.floating_front_release --captures-root $batchA --captures-root $batchB `
    --out "$archives/$dataset"
```

The packager prints the release manifest SHA-256. During release preparation, pass that verified digest as `--release-sha256` to the installer and `-ReleaseSha256` to the launcher. Save the verified digest in `training/floating-front-100-v1.sha256` when freezing the release. The checkout pin then verifies every installation and replay.

## Replay historical Combined V1

On Jeremy's PC, use `$shared/floating-combined-v1-32d151f5-java26.3`. Pass the original archive parent as `--archives-root`. Each release subdirectory there contains its original `manifest.json` and shard ZIPs. The checked-in [Combined V1 index](../training/FLOATING_DATASETS.md) pins the inputs and active framing decisions.

```powershell
$historicalFixture = "$shared/floating-combined-v1-32d151f5-java26.3"
foreach ($dataset in @('visible-builds-v2', 'visible-builds-v3', 'decorative-builds-v1')) {
    python -B -m training.floating_shared_inputs --archives-root $historicalArchives --fixture-root $historicalFixture --dataset $dataset
    if ($LASTEXITCODE) { throw "Fixture setup failed for $dataset" }
}
$fields = "$shared/visible-builds-v1-a702f1d1-java26.3/fields"
.\Invoke-FloatingReplay.ps1 -Cohort visible -FixtureRoot $historicalFixture `
    -FieldArchive $fields -ModelRoot $models -RunDirectory $visibleOutput -Python $benchmarkPython
.\Invoke-FloatingReplay.ps1 -Cohort decorative -FixtureRoot $historicalFixture `
    -ModelRoot $models -RunDirectory $historicalDecorativeOutput -Python $benchmarkPython
```
