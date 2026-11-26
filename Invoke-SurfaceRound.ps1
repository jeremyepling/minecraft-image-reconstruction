[CmdletBinding()]
param([string] $Python, [string] $Out, [int] $Workers = 3,
      [ValidateSet('surfaces', 'normals', 'saved', 'building-sheet', 'candidate', 'compare', 'perimeters', 'detection-sheet', 'separation', 'recovery', 'recovery-summary', 'single-wall', 'ledge-join', 'afterpass', 'acceptance', 'guard-zoom', 'crop-split', 'contained', 'crop-review')] [string] $Round = 'surfaces',
      [string] $Name = 'candidate', [string] $Outlines, [string] $Pitch, [string] $Order,
      [string] $Split, [string] $Fields, [string[]] $Runs, [string] $SourceRun)

$ErrorActionPreference = 'Stop'
$options = @($Python)
if (-not $Python) {
    $options = @(
        (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
        'python',
        (Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
    )
}

Push-Location -LiteralPath $PSScriptRoot
$previousPythonPath = $env:PYTHONPATH
$previousImageOnly = $env:MCIR_IMAGE_ONLY
try {
    $env:MCIR_IMAGE_ONLY = '1'
    $module = @{
        surfaces = 'benchmark.surface_round'
        normals = 'benchmark.normal_round'
        saved = 'benchmark.score_saved_planes'
        'building-sheet' = 'benchmark.normal_building_sheet'
        candidate = 'benchmark.compare_surface_methods'
        compare = 'benchmark.compare_surface_methods'
        perimeters = 'benchmark.perimeter_round'
        'detection-sheet' = 'benchmark.perimeter_sheet'
        separation = 'benchmark.split_round'
        recovery = 'benchmark.recovery_round'
        'recovery-summary' = 'benchmark.recovery_summary'
        'single-wall' = 'benchmark.single_wall_round'
        'ledge-join' = 'benchmark.ledge_join_round'
        afterpass = 'benchmark.afterpass_round'
        acceptance = 'benchmark.plane_acceptance_round'
        'guard-zoom' = 'benchmark.guard_zoom_round'
        'crop-split' = 'benchmark.crop_split_round'
        contained = 'benchmark.contained_round'
        'crop-review' = 'benchmark.crop_review_sheet'
    }[$Round]
    $imports = 'import numpy, PIL'
    if ($Round -in 'perimeters', 'detection-sheet', 'separation', 'recovery','recovery-summary','single-wall','ledge-join','afterpass','acceptance','guard-zoom','crop-split','contained','crop-review') {
        $imports = "import sys,numpy,PIL; sys.exit(0 if (numpy.__version__,PIL.__version__)==('2.3.5','13.0.0') else 1)"
    }
    $needsInference = $Round -in 'normals','separation'
    if ($Round -eq 'recovery') {
        $cachePath = if ($Fields) { $Fields } else { Join-Path $PSScriptRoot '../normal-round/runs/normal-fields' }
        $sourceFrames = (Get-Content -Raw 'benchmark/reviews/separation-01/manifest.json' | ConvertFrom-Json).frames
        $needsInference = [bool]($sourceFrames | Where-Object {
            -not (Test-Path -LiteralPath (Join-Path $cachePath ($_.sha256+'.npz')))
        } | Select-Object -First 1)
    }
    if ($Round -eq 'candidate' -and -not $Fields) {
        $defaults = Get-Content -LiteralPath 'engine/surface_defaults.json' -Raw | ConvertFrom-Json
        $activeOutline = if ($Outlines) { $Outlines } else { $defaults.outlines }
        $activeOrder = if ($Order) { $Order } else { $defaults.order }
        $needsInference = $activeOutline -like 'normal-*' -or $activeOrder -eq 'depth-anything'
    }
    if ($needsInference) {
        $imports = "import onnxruntime; $imports"
        $probeRuntime = Join-Path $PSScriptRoot '../surface-round/runs/depth-probe/runtime'
        if (Test-Path -LiteralPath $probeRuntime) {
            $env:PYTHONPATH = (@($previousPythonPath, $probeRuntime) | Where-Object { $_ }) -join [IO.Path]::PathSeparator
        }
    }
    foreach ($candidate in $options) {
        $exe = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue -TotalCount 1
        if (-not $exe) { continue }
        & $exe.Source -c $imports 2>$null
        if ($LASTEXITCODE -eq 0) {
            if ($needsInference) {
                & $exe.Source -m benchmark.normal_assets --download
                if ($LASTEXITCODE -ne 0) { throw 'Failed to prepare the pinned local model files.' }
            }
            $arguments = @('-m', $module)
            if ($Round -eq 'recovery-summary') { $arguments += '--sheet' }
            if ($Round -eq 'detection-sheet' -and $SourceRun) { $arguments += @('--run', $SourceRun) }
            if ($Round -in 'perimeters','separation','recovery','afterpass','guard-zoom' -and $Fields) { $arguments += @('--fields', $Fields) }
            if ($Round -in 'surfaces', 'normals','separation','recovery','recovery-summary','single-wall','ledge-join','afterpass','acceptance','guard-zoom','crop-split','contained') { $arguments += @('--workers', $Workers) }
            if ($Round -eq 'candidate') {
                $arguments += @('run', '--name', $Name)
                $stageOptions = @{outlines=$Outlines; pitch=$Pitch; order=$Order; split=$Split; fields=$Fields}
                foreach ($option in $stageOptions.GetEnumerator()) {
                    if ($option.Value) { $arguments += @("--$($option.Key)", $option.Value) }
                }
            }
            if ($Round -eq 'compare') { $arguments += @('compare') + $Runs }
            if ($Round -in 'saved', 'compare' -and $Out) {
                & $exe.Source @arguments | Tee-Object -FilePath $Out
            } else {
                if ($Out) { $arguments += @('--out', $Out) }
                & $exe.Source @arguments
            }
            if ($LASTEXITCODE -ne 0) { throw "The surface round exited with code $LASTEXITCODE" }
            return
        }
    }
    throw 'Install the suite requirements (plus onnxruntime==1.23.2 for normals) and pass that Python executable with -Python.'
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:MCIR_IMAGE_ONLY = $previousImageOnly
    Pop-Location
}
