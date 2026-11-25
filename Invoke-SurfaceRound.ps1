[CmdletBinding()]
param([string] $Python, [string] $Out, [int] $Workers = 3,
      [ValidateSet('surfaces', 'normals', 'saved', 'building-sheet')] [string] $Round = 'surfaces')

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
try {
    $module = @{
        surfaces = 'benchmark.surface_round'
        normals = 'benchmark.normal_round'
        saved = 'benchmark.score_saved_planes'
        'building-sheet' = 'benchmark.normal_building_sheet'
    }[$Round]
    $imports = 'import numpy, PIL'
    if ($Round -eq 'normals') {
        $imports = 'import numpy, PIL, onnxruntime'
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
            if ($Round -eq 'normals') {
                & $exe.Source -m benchmark.normal_assets --download
                if ($LASTEXITCODE -ne 0) { throw 'Failed to prepare the pinned local model files.' }
            }
            $arguments = @('-m', $module)
            if ($Round -in 'surfaces', 'normals') { $arguments += @('--workers', $Workers) }
            if ($Round -eq 'saved' -and $Out) { throw 'Saved scoring prints the per-building results to the terminal.' }
            if ($Out) { $arguments += @('--out', $Out) }
            & $exe.Source @arguments
            if ($LASTEXITCODE -ne 0) { throw "The surface round exited with code $LASTEXITCODE" }
            return
        }
    }
    throw 'Install the suite requirements (plus onnxruntime==1.23.2 for normals) and pass that Python executable with -Python.'
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
