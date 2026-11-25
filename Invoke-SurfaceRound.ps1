[CmdletBinding()]
param([string] $Python, [string] $Out, [int] $Workers = 3)

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
try {
    foreach ($candidate in $options) {
        $exe = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        & $exe.Source -c 'import numpy, PIL' 2>$null
        if ($LASTEXITCODE -eq 0) {
            $arguments = @('-m', 'benchmark.surface_round', '--workers', $Workers)
            if ($Out) { $arguments += @('--out', $Out) }
            & $exe.Source @arguments
            if ($LASTEXITCODE -ne 0) { throw "The surface round exited with code $LASTEXITCODE" }
            return
        }
    }
    throw 'Install benchmark/suites/requirements.txt in a Python environment and pass its executable with -Python.'
} finally {
    Pop-Location
}
