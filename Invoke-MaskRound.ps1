[CmdletBinding()]
param(
    [string] $Python,
    [string] $Fields,
    [string] $Lines,
    [string] $Out,
    [ValidateSet('full', 'first')][string] $Phase = 'full'
)

$ErrorActionPreference = 'Stop'
$previousMode = $env:MCIR_IMAGE_ONLY
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:MCIR_IMAGE_ONLY = '1'
    if ($Out -and (Test-Path -LiteralPath $Out)) {
        throw "Output already exists. Choose a new run directory: $Out"
    }
    $options = if ($Python) { @($Python) } else {
        @((Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
          (Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'),
          'python')
    }
    $interpreter = $null
    $base = $null
    foreach ($name in $options) {
        $app = Get-Command -Name $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $app) { continue }
        & $app.Source -c 'import sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,13) else 1)' 2>$null
        if ($LASTEXITCODE -ne 0) { continue }
        if (-not $base) { $base = $app.Source }
        & $app.Source -c "import sys, numpy, PIL; sys.exit(0 if (numpy.__version__,PIL.__version__)==('2.3.5','13.0.0') else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $interpreter = $app.Source; break }
    }
    if (-not $interpreter) {
        if (-not $base) { throw 'This replay needs Python 3.11, 3.12 or 3.13. Use -Python to select it.' }
        $venv = Join-Path $PSScriptRoot ("runs/mask-python-" + [guid]::NewGuid().ToString('N'))
        & $base -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the run-specific Python environment.' }
        $interpreter = Join-Path $venv 'Scripts/python.exe'
        & $interpreter -m pip install --disable-pip-version-check --only-binary ':all:' 'numpy==2.3.5' 'Pillow==13.0.0'
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the pinned NumPy and Pillow wheels.' }
    }
    $arguments = @('-m', 'benchmark.mask_replay', '--phase', $Phase)
    if ($Fields) { $arguments += @('--fields', $Fields) }
    if ($Lines)  { $arguments += @('--lines', $Lines) }
    if ($Out)    { $arguments += @('--out', $Out) }
    & $interpreter @arguments
    if ($LASTEXITCODE -ne 0) { throw "The mask comparison exited with code $LASTEXITCODE." }
} finally {
    $env:MCIR_IMAGE_ONLY = $previousMode
    Pop-Location
}
