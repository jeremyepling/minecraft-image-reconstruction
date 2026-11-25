[CmdletBinding()]
param([string] $Python, [string] $Atlas, [string] $Out)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $candidates = if ($Python) { @($Python) } else {
        @((Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
          (Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'),
          'python')
    }
    $interpreter = $null
    foreach ($name in $candidates) {
        $app = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $app) { continue }
        & $app.Source -c "import sys,numpy,PIL; sys.exit(0 if (numpy.__version__,PIL.__version__)==('2.3.5','13.0.0') else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $interpreter = $app.Source; break }
    }
    if (-not $interpreter) { throw 'Install benchmark/suites/requirements.txt and select that interpreter with -Python.' }
    $opts = @('-B','-m','benchmark.count_round')
    if ($Out) { $opts += @('--out',$Out) }
    if ($Atlas) { $opts += @('--atlas',$Atlas) }
    & $interpreter @opts
    if ($LASTEXITCODE -ne 0) { throw "Block count replay exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
