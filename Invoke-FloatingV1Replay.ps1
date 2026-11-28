[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $FixtureRoot,
    [Parameter(Mandatory)][string] $RunDirectory,
    [Parameter(Mandatory)][string[]] $Case,
    [string] $Python = 'python',
    [ValidateRange(1, 128)][int] $Workers = 1
)

$ErrorActionPreference = 'Stop'
$fixture = (Resolve-Path -LiteralPath $FixtureRoot).ProviderPath
$run = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RunDirectory)
$relativeRun = [IO.Path]::GetRelativePath($fixture, $run)
if ($relativeRun -ne '..' -and $relativeRun -notmatch '^\.\.[\\/]' -and -not [IO.Path]::IsPathRooted($relativeRun)) {
    throw 'Choose a new run directory outside the fixture root.'
}

$release = Join-Path $fixture 'release'
$fields = Join-Path $fixture 'fields'
$atlas = Join-Path $fixture 'atlas'
foreach ($folder in @($release, $fields, $atlas)) {
    if (-not (Test-Path -LiteralPath $folder -PathType Container)) { throw "Missing fixture directory $folder" }
}

$null = New-Item -ItemType Directory -Path $run
$arguments = @('-B', '-m', 'benchmark.floating_v1_run', '--selective', '--release', $release,
    '--field-archive', $fields, '--cache', (Join-Path $run 'cache'), '--profile', (Join-Path $run 'profile.json'),
    '--through', 'score', '--orientation', 'charge', '--min-area', '5000', '--edge-pixels', '2.5', '--workers', $Workers)
foreach ($frame in $Case) { $arguments += @('--case', $frame) }

$previousImageOnly = $env:MCIR_IMAGE_ONLY
$previousAtlas = $env:MCIR_PREPARED_ATLAS_26_3
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:MCIR_IMAGE_ONLY = '1'
    $env:MCIR_PREPARED_ATLAS_26_3 = $atlas
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Floating replay failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
    $env:MCIR_IMAGE_ONLY = $previousImageOnly
    $env:MCIR_PREPARED_ATLAS_26_3 = $previousAtlas
}
