[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $FixtureRoot,
    [Parameter(Mandatory)][string] $RunDirectory,
    [ValidateSet('front-full', 'front-decorative', 'visible', 'decorative')][string] $Cohort = 'front-full',
    [string[]] $Case,
    [string] $FieldArchive,
    [string] $ModelRoot,
    [string] $Python = 'python',
    [ValidateRange(1, 128)][int] $Workers = 1,
    [ValidateSet('noon', 'morning', 'afternoon', 'all')][string] $Lighting,
    [ValidatePattern('^[0-9a-f]{64}$')][string] $ReleaseSha256
)

$ErrorActionPreference = 'Stop'
$fixture = (Resolve-Path -LiteralPath $FixtureRoot).ProviderPath
$run = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RunDirectory)
$arguments = @('-B', '-m', 'benchmark.floating_expanded_run', '--shared-root', $fixture,
    '--out', $run, '--cohort', $Cohort, '--workers', $Workers)
if ($FieldArchive) { $arguments += @('--baseline', (Resolve-Path -LiteralPath $FieldArchive).ProviderPath) }
if ($Lighting) { $arguments += @('--lighting', $Lighting) }
if ($ReleaseSha256) { $arguments += @('--release-sha256', $ReleaseSha256) }
foreach ($frame in $Case) { $arguments += @('--case', $frame) }

$previousImageOnly = $env:MCIR_IMAGE_ONLY
$previousNormal = $env:MINECRAFT_MOGE_ONNX
$previousDepth = $env:MINECRAFT_DEPTH_ONNX
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:MCIR_IMAGE_ONLY = '1'
    if ($ModelRoot) {
        $models = (Resolve-Path -LiteralPath $ModelRoot).ProviderPath
        $env:MINECRAFT_MOGE_ONNX = Join-Path $models 'moge-2-vits.onnx'
        $env:MINECRAFT_DEPTH_ONNX = Join-Path $models 'depth-v2-small-uint8.onnx'
    }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Floating replay failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
    $env:MCIR_IMAGE_ONLY = $previousImageOnly
    $env:MINECRAFT_MOGE_ONNX = $previousNormal
    $env:MINECRAFT_DEPTH_ONNX = $previousDepth
}
