[CmdletBinding()]
param(
    [Parameter(Mandatory, Position=0)][string] $Image,
    [Parameter(Mandatory)][string] $Out,
    [ValidateSet('surfaces','counts')][string] $Through = 'counts',
    [string] $Python,
    [string] $ModelPython,
    [string] $ScaleSource,
    [string] $ScaleCheckpoint,
    [string] $MobileSource,
    [string] $DepthProSource,
    [string] $DepthProCheckpoint,
    [ValidateSet('wall-recovery','wall-recovery-single-v2','wall-recovery-single-pro')][string] $SurfaceBoundary,
    [ValidateSet('cuda','cpu')][string] $Device
)

$ErrorActionPreference = 'Stop'
$sourceImage = (Resolve-Path -LiteralPath $Image).Path
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
$scaleRun = Join-Path $PSScriptRoot '../mask-scalelsd/runs'
$localDeps = Join-Path $PSScriptRoot '.deps'
$installedModels = Join-Path $PSScriptRoot '../surface-strip-entry/.deps'
$changes = @{}

function Set-ModelOption([string]$Name, [string]$Explicit, [string[]]$Candidates) {
    $original = [Environment]::GetEnvironmentVariable($Name)
    $value = $Explicit
    if (!$value) { $value = $original }
    if (!$value) { $value = $Candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1 }
    if ($value) {
        if ($Name -ne 'MCIR_RECOVERY_DEVICE' -and (Test-Path -LiteralPath $value)) {
            $value = (Resolve-Path -LiteralPath $value).Path
        }
        $changes[$Name] = $original
        [Environment]::SetEnvironmentVariable($Name, $value)
    }
}

try {
    if (!$Python) {
        $Python = @((Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
                    (Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')) |
            Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if (!$Python) { $Python = 'python' }
    }
    Set-ModelOption 'MCIR_RECOVERY_PYTHON' $ModelPython @((Join-Path $scaleRun 'env/Scripts/python.exe'))
    Set-ModelOption 'MCIR_SCALELSD_SOURCE' $ScaleSource @((Join-Path $localDeps 'scalelsd'), (Join-Path $scaleRun 'upstream'))
    $checkpointName = 'scalelsd-vitbase-v2-train-sa1b.pt'
    Set-ModelOption 'MCIR_SCALELSD_CHECKPOINT' $ScaleCheckpoint @((Join-Path $localDeps $checkpointName), (Join-Path $scaleRun $checkpointName))
    Set-ModelOption 'MCIR_MOBILESAM_SOURCE' $MobileSource @((Join-Path $localDeps 'MobileSAM'), (Join-Path $PSScriptRoot '../surface-recovery-round/runs/mobile-sam-source'))
    $depthProDeps=Join-Path $PSScriptRoot '../surface-one-depth-pro/.deps'
    Set-ModelOption 'MCIR_DEPTH_PRO_SOURCE' $DepthProSource @((Join-Path $localDeps 'ml-depth-pro'), (Join-Path $depthProDeps 'ml-depth-pro'))
    Set-ModelOption 'MCIR_DEPTH_PRO_CHECKPOINT' $DepthProCheckpoint @((Join-Path $localDeps 'depth_pro.pt'), (Join-Path $depthProDeps 'depth_pro.pt'))
    Set-ModelOption 'MCIR_RECOVERY_DEVICE' $Device @()
    foreach ($model in @{
        MINECRAFT_MOGE_ONNX='moge-2-vits.onnx'
        MINECRAFT_DEPTH_ONNX='depth-v2-small-uint8.onnx'
        MINECRAFT_MLSD_TFLITE='M-LSD_512_tiny_fp16.tflite'
    }.GetEnumerator()) {
        Set-ModelOption $model.Key '' @((Join-Path $localDeps $model.Value), (Join-Path $installedModels $model.Value))
    }
    $extra = @('../surface-round/runs/depth-probe/runtime','../surface-round/runs/mlsd-probe/lib') |
        ForEach-Object { Join-Path $PSScriptRoot $_ } | Where-Object { Test-Path -LiteralPath $_ }
    if ($extra) {
        $changes['PYTHONPATH'] = $env:PYTHONPATH
        $env:PYTHONPATH = (@($env:PYTHONPATH) + @($extra) | Where-Object { $_ }) -join [IO.Path]::PathSeparator
    }
    $options=@()
    if ($SurfaceBoundary) { $options+=@('--surface-boundary',$SurfaceBoundary) }
    & $Python (Join-Path $PSScriptRoot 'reconstruct.py') build $sourceImage --version 26.3 --through $Through @options --out $destination
    if ($LASTEXITCODE -ne 0) { throw "Wall Recovery exited with code $LASTEXITCODE" }
} finally {
    foreach ($name in $changes.Keys) { [Environment]::SetEnvironmentVariable($name, $changes[$name]) }
}
