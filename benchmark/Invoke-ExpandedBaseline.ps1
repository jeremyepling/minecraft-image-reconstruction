param(
    [int] $Workers = 7,
    [int] $GpuSlots = 3,
    [ValidateSet('cuda','cpu')][string] $Device = 'cuda',
    [string[]] $PriorityIds = @(),
    [string[]] $FrameIds = @()
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$worktrees = Split-Path -Parent $root
$scale = Join-Path $worktrees 'mask-scalelsd/runs'
$planarity = Join-Path $worktrees 'afterpass-planarity/runs/assets'
$models = Join-Path $worktrees 'surface-strip-entry/.deps'
$runtime = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$settings = @{
    MCIR_RECOVERY_PYTHON = Join-Path $scale 'env/Scripts/python.exe'
    MCIR_SCALELSD_SOURCE = Join-Path $scale 'upstream'
    MCIR_SCALELSD_CHECKPOINT = Join-Path $scale 'scalelsd-vitbase-v2-train-sa1b.pt'
    MCIR_MOBILESAM_SOURCE = Join-Path $worktrees 'surface-recovery-round/runs/mobile-sam-source'
    MCIR_PLANARITY_SOURCE = Join-Path $planarity 'PixelwisePlanarity'
    MCIR_PLANARITY_UTILS3D = Join-Path $planarity 'utils3d'
    MCIR_PLANARITY_CHECKPOINT = Join-Path $planarity 'model.pt'
    MCIR_PLANARITY_PYTHONPATH = Join-Path $planarity 'deps'
    MINECRAFT_MOGE_ONNX = Join-Path $models 'moge-2-vits.onnx'
    MINECRAFT_DEPTH_ONNX = Join-Path $models 'depth-v2-small-uint8.onnx'
    MINECRAFT_MLSD_TFLITE = Join-Path $models 'M-LSD_512_tiny_fp16.tflite'
    PYTHONPATH = @($env:PYTHONPATH, (Join-Path $worktrees 'surface-round/runs/depth-probe/runtime'),
                   (Join-Path $worktrees 'surface-round/runs/mlsd-probe/lib')) -ne '' -join [IO.Path]::PathSeparator
    MCIR_RECOVERY_DEVICE = $Device
    MCIR_IMAGE_ONLY = '1'
    OPENBLAS_NUM_THREADS = '1'
    OMP_NUM_THREADS = '1'
    MKL_NUM_THREADS = '1'
}

$options = @('--workers', $Workers, '--gpu-slots', $GpuSlots)
if ($PriorityIds) { $options += @('--priority-ids') + $PriorityIds }
if ($FrameIds) { $options += @('--ids') + $FrameIds }
$previous = @{}
Push-Location -LiteralPath $root
try {
    foreach ($name in $settings.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name)
        [Environment]::SetEnvironmentVariable($name, $settings[$name])
    }
    & $runtime -B -m benchmark.expanded_baseline @options
    if ($LASTEXITCODE -ne 0) { throw "Expanded build batch exited with code $LASTEXITCODE" }
} finally {
    foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name, $previous[$name]) }
    Pop-Location
}
