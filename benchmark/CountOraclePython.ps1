# Adapter for Invoke-WallRecovery.ps1's -Python argument, preserving its normal model environment.
& $env:MCIR_COUNT_ORACLE_PYTHON -m benchmark.material_count_oracle @args
exit $LASTEXITCODE
