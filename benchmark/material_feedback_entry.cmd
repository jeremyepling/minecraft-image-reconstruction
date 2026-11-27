@echo off
"%MCIR_BYPASS_PYTHON%" -m benchmark.material_feedback_bypasses --ordinary "%MCIR_BYPASS_METHOD%" %*
exit /b %errorlevel%
