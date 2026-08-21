[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$webPython = Join-Path $resolvedProjectRoot "venv_docnorm_web\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $webPython -PathType Leaf)) {
    throw "venv_docnorm_web is missing."
}

$testId = [Guid]::NewGuid().ToString("N").Substring(0, 10)
$baseTemp = Join-Path $env:LOCALAPPDATA "DocNormMD\tests\$testId"
$env:DOCNORM_PROJECT_ROOT = $resolvedProjectRoot
$env:DOCNORM_PIPELINE_ROOT = Join-Path $resolvedProjectRoot "pipeline_v2"
$env:DOCNORM_ODL_PYTHON = Join-Path $resolvedProjectRoot "venv_opendataloader\Scripts\python.exe"
$env:DOCNORM_ODL_WORKER = Join-Path $resolvedProjectRoot "pipeline_v2\src\rag_pipeline\workers\opendataloader_local_worker.py"
$env:DOCNORM_ODL_HYBRID_WORKER = Join-Path $PSScriptRoot "docnorm_web\opendataloader_hybrid_document_worker.py"
$env:DOCNORM_ODL_HYBRID_EXECUTABLE = Join-Path $resolvedProjectRoot "venv_opendataloader\Scripts\opendataloader-pdf-hybrid.exe"
$env:DOCNORM_ODL_HYBRID_URL = "http://127.0.0.1:5002"
$env:DOCNORM_VLM_PYTHON = Join-Path $resolvedProjectRoot "venv\Scripts\python.exe"
$env:DOCNORM_MATH_WORKER = Join-Path $PSScriptRoot "docnorm_web\qwen_math_worker.py"
$env:DOCNORM_MATH_MAX_VLM_PAGES = "8"
$env:DOCNORM_SCAN_PYTHON = Join-Path $resolvedProjectRoot "venv\Scripts\python.exe"
$env:DOCNORM_SCAN_WORKER = Join-Path $PSScriptRoot "docnorm_web\easyocr_geometry_worker.py"
$env:DOCNORM_RUNS_ROOT = Join-Path $baseTemp "runs"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Push-Location $PSScriptRoot
try {
    & $webPython -m pytest --basetemp $baseTemp
    if ($LASTEXITCODE -ne 0) { throw "DocNorm MD tests failed." }
}
finally {
    Pop-Location
}

Write-Host "DOCNORM_MD_SMOKE=PASS" -ForegroundColor Green
