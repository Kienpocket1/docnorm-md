[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [ValidateRange(1024, 65535)]
    [int]$Port = 8010
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$webPython = Join-Path $resolvedProjectRoot "venv_docnorm_web\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $webPython -PathType Leaf)) {
    throw "venv_docnorm_web is missing. Run install_windows.ps1 first."
}

function Test-LocalPort {
    param(
        [string]$ComputerName = "127.0.0.1",
        [int]$TargetPort
    )
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($ComputerName, $TargetPort, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(500)) { return $false }
        $client.EndConnect($async)
        return $true
    }
    catch { return $false }
    finally { $client.Dispose() }
}

$hybridPort = 5002
$hybridUrl = "http://127.0.0.1:$hybridPort"
$hybridExecutable = Join-Path $resolvedProjectRoot "venv_opendataloader\Scripts\opendataloader-pdf-hybrid.exe"
$hybridWorker = Join-Path $PSScriptRoot "docnorm_web\opendataloader_hybrid_document_worker.py"
$odlPython = Join-Path $resolvedProjectRoot "venv_opendataloader\Scripts\python.exe"
$vlmPython = Join-Path $resolvedProjectRoot "venv\Scripts\python.exe"
$mathWorker = Join-Path $PSScriptRoot "docnorm_web\qwen_math_worker.py"
$scanWorker = Join-Path $PSScriptRoot "docnorm_web\easyocr_geometry_worker.py"
$hybridProcess = $null
$hybridStartedByDocNorm = $false

$env:DOCNORM_PROJECT_ROOT = $resolvedProjectRoot
$env:DOCNORM_PIPELINE_ROOT = Join-Path $resolvedProjectRoot "pipeline_v2"
$env:DOCNORM_ODL_PYTHON = $odlPython
$env:DOCNORM_ODL_WORKER = Join-Path $resolvedProjectRoot "pipeline_v2\src\rag_pipeline\workers\opendataloader_local_worker.py"
$env:DOCNORM_ODL_HYBRID_WORKER = $hybridWorker
$env:DOCNORM_ODL_HYBRID_EXECUTABLE = $hybridExecutable
$env:DOCNORM_ODL_HYBRID_URL = $hybridUrl
$env:DOCNORM_ODL_HYBRID_TIMEOUT_MS = "3600000"
$env:DOCNORM_ODL_HYBRID_PROCESS_TIMEOUT_SECONDS = "7200"
$env:DOCNORM_VLM_PYTHON = $vlmPython
$env:DOCNORM_MATH_WORKER = $mathWorker
$env:DOCNORM_SCAN_PYTHON = $vlmPython
$env:DOCNORM_SCAN_WORKER = $scanWorker
$env:DOCNORM_MATH_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
if (-not $env:DOCNORM_MATH_MAX_VLM_PAGES) { $env:DOCNORM_MATH_MAX_VLM_PAGES = "8" }
if (-not $env:DOCNORM_MATH_TARGET_LONG_SIDE) { $env:DOCNORM_MATH_TARGET_LONG_SIDE = "1900" }
if (-not $env:DOCNORM_MATH_MAX_NEW_TOKENS) { $env:DOCNORM_MATH_MAX_NEW_TOKENS = "1800" }
if (-not $env:DOCNORM_MATH_WORKER_TIMEOUT_SECONDS) { $env:DOCNORM_MATH_WORKER_TIMEOUT_SECONDS = "3600" }
if (-not $env:DOCNORM_MATH_MIN_FREE_VRAM_MB) { $env:DOCNORM_MATH_MIN_FREE_VRAM_MB = "2800" }
if (-not $env:DOCNORM_SCAN_TARGET_LONG_SIDE) { $env:DOCNORM_SCAN_TARGET_LONG_SIDE = "2200" }
if (-not $env:DOCNORM_SCAN_WORKER_TIMEOUT_SECONDS) { $env:DOCNORM_SCAN_WORKER_TIMEOUT_SECONDS = "7200" }
$env:DOCNORM_RUNS_ROOT = Join-Path $resolvedProjectRoot "docnorm_runs"
$env:DOCNORM_HOST = "127.0.0.1"
$env:DOCNORM_PORT = [string]$Port
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TORCHDYNAMO_DISABLE = "1"
$env:TORCH_COMPILE_DISABLE = "1"

if (-not $env:DOCLING_DEVICE -and (Test-Path -LiteralPath $odlPython -PathType Leaf)) {
    $cudaReady = & $odlPython -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
    if ($LASTEXITCODE -eq 0 -and "$cudaReady".Trim() -eq "1") {
        $env:DOCLING_DEVICE = "cuda"
        $env:CUDA_VISIBLE_DEVICES = "0"
        Write-Host "Docling accelerator: CUDA" -ForegroundColor Green
    }
    else {
        $env:DOCLING_DEVICE = "cpu"
        Write-Host "Docling accelerator: CPU" -ForegroundColor Yellow
    }
}

$startHybrid = ($env:DOCNORM_START_HYBRID -eq "1")
if (Test-LocalPort -TargetPort $hybridPort) {
    if ($startHybrid) {
        Write-Host "Hybrid OCR backend already ready: $hybridUrl" -ForegroundColor Green
    }
    else {
        Write-Warning (
            "A Hybrid backend is already listening on port 5002. It may keep " +
            "several GB of VRAM; stop that old process for the fastest Geometry/Qwen jobs."
        )
    }
}
elseif (
    $startHybrid -and
    (Test-Path -LiteralPath $hybridExecutable -PathType Leaf) -and
    (Test-Path -LiteralPath $hybridWorker -PathType Leaf)
) {
    $hybridLogRoot = Join-Path $env:DOCNORM_RUNS_ROOT "_hybrid_backend"
    New-Item -ItemType Directory -Path $hybridLogRoot -Force | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $hybridStdout = Join-Path $hybridLogRoot "hybrid_${stamp}_stdout.log"
    $hybridStderr = Join-Path $hybridLogRoot "hybrid_${stamp}_stderr.log"
    Write-Host "Starting local Hybrid OCR backend (vi,en)..." -ForegroundColor Cyan
    $hybridProcess = Start-Process `
        -FilePath $hybridExecutable `
        -ArgumentList @("--port", "$hybridPort", "--force-ocr", "--ocr-lang", "vi,en") `
        -WorkingDirectory (Split-Path -Parent $hybridExecutable) `
        -RedirectStandardOutput $hybridStdout `
        -RedirectStandardError $hybridStderr `
        -WindowStyle Hidden `
        -PassThru
    $hybridStartedByDocNorm = $true
    $deadline = (Get-Date).AddSeconds(600)
    while ((Get-Date) -lt $deadline) {
        if (Test-LocalPort -TargetPort $hybridPort) { break }
        $hybridProcess.Refresh()
        if ($hybridProcess.HasExited) {
            $tail = if (Test-Path -LiteralPath $hybridStderr) {
                (Get-Content -LiteralPath $hybridStderr -Tail 30) -join [Environment]::NewLine
            } else { "No backend stderr log was created." }
            throw "Hybrid OCR backend exited during startup.`n$tail"
        }
        Start-Sleep -Seconds 2
    }
    if (-not (Test-LocalPort -TargetPort $hybridPort)) {
        & taskkill.exe /PID $hybridProcess.Id /T /F 2>$null | Out-Null
        throw "Hybrid OCR backend did not become ready within 600 seconds. See: $hybridStderr"
    }
    Write-Host "Hybrid OCR backend ready: $hybridUrl" -ForegroundColor Green
}
else {
    Write-Host (
        "Hybrid backend is on standby. Geometry OCR handles scans without " +
        "preloading Docling; set DOCNORM_START_HYBRID=1 only when needed."
    ) -ForegroundColor DarkGray
}

$url = "http://127.0.0.1:$Port"
Write-Host "DocNorm MD: $url" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray

$browserJob = Start-Job -ScriptBlock {
    param($HealthUrl, $OpenUrl)
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -TimeoutSec 2 | Out-Null
            Start-Process $OpenUrl
            return
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
} -ArgumentList "$url/api/health", $url

try {
    Push-Location $PSScriptRoot
    & $webPython -m uvicorn docnorm_web.main:app --host 127.0.0.1 --port $Port
}
finally {
    Pop-Location
    Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
    Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
    if ($hybridStartedByDocNorm -and $null -ne $hybridProcess) {
        $hybridProcess.Refresh()
        if (-not $hybridProcess.HasExited) {
            & taskkill.exe /PID $hybridProcess.Id /T /F 2>$null | Out-Null
        }
        Write-Host "Stopped DocNorm Hybrid OCR backend." -ForegroundColor DarkGray
    }
}
