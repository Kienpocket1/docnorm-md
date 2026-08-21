[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$venvRoot = Join-Path $resolvedProjectRoot "venv_docnorm_web"
$webPython = Join-Path $venvRoot "Scripts\python.exe"
$pipelineRoot = Join-Path $resolvedProjectRoot "pipeline_v2"
$odlPython = Join-Path $resolvedProjectRoot "venv_opendataloader\Scripts\python.exe"
$hybridExecutable = Join-Path $resolvedProjectRoot "venv_opendataloader\Scripts\opendataloader-pdf-hybrid.exe"
$localWorker = Join-Path $pipelineRoot "src\rag_pipeline\workers\opendataloader_local_worker.py"
$hybridWorker = Join-Path $PSScriptRoot "docnorm_web\opendataloader_hybrid_document_worker.py"
$vlmPython = Join-Path $resolvedProjectRoot "venv\Scripts\python.exe"
$mathWorker = Join-Path $PSScriptRoot "docnorm_web\qwen_math_worker.py"
$scanWorker = Join-Path $PSScriptRoot "docnorm_web\easyocr_geometry_worker.py"

function Invoke-PythonProbe {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string]$Code
    )
    $previousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5 may promote harmless native stderr warnings to
        # NativeCommandError while ErrorActionPreference is Stop. Capture both
        # streams so the real process exit code remains authoritative.
        $ErrorActionPreference = "Continue"
        $probeOutput = & $Executable -c $Code 2>&1
        $probeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($null -ne $probeOutput) {
        $probeOutput | ForEach-Object { Write-Host $_ }
    }
    return $probeExitCode
}

Write-Host "ProjectRoot: $resolvedProjectRoot" -ForegroundColor Cyan
Write-Host "Creating isolated Python 3.12 environment..." -ForegroundColor Cyan
$createdVenv = $false
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_NO_INPUT = "1"

if (-not (Test-Path -LiteralPath $webPython -PathType Leaf)) {
    $pyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        & $pyLauncher.Source -3.12 -m venv $venvRoot
        $createdVenv = $true
    }
    else {
        $pythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
        if ($null -eq $pythonCommand) {
            throw "Python 3.12 was not found. Install Python 3.12 first."
        }
        & $pythonCommand.Source -c (
            "import sys; assert sys.version_info[:2] == (3, 12), sys.version"
        )
        if ($LASTEXITCODE -ne 0) {
            throw "The python command is not Python 3.12."
        }
        & $pythonCommand.Source -m venv $venvRoot
        $createdVenv = $true
    }
}

if (-not (Test-Path -LiteralPath $webPython -PathType Leaf)) {
    throw "Unable to create venv_docnorm_web."
}

& $webPython -m pip --version
if ($LASTEXITCODE -ne 0) {
    & $webPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw "Unable to initialize pip." }
}

if ($createdVenv -or $env:DOCNORM_FORCE_TOOLING_UPGRADE -eq "1") {
    Write-Host "Preparing pip tooling (first install only)..." -ForegroundColor Cyan
    & $webPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "Unable to upgrade pip tooling." }
}
else {
    Write-Host "Reusing existing pip tooling; skipped network upgrade." -ForegroundColor DarkGray
}

Write-Host "Installing/updating DocNorm MD web package..." -ForegroundColor Cyan
& $webPython -m pip install -e "${PSScriptRoot}[dev]"
if ($LASTEXITCODE -ne 0) { throw "Unable to install DocNorm MD web dependencies." }

if (Test-Path -LiteralPath (Join-Path $pipelineRoot "pyproject.toml") -PathType Leaf) {
    & $webPython -m pip install -e $pipelineRoot
    if ($LASTEXITCODE -ne 0) { throw "Unable to install pipeline_v2 in editable mode." }
}
else {
    Write-Warning "pipeline_v2 was not found. BASIC_FALLBACK mode remains available."
}

& $webPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip check failed." }

$env:DOCNORM_PROJECT_ROOT = $resolvedProjectRoot
$env:DOCNORM_RUNS_ROOT = Join-Path $resolvedProjectRoot "docnorm_runs"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
& $webPython -c (
    "from docnorm_web.main import app; " +
    "from docnorm_web.config import get_settings; " +
    "print('app_import=PASS'); print('runs_root=' + str(get_settings().runs_root))"
)
if ($LASTEXITCODE -ne 0) { throw "DocNorm MD import probe failed." }

$hybridReady = (
    (Test-Path -LiteralPath $odlPython -PathType Leaf) -and
    (Test-Path -LiteralPath $hybridExecutable -PathType Leaf) -and
    (Test-Path -LiteralPath $localWorker -PathType Leaf) -and
    (Test-Path -LiteralPath $hybridWorker -PathType Leaf)
)
if ($hybridReady) {
    $hybridProbe = Invoke-PythonProbe -Executable $odlPython -Code (
        "import importlib.metadata as m, docling; " +
        "version=m.version('opendataloader-pdf'); " +
        "assert version == '2.5.0', version; " +
        "print('opendataloader_version=' + version); print('docling_import=PASS')"
    )
    if ($hybridProbe -ne 0) {
        throw "Hybrid dependencies failed import/version validation."
    }
    Write-Host "hybrid_ocr_ready=true" -ForegroundColor Green
}
else {
    Write-Warning (
        "Optional Hybrid OCR components were not found. Geometry OCR can still " +
        "handle scans when venv has EasyOCR/OpenCV/CUDA."
    )
    Write-Host "hybrid_ocr_ready=false" -ForegroundColor Yellow
}

$mathVlmReady = (
    (Test-Path -LiteralPath $vlmPython -PathType Leaf) -and
    (Test-Path -LiteralPath $mathWorker -PathType Leaf)
)
if ($mathVlmReady) {
    $mathProbe = Invoke-PythonProbe -Executable $vlmPython -Code (
        "import torch, transformers, bitsandbytes, pymupdf; " +
        "assert torch.cuda.is_available(), 'CUDA unavailable'; " +
        "print('qwen_python_imports=PASS'); " +
        "print('qwen_cuda=' + torch.cuda.get_device_name(0))"
    )
    if ($mathProbe -ne 0) {
        Write-Warning "Qwen math environment exists but failed CUDA/import validation."
        $mathVlmReady = $false
    }
}
if ($mathVlmReady) {
    Write-Host "math_vlm_ready=true" -ForegroundColor Green
}
else {
    Write-Warning (
        "Selective math fallback is unavailable. Keep <ProjectRoot>\venv with " +
        "Qwen3-VL 2B, Transformers, bitsandbytes, PyMuPDF and CUDA enabled."
    )
    Write-Host "math_vlm_ready=false" -ForegroundColor Yellow
}

$geometryOcrReady = (
    (Test-Path -LiteralPath $vlmPython -PathType Leaf) -and
    (Test-Path -LiteralPath $scanWorker -PathType Leaf)
)
if ($geometryOcrReady) {
    $geometryProbe = Invoke-PythonProbe -Executable $vlmPython -Code (
        "import torch, easyocr, cv2, pymupdf; " +
        "assert torch.cuda.is_available(), 'CUDA unavailable'; " +
        "print('geometry_ocr_imports=PASS'); " +
        "print('geometry_ocr_cuda=' + torch.cuda.get_device_name(0))"
    )
    if ($geometryProbe -ne 0) {
        Write-Warning "Geometry OCR environment failed EasyOCR/OpenCV/CUDA validation."
        $geometryOcrReady = $false
    }
}
if ($geometryOcrReady) {
    Write-Host "geometry_ocr_ready=true" -ForegroundColor Green
}
else {
    Write-Warning (
        "Geometry OCR is unavailable. Install EasyOCR and OpenCV in " +
        "<ProjectRoot>\venv or keep the Hybrid backend enabled."
    )
    Write-Host (
        "Fix: & `"$vlmPython`" -m pip install easyocr opencv-python-headless pillow"
    ) -ForegroundColor Yellow
    Write-Host "geometry_ocr_ready=false" -ForegroundColor Yellow
}

Write-Host "DOCNORM_MD_INSTALL=PASS" -ForegroundColor Green
Write-Host "Run: .\run_windows.ps1 -ProjectRoot `"$resolvedProjectRoot`"" -ForegroundColor Green
