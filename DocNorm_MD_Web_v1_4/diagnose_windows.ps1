[CmdletBinding()]
param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$webPython = Join-Path $resolvedProjectRoot "venv_docnorm_web\Scripts\python.exe"
$odlPython = Join-Path $resolvedProjectRoot "venv_opendataloader\Scripts\python.exe"
$pipelineRoot = Join-Path $resolvedProjectRoot "pipeline_v2"
$hybridExecutable = Join-Path $resolvedProjectRoot "venv_opendataloader\Scripts\opendataloader-pdf-hybrid.exe"
$hybridWorker = Join-Path $PSScriptRoot "docnorm_web\opendataloader_hybrid_document_worker.py"
$vlmPython = Join-Path $resolvedProjectRoot "venv\Scripts\python.exe"
$mathWorker = Join-Path $PSScriptRoot "docnorm_web\qwen_math_worker.py"
$scanWorker = Join-Path $PSScriptRoot "docnorm_web\easyocr_geometry_worker.py"

Write-Host "=== DOCNORM MD DIAGNOSTIC ===" -ForegroundColor Cyan
Write-Host "project_root=$resolvedProjectRoot"
Write-Host "web_python_exists=$(Test-Path -LiteralPath $webPython -PathType Leaf)"
Write-Host "odl_python_exists=$(Test-Path -LiteralPath $odlPython -PathType Leaf)"
Write-Host "pipeline_exists=$(Test-Path -LiteralPath (Join-Path $pipelineRoot 'pyproject.toml') -PathType Leaf)"
Write-Host "hybrid_executable_exists=$(Test-Path -LiteralPath $hybridExecutable -PathType Leaf)"
Write-Host "hybrid_document_worker_exists=$(Test-Path -LiteralPath $hybridWorker -PathType Leaf)"
Write-Host "vlm_python_exists=$(Test-Path -LiteralPath $vlmPython -PathType Leaf)"
Write-Host "math_worker_exists=$(Test-Path -LiteralPath $mathWorker -PathType Leaf)"
Write-Host "geometry_worker_exists=$(Test-Path -LiteralPath $scanWorker -PathType Leaf)"

if (Test-Path -LiteralPath $webPython -PathType Leaf) {
    & $webPython -c (
        "import sys, fastapi, uvicorn, docx, pymupdf, pydantic; " +
        "print('python=' + sys.version.split()[0]); " +
        "print('web_imports=PASS'); " +
        "import rag_pipeline; print('rag_pipeline_import=PASS')"
    )
    & $webPython -m pip check
}

if (Test-Path -LiteralPath $odlPython -PathType Leaf) {
    & $odlPython -c (
        "import importlib.metadata as m, sys; " +
        "print('odl_python=' + sys.version.split()[0]); " +
        "print('opendataloader_version=' + m.version('opendataloader-pdf')); " +
        "import docling; print('docling_import=PASS')"
    )
}

if (Test-Path -LiteralPath $vlmPython -PathType Leaf) {
    & $vlmPython -c (
        "import torch, transformers, bitsandbytes, pymupdf, easyocr, cv2; " +
        "print('vlm_torch=' + torch.__version__); " +
        "print('vlm_cuda_available=' + str(torch.cuda.is_available())); " +
        "print('vlm_gpu=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')); " +
        "print('qwen_python_imports=PASS'); " +
        "print('geometry_ocr_imports=PASS')"
    )
}

$client = $null
try {
    $client = New-Object System.Net.Sockets.TcpClient
    $async = $client.BeginConnect("127.0.0.1", 5002, $null, $null)
    $reachable = $async.AsyncWaitHandle.WaitOne(500)
    if ($reachable) { $client.EndConnect($async) }
    Write-Host "hybrid_backend_reachable=$($reachable.ToString().ToLowerInvariant())"
}
catch { Write-Host "hybrid_backend_reachable=false" }
finally { if ($null -ne $client) { $client.Dispose() } }

& java -version
Write-Host "DOCNORM_MD_DIAGNOSTIC_DONE" -ForegroundColor Green
