[CmdletBinding()]
param(
    [string]$DestinationDirectory = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$destination = [System.IO.Path]::GetFullPath($DestinationDirectory)
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$staging = Join-Path $env:TEMP "DocNormMD_Submission_$stamp"
$packageRoot = Join-Path $staging "DocNorm_MD_Web_v1_4"
New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null

$allowlist = @(
    "docnorm_web",
    "tests",
    "docs",
    "pyproject.toml",
    "requirements-web.txt",
    "README.md",
    "VERSION",
    "SHA256SUMS",
    "THIRD_PARTY_NOTICES.md",
    ".gitignore",
    "install_windows.ps1",
    "run_windows.ps1",
    "diagnose_windows.ps1",
    "smoke_test_windows.ps1",
    "launch_docnorm.cmd",
    "build_submission.ps1"
)

foreach ($item in $allowlist) {
    $source = Join-Path $PSScriptRoot $item
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing package item: $item" }
    Copy-Item -LiteralPath $source -Destination $packageRoot -Recurse -Force
}

$forbidden = Get-ChildItem -LiteralPath $packageRoot -Recurse -Force | Where-Object {
    $_.Name -match "^(venv|\.env|data_local|model_cache|pipeline_v2_output|legacy_artifacts|__pycache__|\.git)$" -or
    $_.Extension -in @(".pyc", ".pem", ".key")
}
if ($forbidden) { throw "Forbidden files found in submission package." }

$zipPath = Join-Path $destination "DocNorm_MD_Web_v1.4.zip"
if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
Compress-Archive -LiteralPath $packageRoot -DestinationPath $zipPath -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash
Write-Host "ZIP: $zipPath" -ForegroundColor Cyan
Write-Host "SHA256: $hash" -ForegroundColor Cyan
Write-Host "DOCNORM_MD_PACKAGE=PASS" -ForegroundColor Green
