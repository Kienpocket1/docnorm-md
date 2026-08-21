[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$installer = Join-Path $PSScriptRoot "DocNorm_MD_Web_v1_4\install_windows.ps1"

if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) {
    throw "DocNorm web installer was not found: $installer"
}

& powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File $installer `
    -ProjectRoot $PSScriptRoot

if ($LASTEXITCODE -ne 0) {
    throw "DocNorm installation failed with exit code $LASTEXITCODE."
}

Write-Host "DOCNORM_GITHUB_INSTALL=PASS" -ForegroundColor Green
