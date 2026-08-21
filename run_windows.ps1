[CmdletBinding()]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8010
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "DocNorm_MD_Web_v1_4\run_windows.ps1"

if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "DocNorm web launcher was not found: $launcher"
}

& powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File $launcher `
    -ProjectRoot $PSScriptRoot `
    -Port $Port

exit $LASTEXITCODE
