[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$forbiddenDirectories = @(
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "venv_docnorm_web",
    "model_cache",
    "data_local",
    "docnorm_runs",
    "pipeline_v2_output",
    "mineru_output",
    "legacy_artifacts"
)
$forbiddenExtensions = @(
    ".pyc", ".pdf", ".docx", ".xlsx", ".zip", ".pt", ".pth", ".bin",
    ".onnx", ".safetensors", ".pem", ".key"
)

$badDirectories = Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -Directory |
    Where-Object { $_.Name -in $forbiddenDirectories }
$badFiles = Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File |
    Where-Object { $_.Extension.ToLowerInvariant() -in $forbiddenExtensions }
$largeFiles = Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File |
    Where-Object Length -gt 50MB

if ($badDirectories) {
    $badDirectories | Select-Object FullName
    throw "Forbidden generated/local directories were found."
}
if ($badFiles) {
    $badFiles | Select-Object FullName
    throw "Forbidden document/model/archive files were found."
}
if ($largeFiles) {
    $largeFiles | Select-Object FullName, Length
    throw "Files larger than 50 MB were found."
}

$secretPattern = "-----BEGIN .*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}"
$secretMatches = Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File |
    Where-Object { $_.FullName -ne $PSCommandPath } |
    Where-Object {
        $_.Extension.ToLowerInvariant() -in @(
            ".py", ".ps1", ".toml", ".json", ".yaml", ".yml", ".md"
        )
    } |
    Select-String -Pattern $secretPattern
if ($secretMatches) {
    $secretMatches | Select-Object Path, LineNumber, Line
    throw "A possible credential was found."
}

Write-Host "DOCNORM_REPOSITORY_VERIFY=PASS" -ForegroundColor Green
