param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ImagePath,

    [ValidateSet("low", "high", "original", "auto")]
    [string]$Detail = "high",

    [int]$TimeoutSeconds = 60,

    [int]$MaxRetries = 0,

    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ResolvedImage = Resolve-Path -LiteralPath $ImagePath -ErrorAction Stop
$ImageStem = [System.IO.Path]::GetFileNameWithoutExtension($ResolvedImage.Path)
$SafeStem = $ImageStem -replace '[^A-Za-z0-9._-]', '_'
$ResultRoot = Join-Path $RepoRoot "court_grounding\outputs\windows\$SafeStem"
$JsonPath = Join-Path $ResultRoot "predictions.json"
$VisualizeDir = Join-Path $ResultRoot "visualized"
$KeyFile = Join-Path $RepoRoot "court_grounding\src\deepseek_key_local.py"

if (-not (Test-Path -LiteralPath $KeyFile -PathType Leaf) -and -not $env:DEEPSEEK_API_KEY) {
    throw @"
DeepSeek API key was not found.
Create this ignored local file:
  $KeyFile
with exactly:
  DEEPSEEK_API_KEY = "your-key"
"@
}

$Python = Get-Command $PythonCommand -ErrorAction Stop
New-Item -ItemType Directory -Path $ResultRoot -Force | Out-Null

$Arguments = @(
    "-m", "court_grounding.src.infer_deepseek",
    "--image", $ResolvedImage.Path,
    "--output", $JsonPath,
    "--visualize-dir", $VisualizeDir,
    "--detail", $Detail,
    "--timeout", $TimeoutSeconds,
    "--max-retries", $MaxRetries
)

Write-Host "Image:  $($ResolvedImage.Path)" -ForegroundColor Cyan
Write-Host "Detail: $Detail" -ForegroundColor Cyan
Write-Host "Running DeepSeek court grounding..." -ForegroundColor Cyan

Push-Location $RepoRoot
try {
    & $Python.Source @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Inference exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$ExpectedVisualization = Join-Path $VisualizeDir ("00000_" + $ImageStem + ".jpg")
Write-Host ""
Write-Host "JSON:          $JsonPath" -ForegroundColor Green
if (Test-Path -LiteralPath $ExpectedVisualization -PathType Leaf) {
    Write-Host "Visualization: $ExpectedVisualization" -ForegroundColor Green
} else {
    Write-Warning "No visualization was generated. Inspect the JSON error field: $JsonPath"
}

