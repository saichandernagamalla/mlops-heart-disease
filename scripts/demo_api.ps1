param(
    [string]$BaseUrl = "http://localhost:8001"
)

$ErrorActionPreference = "Stop"
$sample = Get-Content -LiteralPath (Join-Path $PSScriptRoot "sample_patient.json") -Raw

Write-Output "MODEL READINESS"
$health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 10
if (-not $health.model_loaded) {
    throw "The service is not ready: model is missing."
}
$health | ConvertTo-Json

Write-Output "PREDICTION"
$prediction = Invoke-RestMethod -Method Post -Uri "$BaseUrl/predict" `
    -ContentType "application/json" -Body $sample -TimeoutSec 10
if ($prediction.prediction -notin @(0, 1) -or $null -eq $prediction.confidence_positive -or
    $prediction.confidence_positive -lt 0 -or $prediction.confidence_positive -gt 1) {
    throw "Prediction response does not satisfy the API contract."
}
$prediction | ConvertTo-Json

Write-Output "PROMETHEUS COUNTERS AND LATENCY"
$metrics = Invoke-RestMethod -Uri "$BaseUrl/metrics" -TimeoutSec 10
if ($metrics -notmatch 'prediction_requests_total\{outcome="success"\}') {
    throw "The metrics endpoint did not record a successful prediction."
}
$metrics -split "`n" | Where-Object {
    $_ -match '^prediction_requests_total|^prediction_latency_seconds_(count|sum)'
}
Write-Output "Live dashboard: $BaseUrl/monitor"
