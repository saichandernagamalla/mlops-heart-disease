param(
    [string]$PackageIndex = "https://pypi.org/simple"
)

$ErrorActionPreference = "Stop"

$imageName = "heart-mlops:local"
$containerName = "heart-api-verification"
$containerId = $null
$baseUrl = "http://127.0.0.1:8000"

docker build --build-arg "PIP_INDEX_URL=$PackageIndex" -t $imageName .
if ($LASTEXITCODE -ne 0) {
    throw "Docker image build failed with exit code $LASTEXITCODE."
}

try {
    $containerId = docker run --rm --detach --name $containerName -p 8000:8000 $imageName
    if ($LASTEXITCODE -ne 0) {
        throw "Docker container failed to start with exit code $LASTEXITCODE."
    }

    $health = $null
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            $health = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 2
            if ($health.model_loaded) {
                break
            }
        }
        catch {
            Start-Sleep -Seconds 1
        }
        Start-Sleep -Milliseconds 300
    }

    if (-not $health -or -not $health.model_loaded) {
        docker logs $containerName
        throw "The container did not report a loaded model at /health."
    }

    $sample = Get-Content -LiteralPath (Join-Path $PSScriptRoot "sample_patient.json") -Raw

    $prediction = Invoke-RestMethod `
        -Method Post `
        -Uri "$baseUrl/predict" `
        -ContentType "application/json" `
        -Body $sample
    $metrics = Invoke-WebRequest -Uri "$baseUrl/metrics" -UseBasicParsing

    if ($prediction.prediction -notin @(0, 1)) {
        throw "The /predict response did not contain a binary class."
    }
    if ($null -eq $prediction.confidence_positive -or $prediction.confidence_positive -lt 0 -or $prediction.confidence_positive -gt 1) {
        throw "The /predict response did not contain a probability between 0 and 1."
    }
    if ($metrics.StatusCode -ne 200 -or $metrics.Content -notmatch "prediction_requests_total") {
        throw "The /metrics endpoint did not return Prometheus prediction metrics."
    }

    [PSCustomObject]@{
        image = $imageName
        health = $health
        prediction = $prediction
        metrics_status = $metrics.StatusCode
        prediction_metric_present = $true
    } | ConvertTo-Json -Depth 5
}
finally {
    if ($containerId) {
        docker stop $containerName | Out-Null
    }
}
