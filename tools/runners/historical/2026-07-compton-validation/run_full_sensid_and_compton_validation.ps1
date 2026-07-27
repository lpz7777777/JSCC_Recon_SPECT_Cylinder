param(
    [string]$RepositoryRoot = "",
    [string]$Python = "C:\ProgramData\anaconda3\envs\pytorch\python.exe",
    [string]$ExperimentTag = "Baseline_ER10_Sum400"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
}
$RepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$resultRoot = Join-Path $RepositoryRoot "Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\Result"
$sensitivityDir = Join-Path $resultRoot "440keV_RotateNum20_UniformFullFOV_5e10_CorrectDetector_Full_$ExperimentTag"
$reconstructionDir = Join-Path $RepositoryRoot "Results\Reconstruction\JSCC_ComptonValidation_Geant4_1e9_Iter1000_FullGrid_$ExperimentTag"
$pipelineDir = Join-Path $RepositoryRoot "Results\Pipelines\FullSensiD_Then_ComptonValidation_20260722_$ExperimentTag"
$factorSensi = Join-Path $RepositoryRoot "Factors\440keV_RotateNum20\Sensi_d"
$listFile = Join-Path $RepositoryRoot "List\440keV_RotateNum20_Geant4JSCC\ComptonSensitivity_UniformFullFOV_5e10\List_UniformFullFOV_440keV.csv"

New-Item -ItemType Directory -Force -Path $pipelineDir | Out-Null
New-Item -ItemType Directory -Force -Path $sensitivityDir | Out-Null
New-Item -ItemType Directory -Force -Path $reconstructionDir | Out-Null
$statusPath = Join-Path $pipelineDir "pipeline_status.txt"
$stagePath = Join-Path $pipelineDir "current_stage.txt"

function Set-Stage([string]$Stage, [string]$Status) {
    Set-Content -LiteralPath $stagePath -Value $Stage -Encoding ascii
    Set-Content -LiteralPath $statusPath -Value $Status -Encoding ascii
}

Push-Location $RepositoryRoot
try {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
    if (-not (Test-Path -LiteralPath $listFile)) { throw "Uniform-FOV List not found: $listFile" }

    $backup = Join-Path $RepositoryRoot "Factors\440keV_RotateNum20\Sensi_d_before_$ExperimentTag"
    if ((Test-Path -LiteralPath $factorSensi) -and -not (Test-Path -LiteralPath $backup)) {
        Copy-Item -LiteralPath $factorSensi -Destination $backup
    }

    Set-Stage "sensitivity" "running full 440-keV rotation-averaged Sensi_d"
    & $Python -u "Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\run_compton_sensitivity.py" `
        --factor-dir "Factors\440keV_RotateNum20" `
        --compton-list $listFile `
        --source-photons 5e10 `
        --energy-mev 0.440 `
        --rotate-num 20 `
        --energy-resolution-662kev 0.13 `
        --energy-resolution-reference-kev 511 `
        --energy-threshold-sum-mev 0.350 `
        --input-energies-already-smeared `
        --min-event-effective-support 1 `
        --device cuda `
        --batch-size 256 `
        --checkpoint-every-batches 1000 `
        --progress-every-batches 100 `
        --output-dir $sensitivityDir `
        --install-to-factor-dir `
        --overwrite
    if ($LASTEXITCODE -ne 0) { throw "Sensi_d calculation failed with exit code $LASTEXITCODE" }

    Set-Stage "sensitivity_visualization" "generating full Sensi_d validation figure"
    & $Python -u "Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\visualize_sensi_d_vs_single_photon.py" `
        --factor-dir "Factors\440keV_RotateNum20" `
        --result-dir $sensitivityDir
    if ($LASTEXITCODE -ne 0) { throw "Sensi_d visualization failed with exit code $LASTEXITCODE" }

    Set-Stage "reconstruction" "running 1e9 six-output reconstruction, 1000 MLEM iterations, full Compton grid"
    & $Python -u "run_local_jscc_compton_validation.py" `
        --iterations 1000 `
        --save-step 50 `
        --full-compton-grid `
        --energy-resolution-fwhm 0.13 `
        --energy-resolution-reference-kev 511 `
        --energy-threshold-sum-mev 0.350 `
        --device cuda:0 `
        --sensi-d-path (Join-Path $sensitivityDir "Sensi_d") `
        --output-dir $reconstructionDir `
        --overwrite
    if ($LASTEXITCODE -ne 0) { throw "Reconstruction failed with exit code $LASTEXITCODE" }

    Set-Stage "reconstruction_visualization" "generating six-output reconstruction figure"
    & $Python -u "tools\visualization\compton\visualize_jscc_compton_validation.py" `
        --result-dir $reconstructionDir `
        --factor-dir "Factors\440keV_RotateNum20"
    if ($LASTEXITCODE -ne 0) { throw "Reconstruction visualization failed with exit code $LASTEXITCODE" }

    Set-Stage "complete" "complete"
}
catch {
    Set-Stage "failed" ("failed: " + $_.Exception.Message)
    throw
}
finally {
    Pop-Location
}
