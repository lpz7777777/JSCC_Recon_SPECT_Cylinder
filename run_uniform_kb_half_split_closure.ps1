param(
    [string]$FactorDir = "Factors/440keV_RotateNum20",
    [string]$ListFile = "List/440keV_RotateNum20_Geant4JSCC/ComptonSensitivity_UniformFullFOV_5e10/List_UniformFullFOV_440keV.csv",
    [double]$SourcePhotons = 5e10,
    [double]$HalfFraction = 0.5,
    [int]$BatchSize = 256,
    [string]$OutputName = "440keV_RotateNum20_UniformFullFOV_5e10_KB_HalfSplit_ER13_FWHM511_Sum350"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

if ($HalfFraction -le 0 -or 2.0 * $HalfFraction -gt 1.0) {
    throw "HalfFraction must be positive and satisfy 2*HalfFraction <= 1."
}
$factorPath = Join-Path $repo $FactorDir
$listPath = Join-Path $repo $ListFile
$resultRoot = Join-Path $repo "Auxiliary_Studies/Sensitivity_SPECT_PolarCoor/Result/$OutputName"
$calibrationDir = Join-Path $resultRoot "CalibrationHalfA"
$closureDir = Join-Path $resultRoot "IndependentHalfB_Closure"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

$runLog = Join-Path $resultRoot "run.log"
"[$(Get-Date -Format o)] stage=calibration status=starting" | Tee-Object -FilePath $runLog

& conda run --no-capture-output -n pytorch python -u `
    .\Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\run_compton_sensitivity.py `
    --factor-dir $factorPath `
    --compton-list $listPath `
    --source-photons $SourcePhotons `
    --energy-mev 0.440 `
    --rotate-num 20 `
    --event-start-fraction 0.0 `
    --event-fraction $HalfFraction `
    --energy-resolution-662kev 0.13 `
    --energy-resolution-reference-kev 511 `
    --energy-threshold-sum-mev 0.350 `
    --input-energies-already-smeared `
    --device cuda `
    --batch-size $BatchSize `
    --output-dir $calibrationDir `
    --overwrite 2>&1 | Tee-Object -FilePath $runLog -Append
if ($LASTEXITCODE -ne 0) { throw "Half-A Sensi_d calculation failed with exit code $LASTEXITCODE." }

$candidate = Join-Path $calibrationDir "Sensi_d"
if (-not (Test-Path -LiteralPath $candidate)) { throw "Half-A Sensi_d is missing: $candidate" }
"[$(Get-Date -Format o)] stage=closure status=starting" | Tee-Object -FilePath $runLog -Append

& conda run --no-capture-output -n pytorch python -u `
    .\Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\validate_uniform_compton_closure.py `
    --factor-dir $factorPath `
    --compton-list $listPath `
    --sensi-d $candidate `
    --source-photons $SourcePhotons `
    --event-start-fraction $HalfFraction `
    --event-fraction $HalfFraction `
    --energy-mev 0.440 `
    --rotate-num 20 `
    --energy-resolution-fwhm 0.13 `
    --energy-resolution-reference-kev 511 `
    --energy-threshold-sum-mev 0.350 `
    --device cuda `
    --batch-size $BatchSize `
    --output-dir $closureDir 2>&1 | Tee-Object -FilePath $runLog -Append
if ($LASTEXITCODE -ne 0) { throw "Independent Half-B closure failed with exit code $LASTEXITCODE." }

"[$(Get-Date -Format o)] stage=complete status=complete" | Tee-Object -FilePath $runLog -Append
Write-Host "Calibration: $calibrationDir"
Write-Host "Closure:     $closureDir"
