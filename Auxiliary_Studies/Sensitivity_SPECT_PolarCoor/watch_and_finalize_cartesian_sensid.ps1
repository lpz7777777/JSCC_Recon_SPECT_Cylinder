param(
    [Parameter(Mandatory = $true)][int]$ProcessId,
    [string]$RepositoryRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
$resultRoot = Join-Path $PSScriptRoot "Result"
$cartesianResult = Join-Path $resultRoot "440keV_Cartesian_UniformFullFOV_5e10"
$polarResult = Join-Path $resultRoot "440keV_RotateNum20_UniformFullFOV_5e10_Cartesian"
$statusPath = Join-Path $cartesianResult "finalize_status.txt"

Set-Content -LiteralPath $statusPath -Value "waiting pid=$ProcessId"
while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$cartesianSensitivity = Join-Path $cartesianResult "Sensi_d"
$metadata = Join-Path $cartesianResult "run_metadata.json"
if (-not (Test-Path -LiteralPath $cartesianSensitivity) -or
    -not (Test-Path -LiteralPath $metadata)) {
    Set-Content -LiteralPath $statusPath -Value "failed: Cartesian calculation did not produce final outputs"
    exit 1
}

Set-Content -LiteralPath $statusPath -Value "converting Cartesian point efficiency to polar density basis"
Push-Location $RepositoryRoot
try {
    & python "Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\convert_cartesian_sensitivity_to_polar.py" `
        --cartesian-result-dir $cartesianResult `
        --cartesian-input-dir "Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\Input\440keV_CartesianCylinder_R153" `
        --factor-dir "Factors\440keV_RotateNum20" `
        --output-dir $polarResult `
        --install-to-factor-dir `
        --overwrite
    if ($LASTEXITCODE -ne 0) { throw "Cartesian-to-polar conversion failed: $LASTEXITCODE" }

    & python "Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\visualize_sensi_d_vs_single_photon.py" `
        --factor-dir "Factors\440keV_RotateNum20" `
        --result-dir $polarResult
    if ($LASTEXITCODE -ne 0) { throw "Sensitivity visualization failed: $LASTEXITCODE" }
    Set-Content -LiteralPath $statusPath -Value "complete"
}
catch {
    Set-Content -LiteralPath $statusPath -Value ("failed: " + $_.Exception.Message)
    throw
}
finally {
    Pop-Location
}
