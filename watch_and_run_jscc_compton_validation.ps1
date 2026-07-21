param(
    [string]$RepositoryRoot = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $RepositoryRoot = $PSScriptRoot
}
$sensitivityRun = Join-Path $RepositoryRoot "Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\Result\440keV_Cartesian_UniformFullFOV_5e10"
$finalizeStatus = Join-Path $sensitivityRun "finalize_status.txt"
$outputDir = Join-Path $RepositoryRoot "Results\Reconstruction\JSCC_ComptonValidation_Geant4_1e9_Iter1000"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$statusPath = Join-Path $outputDir "pipeline_status.txt"

Set-Content -LiteralPath $statusPath -Value "waiting for formal Cartesian Sensi_d"
while ($true) {
    if (Test-Path -LiteralPath $finalizeStatus) {
        $status = (Get-Content -LiteralPath $finalizeStatus -Raw).Trim()
        if ($status -eq "complete") { break }
        if ($status.StartsWith("failed:")) {
            Set-Content -LiteralPath $statusPath -Value "failed: Sensi_d finalization failed"
            exit 1
        }
    }
    Start-Sleep -Seconds 30
}

Push-Location $RepositoryRoot
try {
    Set-Content -LiteralPath $statusPath -Value "running 1e9 six-output reconstruction"
    & "C:\ProgramData\anaconda3\envs\pytorch\python.exe" `
        "run_local_jscc_compton_validation.py" `
        --iterations 1000 `
        --save-step 50 `
        --device cuda:0 `
        --output-dir $outputDir `
        --overwrite
    if ($LASTEXITCODE -ne 0) { throw "Reconstruction failed: $LASTEXITCODE" }

    Set-Content -LiteralPath $statusPath -Value "generating visualization"
    & python "visualize_jscc_compton_validation.py" --result-dir $outputDir
    if ($LASTEXITCODE -ne 0) { throw "Visualization failed: $LASTEXITCODE" }
    Set-Content -LiteralPath $statusPath -Value "complete"
}
catch {
    Set-Content -LiteralPath $statusPath -Value ("failed: " + $_.Exception.Message)
    throw
}
finally {
    Pop-Location
}
