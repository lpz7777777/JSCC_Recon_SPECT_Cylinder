$repo = $PSScriptRoot
$sensitivity = Join-Path $repo "Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\Result\440keV_Cartesian_UniformFullFOV_5e10"
$reconstruction = Join-Path $repo "Results\Reconstruction\JSCC_ComptonValidation_Geant4_1e9_Iter1000"

Write-Host "=== Formal Cartesian Sensi_d ==="
if (Test-Path "$sensitivity\process.pid") {
    $processId = [int](Get-Content "$sensitivity\process.pid")
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "running pid=$processId cpu_seconds=$([math]::Round($process.CPU,1))"
    } else {
        Write-Host "process pid=$processId is no longer running"
    }
}
Get-Content "$sensitivity\run_stdout.log" -Tail 5 -ErrorAction SilentlyContinue
Write-Host "finalizer: $((Get-Content "$sensitivity\finalize_status.txt" -Raw -ErrorAction SilentlyContinue).Trim())"

Write-Host "`n=== Six-output reconstruction ==="
Write-Host "status: $((Get-Content "$reconstruction\pipeline_status.txt" -Raw -ErrorAction SilentlyContinue).Trim())"
Get-Content "$reconstruction\pipeline_stdout.log" -Tail 12 -ErrorAction SilentlyContinue
Get-Content "$reconstruction\pipeline_stderr.log" -Tail 8 -ErrorAction SilentlyContinue

Write-Host "`n=== GPU compute processes ==="
& nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
