param([string]$ExperimentTag = "Baseline_ER10_Sum400")
$repo = $PSScriptRoot
$pipeline = Join-Path $repo "Results\Pipelines\FullSensiD_Then_ComptonValidation_20260722_$ExperimentTag"
$sensitivity = Join-Path $repo "Auxiliary_Studies\Sensitivity_SPECT_PolarCoor\Result\440keV_RotateNum20_UniformFullFOV_5e10_CorrectDetector_Full_$ExperimentTag"
$reconstruction = Join-Path $repo "Results\Reconstruction\JSCC_ComptonValidation_Geant4_1e9_Iter1000_FullGrid_$ExperimentTag"

Write-Host "=== Pipeline ==="
Write-Host "stage:  $((Get-Content (Join-Path $pipeline 'current_stage.txt') -Raw -ErrorAction SilentlyContinue).Trim())"
Write-Host "status: $((Get-Content (Join-Path $pipeline 'pipeline_status.txt') -Raw -ErrorAction SilentlyContinue).Trim())"
if (Test-Path (Join-Path $pipeline "pipeline.pid")) {
    $pipelinePid = [int](Get-Content (Join-Path $pipeline "pipeline.pid"))
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$pipelinePid" -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "runner: running pid=$pipelinePid"
    } else {
        Write-Host "runner: pid=$pipelinePid is not running"
    }
}
$workers = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match "run_compton_sensitivity.py|run_local_jscc_compton_validation.py|visualize_jscc_compton_validation.py"
}
foreach ($worker in $workers) {
    Write-Host "worker: $($worker.Name) pid=$($worker.ProcessId) parent=$($worker.ParentProcessId)"
}

Write-Host "`n=== Latest output ==="
Get-Content (Join-Path $pipeline "pipeline_stdout.log") -Tail 15 -ErrorAction SilentlyContinue
Get-Content (Join-Path $pipeline "pipeline_stderr.log") -Tail 8 -ErrorAction SilentlyContinue

Write-Host "`n=== Products ==="
if (Test-Path (Join-Path $sensitivity "run_metadata.json")) { Write-Host "Sensi_d complete" }
if (Test-Path (Join-Path $sensitivity "Sensi_d_vs_single_photon.png")) { Write-Host "Sensi_d figure complete" }
if (Test-Path (Join-Path $reconstruction "run_manifest.json")) { Write-Host "reconstruction complete" }
if (Test-Path (Join-Path $reconstruction "JSCC_ComptonValidation_1e9_Iter1000.png")) { Write-Host "reconstruction figure complete" }

Write-Host "`n=== GPU ==="
& nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
