param(
    [int]$Tail = 25
)

$ErrorActionPreference = "Continue"

$Root = $PSScriptRoot
$ProjectRoot = (Resolve-Path (Join-Path $Root "..\..")).Path
$RunsRoot = Join-Path $Root "runs"
$LogRoot = Join-Path $ProjectRoot "run_logs\ScatterGen_DetectorLocal_20260716"
$StateFile = Join-Path $LogRoot "detector_local_scatter_state.json"
$WorkerPidFile = Join-Path $LogRoot "detector_local_scatter_worker.pid"
$ExpectedBytes = 2397081600
$MatrixNames = @(
    "Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat",
    "SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat"
)
$RunNames = @(
    "JSCC_218keV",
    "JSCC_440keV",
    "JSCC_440keV_to_218keVwin"
)

function Format-Bytes {
    param([double]$Bytes)
    if ($Bytes -ge 1GB) { return ("{0:N3} GiB" -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ("{0:N1} MiB" -f ($Bytes / 1MB)) }
    return ("{0:N0} B" -f $Bytes)
}

Write-Host "State file: $StateFile"
if (Test-Path -LiteralPath $StateFile) {
    $state = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
    Write-Host ("Status: {0}; current: {1}; updated: {2}; workerPid: {3}" -f `
        $state.status, $state.current, $state.updated, $state.workerPid)
    $state.runs | Select-Object name,status,pid,started,finished,exitCode | Format-Table -AutoSize
} else {
    Write-Host "No state file yet."
}

if (Test-Path -LiteralPath $WorkerPidFile) {
    $workerPid = [int](Get-Content -LiteralPath $WorkerPidFile -Raw)
    $worker = Get-Process -Id $workerPid -ErrorAction SilentlyContinue
    if ($worker) {
        Write-Host ("Worker process alive: PID {0}, CPU {1:N1}s" -f $worker.Id, $worker.CPU)
    } else {
        Write-Host "Worker process is not alive."
    }
}

$scatterProcesses = Get-Process | Where-Object { $_.ProcessName -like "ScatterGen*" }
if ($scatterProcesses) {
    $scatterProcesses | Select-Object Id,ProcessName,CPU,StartTime,Path | Format-Table -AutoSize
} else {
    Write-Host "No ScatterGen process is currently visible."
}

Write-Host ""
Write-Host "GPU:"
try {
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
} catch {
    Write-Host "nvidia-smi unavailable."
}

Write-Host ""
Write-Host "Output matrices:"
foreach ($runName in $RunNames) {
    $runDir = Join-Path $RunsRoot $runName
    Write-Host "[$runName]"
    foreach ($matrix in $MatrixNames) {
        $path = Join-Path $runDir $matrix
        if (Test-Path -LiteralPath $path) {
            $item = Get-Item -LiteralPath $path
            $percent = 100.0 * $item.Length / $ExpectedBytes
            Write-Host ("  {0}: {1} ({2:N1}%) updated {3}" -f `
                $matrix, (Format-Bytes $item.Length), $percent, $item.LastWriteTime)
        } else {
            Write-Host "  ${matrix}: missing"
        }
    }
}

Write-Host ""
Write-Host "Log tails:"
foreach ($runName in $RunNames) {
    $runDir = Join-Path $RunsRoot $runName
    $log = Join-Path $runDir "ScatterGen_detector_local.log"
    $err = Join-Path $runDir "ScatterGen_detector_local.err.log"
    if (Test-Path -LiteralPath $log) {
        Write-Host ""
        Write-Host "[$runName stdout tail]"
        Get-Content -LiteralPath $log -Tail $Tail
    }
    if (Test-Path -LiteralPath $err) {
        $errItem = Get-Item -LiteralPath $err
        if ($errItem.Length -gt 0) {
            Write-Host ""
            Write-Host "[$runName stderr tail]"
            Get-Content -LiteralPath $err -Tail $Tail
        }
    }
}
