param(
    [switch]$Worker,
    [int]$Cuda = 0
)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$ProjectRoot = (Resolve-Path (Join-Path $Root "..\..")).Path
$RunsRoot = Join-Path $Root "runs"
$LogRoot = Join-Path $ProjectRoot "run_logs\ScatterGen_DetectorLocal_20260716"
$StateFile = Join-Path $LogRoot "detector_local_scatter_state.json"
$WorkerPidFile = Join-Path $LogRoot "detector_local_scatter_worker.pid"
$Exe = Join-Path $Root "ScatterGen_RayTracing_CircularHole\ScatterGen_CircularHole_detector_local.exe"
$PeName = "PE_SysMat_shift_0.000000_0.000000_0.000000_v3.sysmat"
$ExpectedMatrixBytes = 2397081600
$ScatterName = "Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat"
$CombinedName = "SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat"

$Tasks = @(
    [ordered]@{ Name = "JSCC_218keV"; Pe = $PeName },
    [ordered]@{ Name = "JSCC_440keV"; Pe = $PeName },
    [ordered]@{ Name = "JSCC_440keV_to_218keVwin"; Pe = $PeName }
)

function Write-State {
    param([hashtable]$State)
    $State.updated = (Get-Date).ToString("s")
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StateFile -Encoding UTF8
}

function Test-RunCompleted {
    param([string]$RunDir)
    $scatterPath = Join-Path $RunDir $ScatterName
    $combinedPath = Join-Path $RunDir $CombinedName
    if (-not (Test-Path -LiteralPath $scatterPath)) { return $false }
    if (-not (Test-Path -LiteralPath $combinedPath)) { return $false }
    $scatter = Get-Item -LiteralPath $scatterPath
    $combined = Get-Item -LiteralPath $combinedPath
    return ($scatter.Length -eq $ExpectedMatrixBytes -and $combined.Length -eq $ExpectedMatrixBytes)
}

if (-not $Worker) {
    New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
    if (-not (Test-Path -LiteralPath $Exe)) {
        throw "ScatterGen executable not found: $Exe"
    }
    $process = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $PSCommandPath,
            "-Worker",
            "-Cuda", [string]$Cuda
        ) `
        -WindowStyle Hidden `
        -PassThru
    [string]$process.Id | Set-Content -LiteralPath $WorkerPidFile -Encoding ASCII
    "Started detector-local ScatterGen worker PID $($process.Id)."
    "Monitor with:"
    "  powershell -ExecutionPolicy Bypass -File `"$Root\monitor_jscc_detector_local_scatter.ps1`""
    return
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
if (-not (Test-Path -LiteralPath $Exe)) {
    throw "ScatterGen executable not found: $Exe"
}

$env:DETECTOR_LOCAL_SCATTER_ORIENTATION_BINS = "17"
$env:DETECTOR_LOCAL_SCATTER_COSINE_SAMPLES = "64"
$env:DETECTOR_LOCAL_SCATTER_AZIMUTH_SAMPLES = "64"
$env:DETECTOR_LOCAL_SCATTER_POSITION_SAMPLES_PER_AXIS = "4"
$env:SCATTER_WRITE_COMPONENTS = "0"
Remove-Item Env:\SCATTER_PAIR_LENGTH_CACHE -ErrorAction SilentlyContinue

$state = @{
    status = "running"
    cuda = $Cuda
    workerPid = $PID
    executable = $Exe
    started = (Get-Date).ToString("s")
    updated = (Get-Date).ToString("s")
    current = $null
    runs = @()
}
foreach ($task in $Tasks) {
    $state.runs += @{
        name = $task.Name
        status = "pending"
        started = $null
        finished = $null
        exitCode = $null
        pid = $null
        log = $null
        err = $null
    }
}
Write-State $state

for ($i = 0; $i -lt $Tasks.Count; $i++) {
    $task = $Tasks[$i]
    $runDir = Join-Path $RunsRoot $task.Name
    $pePath = Join-Path $runDir $task.Pe
    if (-not (Test-Path -LiteralPath $runDir)) {
        throw "Run directory not found: $runDir"
    }
    if (-not (Test-Path -LiteralPath $pePath)) {
        throw "PE matrix not found: $pePath"
    }

    if (Test-RunCompleted $runDir) {
        $state.runs[$i].status = "completed"
        $state.runs[$i].started = $state.runs[$i].started
        $state.runs[$i].finished = (Get-Date).ToString("s")
        $state.runs[$i].exitCode = 0
        $state.runs[$i].log = Join-Path $runDir "ScatterGen_detector_local.log"
        $state.runs[$i].err = Join-Path $runDir "ScatterGen_detector_local.err.log"
        Write-State $state
        continue
    }

    $log = Join-Path $runDir "ScatterGen_detector_local.log"
    $err = Join-Path $runDir "ScatterGen_detector_local.err.log"
    Remove-Item -LiteralPath $log, $err -ErrorAction SilentlyContinue

    $state.current = $task.Name
    $state.runs[$i].status = "running"
    $state.runs[$i].started = (Get-Date).ToString("s")
    $state.runs[$i].log = $log
    $state.runs[$i].err = $err
    Write-State $state

    $process = Start-Process -FilePath $Exe `
        -ArgumentList @("-PE", $task.Pe, "-cuda", [string]$Cuda) `
        -WorkingDirectory $runDir `
        -RedirectStandardOutput $log `
        -RedirectStandardError $err `
        -WindowStyle Hidden `
        -PassThru
    $state.runs[$i].pid = $process.Id
    Write-State $state

    while (-not $process.HasExited) {
        Start-Sleep -Seconds 30
        $process.Refresh()
        Write-State $state
    }
    $process.WaitForExit()

    $state.runs[$i].exitCode = $process.ExitCode
    $state.runs[$i].finished = (Get-Date).ToString("s")
    if ($process.ExitCode -eq 0 -or (Test-RunCompleted $runDir)) {
        if ($null -eq $process.ExitCode) {
            $state.runs[$i].exitCode = 0
        }
        $state.runs[$i].status = "completed"
    } else {
        $state.runs[$i].status = "failed"
        $state.status = "failed"
        Write-State $state
        exit $process.ExitCode
    }
    Write-State $state
}

$state.current = $null
$state.status = "completed"
$state.finished = (Get-Date).ToString("s")
Write-State $state
