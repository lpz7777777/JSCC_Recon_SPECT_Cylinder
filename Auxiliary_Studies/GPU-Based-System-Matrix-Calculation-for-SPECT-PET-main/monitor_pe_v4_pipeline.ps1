param(
    [int]$IntervalSeconds = 15,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$engineRoot = $PSScriptRoot
$statusPath = Join-Path $engineRoot "runs\PEV4_JSCC_pipeline_status.json"
$peMonitor = Join-Path $engineRoot `
    "PEGen_RayTracing_CircularHole\monitor_pe_v4.ps1"

function Read-SharedText([string]$Path) {
    $share = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
    $stream = [System.IO.FileStream]::new(
        $Path,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        $share
    )
    try {
        $reader = [System.IO.StreamReader]::new($stream)
        try {
            return $reader.ReadToEnd()
        } finally {
            $reader.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

do {
    if (-not (Test-Path -LiteralPath $statusPath)) {
        Write-Output "Waiting for $statusPath"
    } else {
        $state = Read-SharedText $statusPath | ConvertFrom-Json
        Write-Output ("[{0}] stage={1} status={2}: {3}" -f `
            $state.last_update, $state.stage, $state.status, $state.message)
        if ($state.stage -eq "pe_v4" -and $state.run_directory) {
            $progress = Join-Path $state.run_directory "PE_v4_progress.json"
            & $peMonitor -ProgressPath $progress -Once
        } elseif ($state.stage -like "scatter_*" -and
            (Test-Path -LiteralPath $state.log_path)) {
            $matches = Select-String -LiteralPath $state.log_path `
                -Pattern 'Crystal chunk scatterStart=(\d+)' | Select-Object -Last 1
            if ($matches) {
                $start = [int]$matches.Matches[0].Groups[1].Value
                $fraction = [math]::Min(1.0, ($start + 16) / 11520.0)
                $stageStart = [DateTimeOffset]::Parse([string]$state.last_update)
                $elapsedSeconds = ([DateTimeOffset]::Now - $stageStart).TotalSeconds
                $etaSeconds = if ($fraction -gt 0) {
                    $elapsedSeconds * (1.0 - $fraction) / $fraction
                } else { 0.0 }
                $eta = [TimeSpan]::FromSeconds([math]::Max(0, $etaSeconds))
                Write-Output ("  inter-crystal scatter chunks: {0}/11520 ({1:P2}), ETA {2:hh\:mm\:ss}" -f `
                    ([math]::Min(11520, $start + 16)), $fraction, $eta)
            }
            Get-Content -LiteralPath $state.log_path -Tail 4
            & nvidia-smi `
                --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu `
                --format=csv,noheader,nounits 2>$null
        }
        if ($state.status -in @("complete", "failed", "prepared")) { break }
    }
    if (-not $Once) { Start-Sleep -Seconds $IntervalSeconds }
} while (-not $Once)
