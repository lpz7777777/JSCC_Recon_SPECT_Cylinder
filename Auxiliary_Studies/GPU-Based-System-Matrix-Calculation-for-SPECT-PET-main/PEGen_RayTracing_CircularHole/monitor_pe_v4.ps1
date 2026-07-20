param(
    [string]$ProgressPath = "PE_v4_progress.json",
    [int]$IntervalSeconds = 10,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
if ($IntervalSeconds -lt 1) {
    throw "IntervalSeconds must be positive."
}

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

function Format-Duration([double]$Seconds) {
    if ([double]::IsNaN($Seconds) -or [double]::IsInfinity($Seconds) -or
        $Seconds -lt 0) {
        return "unknown"
    }
    $span = [TimeSpan]::FromSeconds($Seconds)
    if ($span.TotalDays -ge 1) {
        return ("{0}d {1:hh\:mm\:ss}" -f [math]::Floor($span.TotalDays), $span)
    }
    return $span.ToString("hh\:mm\:ss")
}

$resolved = [System.IO.Path]::GetFullPath($ProgressPath)
do {
    if (-not (Test-Path -LiteralPath $resolved)) {
        Write-Output "Waiting for $resolved"
    } else {
        try {
            $state = Read-SharedText $resolved | ConvertFrom-Json
            $fraction = if ($state.total_rows -gt 0) {
                [double]$state.completed_rows / [double]$state.total_rows
            } else { 0.0 }
            $gpu = & nvidia-smi `
                --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu `
                --format=csv,noheader,nounits 2>$null
            Write-Output ("[{0}] status={1} rows={2}/{3} ({4:P2}) rate={5:N3} M element/s ETA={6}" -f `
                $state.last_update, $state.status, $state.completed_rows, $state.total_rows, `
                $fraction, ([double]$state.elements_per_second / 1e6), `
                (Format-Duration ([double]$state.eta_seconds)))
            Write-Output ("  detector={0} rotation={1} nonzero={2} raw_sum={3:E6} windowed_sum={4:E6}" -f `
                $state.current_detector, $state.current_rotation, $state.nonzero_elements, `
                [double]$state.unwindowed_sum, [double]$state.windowed_sum)
            if ($gpu) {
                Write-Output "  GPU: $gpu"
            }
            if ($state.status -in @("complete", "failed")) {
                break
            }
        } catch {
            Write-Output "Progress file is being updated; retrying: $($_.Exception.Message)"
        }
    }
    if (-not $Once) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while (-not $Once)
