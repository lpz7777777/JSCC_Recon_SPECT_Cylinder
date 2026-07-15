param(
    [Parameter(Mandatory = $true)]
    [int]$ScatterProcessId,

    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,

    [Parameter(Mandatory = $true)]
    [string]$ValidatorPath
)

$ErrorActionPreference = "Stop"
$runDirectoryResolved = (Resolve-Path -LiteralPath $RunDirectory).Path
$validatorResolved = (Resolve-Path -LiteralPath $ValidatorPath).Path
$reportDirectory = Join-Path $runDirectoryResolved "validation_report"
New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
$logPath = Join-Path $reportDirectory "validator.log"
$exitCodePath = Join-Path $reportDirectory "validator.exitcode"
$startedPath = Join-Path $reportDirectory "validator.started"

Set-Content -LiteralPath $startedPath -Encoding ascii -Value (
    "{0:o} waiting_for_pid={1}" -f (Get-Date), $ScatterProcessId
)

try {
    Wait-Process -Id $ScatterProcessId -ErrorAction SilentlyContinue
    $stderrPath = Join-Path $reportDirectory "validator.stderr.log"
    $process = Start-Process -FilePath python -ArgumentList @(
        $validatorResolved,
        $runDirectoryResolved
    ) -NoNewWindow -PassThru -Wait `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $stderrPath
    $exitCode = $process.ExitCode
}
catch {
    $_ | Out-String | Set-Content -LiteralPath $logPath -Encoding utf8
    $exitCode = 1
}

Set-Content -LiteralPath $exitCodePath -Encoding ascii -Value $exitCode
exit $exitCode
