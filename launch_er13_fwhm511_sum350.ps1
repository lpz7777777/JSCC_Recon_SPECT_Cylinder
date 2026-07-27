$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$tag = "ER13_FWHM511_Sum350"
$pipeline = Join-Path $repo "Results\Pipelines\FullSensiD_Then_ComptonValidation_20260722_$tag"
New-Item -ItemType Directory -Force -Path $pipeline | Out-Null
$stdout = Join-Path $pipeline "pipeline_stdout.log"
$stderr = Join-Path $pipeline "pipeline_stderr.log"
Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
$script = Join-Path $repo "run_full_sensid_and_compton_validation.ps1"
$argumentList = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script,
    "-RepositoryRoot", $repo, "-ExperimentTag", $tag
)
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $argumentList `
    -WorkingDirectory $repo -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath (Join-Path $pipeline "pipeline.pid") `
    -Value $process.Id -Encoding ascii
Write-Output "Started $tag pipeline PID=$($process.Id)"
