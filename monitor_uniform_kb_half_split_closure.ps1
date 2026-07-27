param(
    [string]$OutputName = "440keV_RotateNum20_UniformFullFOV_5e10_KB_HalfSplit_ER13_FWHM511_Sum350"
)

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$runDir = Join-Path $repo "Auxiliary_Studies/Sensitivity_SPECT_PolarCoor/Result/$OutputName"
$log = Join-Path $runDir "run.log"
$halfA = Join-Path $runDir "CalibrationHalfA"
$halfB = Join-Path $runDir "IndependentHalfB_Closure"

Write-Host "Run directory: $runDir"
if (Test-Path -LiteralPath $log) {
    Get-Content -LiteralPath $log -Tail 30
} else {
    Write-Host "run.log has not been created yet."
}
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime,Path | Format-Table -AutoSize
foreach ($path in @($halfA, $halfB)) {
    if (Test-Path -LiteralPath $path) {
        Get-ChildItem -LiteralPath $path -File | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
    }
}
