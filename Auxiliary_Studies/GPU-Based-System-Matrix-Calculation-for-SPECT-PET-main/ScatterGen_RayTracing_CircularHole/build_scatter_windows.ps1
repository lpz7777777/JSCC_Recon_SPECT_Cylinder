param(
    [string]$Output = "ScatterGen_CircularHole_detector_local.exe",
    [string]$Architecture = "sm_89"
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$engineRoot = Split-Path $scriptDir -Parent
$shimDir = Join-Path $engineRoot "tests\nvcc_windows_posix_shim"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere.exe was not found. Install Visual Studio C++ build tools."
}
$visualStudio = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $visualStudio) {
    throw "A Visual Studio installation with C++ tools was not found."
}
$vcvars = Join-Path $visualStudio "VC\Auxiliary\Build\vcvars64.bat"
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    $Output
} else {
    Join-Path $scriptDir $Output
}
$command = @(
    'call "{0}" >nul && nvcc -std=c++17 -O3 -lineinfo -arch={1}'
    '-I"{2}" "{3}" "{4}" "{5}" -o "{6}"'
) -join ' '
$command = $command -f `
    $vcvars, $Architecture, $shimDir, `
    (Join-Path $scriptDir "scatter.cu"), `
    (Join-Path $scriptDir "ScatterGen_CircularHole.cpp"), `
    (Join-Path $shimDir "posix_stubs.cpp"), `
    $outputPath
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "ScatterGen build failed with exit code $LASTEXITCODE."
}
Write-Output "Built $outputPath for $Architecture"
