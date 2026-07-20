param(
    [string]$Output = "PEGen_V4_Production.exe",
    [string]$Architecture = "sm_89"
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
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
$source = Join-Path $scriptDir "PEGen_V4_Production.cu"
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    $Output
} else {
    Join-Path $scriptDir $Output
}
$command = 'call "{0}" >nul && nvcc -std=c++17 -O3 -lineinfo -arch={1} "{2}" -o "{3}"' -f `
    $vcvars, $Architecture, $source, $outputPath
& $env:ComSpec /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "PE v4 production build failed with exit code $LASTEXITCODE."
}
Write-Output "Built $outputPath for $Architecture"
