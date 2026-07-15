[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Configuration = "Release",

    [string]$Macro = "smoke.mac",

    [ValidateRange(1, 256)]
    [int]$Threads = 2,

    [string]$BuildDirectory = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "CMakeCache.txt") -PathType Leaf) {
        $BuildDirectory = $PSScriptRoot
    } else {
        $BuildDirectory = Join-Path $PSScriptRoot "build"
    }
}
$build = [System.IO.Path]::GetFullPath($BuildDirectory)
$cache = Join-Path $build "CMakeCache.txt"
if (-not (Test-Path -LiteralPath $cache -PathType Leaf)) {
    throw "CMake cache not found: $cache`nConfigure the project before running this script."
}

$executableCandidates = @(
    (Join-Path $build "gagg_intrinsic.exe"),
    (Join-Path (Join-Path $build $Configuration) "gagg_intrinsic.exe")
)
$executable = $executableCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $executable) {
    $expected = $executableCandidates -join "`n  "
    throw "gagg_intrinsic.exe was not found. Build configuration '$Configuration'. Expected:`n  $expected"
}

$macroCandidates = @(
    (Join-Path $build $Macro),
    (Join-Path (Join-Path $PSScriptRoot "macros") $Macro)
)
$macroPath = $macroCandidates |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $macroPath) {
    throw "Macro not found in the build or source macro directory: $Macro"
}

# Geant4_DIR normally ends in lib/cmake/Geant4. Import geant4.bat when it is
# available so both runtime DLL and physics-data variables are configured.
$geant4Entry = Get-Content -LiteralPath $cache |
    Where-Object { $_ -match '^Geant4_DIR:(?:PATH|UNINITIALIZED)=(.+)$' } |
    Select-Object -First 1
if ($geant4Entry -and $geant4Entry -match '^Geant4_DIR:(?:PATH|UNINITIALIZED)=(.+)$') {
    $geant4CmakeDirectory = $Matches[1]
    $geant4Prefix = [System.IO.Path]::GetFullPath(
        (Join-Path $geant4CmakeDirectory "..\..\..")
    )
    $geant4Bin = Join-Path $geant4Prefix "bin"
    if (Test-Path -LiteralPath $geant4Bin -PathType Container) {
        $geant4Setup = Join-Path $geant4Bin "geant4.bat"
        if (Test-Path -LiteralPath $geant4Setup -PathType Leaf) {
            $environmentLines = & cmd.exe /d /c "call `"$geant4Setup`" >nul && set"
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to initialize the Geant4 environment with $geant4Setup"
            }
            foreach ($line in $environmentLines) {
                if ($line -match '^([^=]+)=(.*)$') {
                    [Environment]::SetEnvironmentVariable(
                        $Matches[1], $Matches[2], "Process"
                    )
                }
            }
            Write-Host "Imported Geant4 environment: $geant4Setup"
        } else {
            $env:PATH = "$geant4Bin;$env:PATH"
            Write-Host "Geant4 DLL directory: $geant4Bin"
            Write-Warning "geant4.bat was not found; Geant4 dataset variables must already be configured."
        }
    } else {
        Write-Warning "Could not infer the Geant4 bin directory from Geant4_DIR=$geant4CmakeDirectory"
    }
} else {
    Write-Warning "Geant4_DIR was not found in CMakeCache.txt; Geant4 DLLs must already be on PATH."
}

Write-Host "Executable: $executable"
Write-Host "Macro:      $macroPath"
Write-Host "Threads:    $Threads"
Write-Host "Output dir: $build"

Push-Location $build
try {
    & $executable $macroPath $Threads
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    $unsignedExitCode = [BitConverter]::ToUInt32(
        [BitConverter]::GetBytes([int]$exitCode), 0
    )
    $hexExitCode = "0x{0:X8}" -f $unsignedExitCode
    if ($unsignedExitCode -eq 0xC0000135) {
        throw "The executable could not load a DLL ($hexExitCode). Run from a Geant4 command prompt or add the Geant4 bin directory to PATH."
    }
    throw "gagg_intrinsic exited with code $exitCode ($hexExitCode)."
}

$expectedOutput = if ($Macro -eq "smoke.mac") {
    Join-Path $build "smoke_GAGG_3x3x3_218keV.csv"
} else {
    $null
}
if ($expectedOutput -and -not (Test-Path -LiteralPath $expectedOutput -PathType Leaf)) {
    throw "The run returned success but the expected smoke CSV is missing: $expectedOutput"
}
if ($expectedOutput) {
    Write-Host "Created: $expectedOutput"
}
