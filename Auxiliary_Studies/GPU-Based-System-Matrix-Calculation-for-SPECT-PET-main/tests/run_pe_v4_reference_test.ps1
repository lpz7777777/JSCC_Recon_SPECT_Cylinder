$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
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
$build = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("pe-v4-reference-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $build | Out-Null
try {
    $source = Join-Path $root "tests\pe_v4_reference_test.cpp"
    $executable = Join-Path $build "pe_v4_reference_test.exe"
    $compile = 'call "{0}" >nul && cl /nologo /std:c++17 /O2 /EHsc /I"{1}" "{2}" /Fe:"{3}"' -f `
        $vcvars, $root, $source, $executable
    & $env:ComSpec /d /s /c $compile
    if ($LASTEXITCODE -ne 0) {
        throw "PE v4 reference test compilation failed with exit code $LASTEXITCODE."
    }
    & $executable
    if ($LASTEXITCODE -ne 0) {
        throw "PE v4 reference test failed with exit code $LASTEXITCODE."
    }
}
finally {
    Remove-Item -LiteralPath $build -Recurse -Force -ErrorAction SilentlyContinue
}
