param(
    [int]$CudaId = 0,
    [int]$FaceSubdivisions = 16,
    [int]$RowsPerChunk = 4,
    [int]$SamplesPerLaunch = 32,
    [switch]$SkipPE,
    [switch]$SkipScatter
)

$ErrorActionPreference = "Stop"
$engineRoot = $PSScriptRoot
$runsRoot = Join-Path $engineRoot "runs"
$peToolDir = Join-Path $engineRoot "PEGen_RayTracing_CircularHole"
$peExecutable = Join-Path $peToolDir "PEGen_V4_Production.exe"
$scatterExecutable = Join-Path $engineRoot `
    "ScatterGen_RayTracing_CircularHole\ScatterGen_CircularHole_detector_local.exe"
$pipelineStatus = Join-Path $runsRoot "PEV4_JSCC_pipeline_status.json"
$expectedMatrixBytes = [int64]11520 * 52020 * 4
$expectedPEModel = "PE_v4_visible_surface_symmetric_halton_layer_grid"

$cases = @(
    [pscustomobject]@{
        Name = "JSCC_218keV_pe_v4"
        Source = "JSCC_218keV"
        PEFrom = "self"
        ScatterStage = "scatter_218"
        Combined = $true
    },
    [pscustomobject]@{
        Name = "JSCC_440keV_pe_v4"
        Source = "JSCC_440keV"
        PEFrom = "self"
        ScatterStage = "scatter_440"
        Combined = $true
    },
    [pscustomobject]@{
        Name = "JSCC_440keV_to_218keVwin_pe_v4"
        Source = "JSCC_440keV_to_218keVwin"
        PEFrom = "JSCC_440keV_pe_v4"
        ScatterStage = "scatter_440_to_218"
        Combined = $false
    }
)

function Write-PipelineStatus(
    [string]$Stage,
    [string]$Status,
    [string]$Message,
    [string]$RunDirectory = "",
    [string]$LogPath = ""
) {
    $state = [ordered]@{
        schema_version = 1
        stage = $Stage
        status = $Status
        message = $Message
        run_directory = $RunDirectory
        log_path = $LogPath
        last_update = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")
        face_subdivisions = $FaceSubdivisions
        rows_per_chunk = $RowsPerChunk
        samples_per_launch = $SamplesPerLaunch
        cuda_id = $CudaId
    }
    $temporary = "$pipelineStatus.tmp"
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $pipelineStatus -Force
}

function Assert-Matrix([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing matrix: $Path"
    }
    $bytes = (Get-Item -LiteralPath $Path).Length
    if ($bytes -ne $expectedMatrixBytes) {
        throw "Wrong matrix size for ${Path}: $bytes; expected $expectedMatrixBytes"
    }
}

function Assert-PEManifest([string]$RunName) {
    $manifestPath = Join-Path (Join-Path $runsRoot $RunName) `
        "PE_v4_manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Missing PE v4 manifest for ${RunName}: $manifestPath"
    }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    } catch {
        throw "Cannot parse PE v4 manifest for ${RunName}: $($_.Exception.Message)"
    }
    if ($manifest.model -ne $expectedPEModel) {
        throw "Stale PE v4 model in ${RunName}: '$($manifest.model)'; " + `
            "expected '$expectedPEModel'. Archive all three legacy _pe_v4 run " + `
            "directories before regenerating; do not mix old scatter with new PE."
    }
    if ([int]$manifest.face_subdivisions -ne $FaceSubdivisions) {
        throw "PE v4 face subdivision mismatch in ${RunName}: " + `
            "$($manifest.face_subdivisions); requested $FaceSubdivisions"
    }
}

function Prepare-RunDirectories {
    foreach ($case in $cases) {
        $source = Join-Path $runsRoot $case.Source
        $target = Join-Path $runsRoot $case.Name
        if (-not (Test-Path -LiteralPath $source -PathType Container)) {
            throw "Missing source run directory: $source"
        }
        New-Item -ItemType Directory -Path $target -Force | Out-Null
        Get-ChildItem -LiteralPath $source -File | Where-Object {
            $_.Name -like "Params_*.dat" -or $_.Name -eq "Params_README.txt"
        } | ForEach-Object {
            $destination = Join-Path $target $_.Name
            Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
        }
    }
}

function Invoke-ParameterGeneration {
    $generatorDir = Join-Path $engineRoot "FileGenerater_3D_Unified"
    Write-PipelineStatus "generate_params" "running" `
        "Regenerating density-aligned JSCC 218/440 response parameters"
    Push-Location $engineRoot
    try {
        & matlab -batch `
            "addpath('$generatorDir'); generate_jscc_218_440_response_params;"
        if ($LASTEXITCODE -ne 0) {
            throw "JSCC parameter generation failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

function Assert-MaterialDensityAlignment {
    $projectRoot = Split-Path (Split-Path $engineRoot -Parent) -Parent
    $validator = Join-Path $engineRoot `
        "tests\validate_jscc_material_density_alignment.py"
    Write-PipelineStatus "validate_materials" "running" `
        "Checking Geant4, XCOM, CUDA, and Params density alignment"
    & python $validator --repo $projectRoot --run-suffix "_pe_v4"
    if ($LASTEXITCODE -ne 0) {
        throw "JSCC material-density alignment validation failed"
    }
}

function Invoke-PEV4([string]$RunName) {
    $runDir = Join-Path $runsRoot $RunName
    $raw = Join-Path $runDir "PE_SysMat_shift_0.000000_0.000000_0.000000_v4.sysmat"
    $windowed = Join-Path $runDir "PE_Windowed_SysMat_shift_0.000000_0.000000_0.000000_v4.sysmat"
    if ((Test-Path -LiteralPath $raw) -and (Test-Path -LiteralPath $windowed)) {
        Assert-Matrix $raw
        Assert-Matrix $windowed
        Assert-PEManifest $RunName
        Write-Output "PE v4 already complete: $RunName"
        return
    }
    $progress = Join-Path $runDir "PE_v4_progress.json"
    $log = Join-Path $runDir "PE_v4_console.log"
    $arguments = @(
        "--cuda", $CudaId,
        "--face-subdiv", $FaceSubdivisions,
        "--rows-per-chunk", $RowsPerChunk,
        "--samples-per-launch", $SamplesPerLaunch,
        "--output-unwindowed", $raw,
        "--output-windowed", $windowed,
        "--progress", $progress,
        "--log", (Join-Path $runDir "PE_v4_progress.tsv"),
        "--manifest", (Join-Path $runDir "PE_v4_manifest.json")
    )
    if ((Test-Path -LiteralPath "$raw.partial") -or
        (Test-Path -LiteralPath "$windowed.partial")) {
        $arguments += "--resume"
    }
    Write-PipelineStatus "pe_v4" "running" "Generating $RunName" $runDir $log
    Push-Location $runDir
    try {
        & $peExecutable @arguments 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) {
            throw "PE v4 failed for $RunName with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    Assert-Matrix $raw
    Assert-Matrix $windowed
}

function Invoke-ScatterV4([pscustomobject]$Case) {
    $runDir = Join-Path $runsRoot $Case.Name
    $peRun = if ($Case.PEFrom -eq "self") { $Case.Name } else { $Case.PEFrom }
    $pePath = Join-Path (Join-Path $runsRoot $peRun) `
        "PE_SysMat_shift_0.000000_0.000000_0.000000_v4.sysmat"
    Assert-Matrix $pePath
    Assert-PEManifest $peRun
    $scatter = Join-Path $runDir `
        "Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat"
    $combined = Join-Path $runDir `
        "SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat"
    if (Test-Path -LiteralPath $scatter) {
        Assert-Matrix $scatter
        if (-not $Case.Combined -or (Test-Path -LiteralPath $combined)) {
            if ($Case.Combined) { Assert-Matrix $combined }
            $peTime = (Get-Item -LiteralPath $pePath).LastWriteTimeUtc
            $scatterTime = (Get-Item -LiteralPath $scatter).LastWriteTimeUtc
            $combinedIsCurrent = -not $Case.Combined -or `
                (Get-Item -LiteralPath $combined).LastWriteTimeUtc -ge $peTime
            if ($scatterTime -ge $peTime -and $combinedIsCurrent) {
                Write-Output "Scatter already complete: $($Case.Name)"
                return
            }
            throw "Stale scatter output in $($Case.Name): its timestamp predates " + `
                "the current PE v4 matrix. Archive the run directory and regenerate."
        }
    }
    $log = Join-Path $runDir "ScatterGen_pe_v4.log"
    Write-PipelineStatus $Case.ScatterStage "running" `
        "Generating scatter response for $($Case.Name)" $runDir $log
    Push-Location $runDir
    try {
        & $scatterExecutable -PE $pePath -cuda $CudaId 2>&1 |
            Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) {
            throw "ScatterGen failed for $($Case.Name) with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    Assert-Matrix $scatter
    if ($Case.Combined) { Assert-Matrix $combined }
    $sourceManifest = Join-Path (Join-Path $runsRoot $peRun) "PE_v4_manifest.json"
    $targetManifest = Join-Path $runDir "PE_v4_manifest.json"
    $sourceManifestFull = [System.IO.Path]::GetFullPath($sourceManifest)
    $targetManifestFull = [System.IO.Path]::GetFullPath($targetManifest)
    if ((Test-Path -LiteralPath $sourceManifest) -and
        -not $sourceManifestFull.Equals(
            $targetManifestFull,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
        Copy-Item -LiteralPath $sourceManifest -Destination `
            $targetManifest -Force
    }
}

try {
    Invoke-ParameterGeneration
    Write-PipelineStatus "prepare" "running" "Preparing PE v4 JSCC run directories"
    Prepare-RunDirectories
    Assert-MaterialDensityAlignment
    if (-not $SkipPE) {
        Write-PipelineStatus "build_pe_v4" "running" `
            "Rebuilding PE v4 from the current source"
        & (Join-Path $peToolDir "build_pe_v4_production.ps1")
        Invoke-PEV4 "JSCC_218keV_pe_v4"
        Invoke-PEV4 "JSCC_440keV_pe_v4"
    }
    if (-not $SkipScatter) {
        Write-PipelineStatus "build_scatter" "running" `
            "Rebuilding ScatterGen with the current XCOM header"
        & (Join-Path $engineRoot `
            "ScatterGen_RayTracing_CircularHole\build_scatter_windows.ps1")
        foreach ($case in $cases) {
            Invoke-ScatterV4 $case
        }
    }
    if ($SkipPE -and $SkipScatter) {
        Write-PipelineStatus "prepared" "prepared" `
            "Run directories and material-density validation are ready"
        Write-Output "PE v4 JSCC inputs and material-density validation are ready."
    } else {
        Write-PipelineStatus "complete" "complete" "All PE v4 JSCC matrices completed"
        Write-Output "All PE v4 JSCC matrices completed."
    }
} catch {
    Write-PipelineStatus "failed" "failed" $_.Exception.Message
    throw
}
