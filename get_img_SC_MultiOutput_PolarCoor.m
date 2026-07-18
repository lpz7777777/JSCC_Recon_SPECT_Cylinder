function outputPaths = get_img_SC_MultiOutput_PolarCoor(folderPath)
%GET_IMG_SC_MULTIOUTPUT_POLARCOOR Display local multi-energy CntStat results.
%
% outputPaths = get_img_SC_MultiOutput_PolarCoor()
% outputPaths = get_img_SC_MultiOutput_PolarCoor(runFolderOrPolarFolder)
%
% The function reads run_manifest.json, discovers every reconstruction task,
% converts selected polar-coordinate snapshots to Cartesian volumes, and saves:
%   Display/mip_comparison.png     - task comparison over selected iterations
%   Display/final_orthogonal.png   - final transverse/coronal/sagittal views
%
% Cartesian interpolation geometry is precomputed once and reused for every
% task, z layer, and iteration. This avoids repeated griddata triangulation.

if nargin < 1 || strlength(string(folderPath)) == 0
    folderPath = uigetdir("./Results/Reconstruction/Figure_Local_SC_MultiOutput", ...
        "Select a reconstruction run folder or its Polar folder");
    if isequal(folderPath, 0)
        outputPaths = struct();
        return;
    end
end

cfg = default_display_config();
[runPath, polarPath, manifestPath] = resolve_run_paths(folderPath);
manifest = jsondecode(fileread(manifestPath));
validate_manifest(manifest, polarPath);

repoRoot = fileparts(mfilename("fullpath"));
factorPath = resolve_reference_factor_path(repoRoot, manifest);
coorPolar = load_named_array( ...
    fullfile(factorPath, "coor_polar.mat"), ...
    fullfile(factorPath, "coor_polar.csv"), ...
    "coor_polar");

pixelNumPolar = size(coorPolar, 1);
pixelNumZ = double(manifest.arguments.pixel_num_z);
pixelNum = double(manifest.pixel_num);
if pixelNumPolar * pixelNumZ ~= pixelNum
    error("Pixel shape mismatch: coor_polar rows=%d, z=%d, manifest pixel_num=%d.", ...
        pixelNumPolar, pixelNumZ, pixelNum);
end

[gridX, gridY] = meshgrid(cfg.xCenters, cfg.yCenters);
interpPlan = build_interpolation_plan(coorPolar(:, 1:2), gridX, gridY);

tasks = manifest.tasks;
taskNum = numel(tasks);
if taskNum < 1
    error("The manifest contains no reconstruction tasks: %s", manifestPath);
end

[iterValues, iterIndices] = choose_display_iterations(tasks, cfg.maxIterationRows);
iterNum = numel(iterValues);
volumes = cell(taskNum, iterNum);
taskLabels = strings(1, taskNum);

fprintf("Reading %d tasks from %s\n", taskNum, polarPath);
fprintf("Selected iterations: %s\n", mat2str(iterValues));

for taskIdx = 1:taskNum
    task = tasks(taskIdx);
    taskLabels(taskIdx) = format_task_label(task.output_file);
    historyPath = fullfile(polarPath, string(task.history_file));
    historyShape = double(task.history_shape(:)).';
    if numel(historyShape) ~= 2 || historyShape(2) ~= pixelNum
        error("Unexpected history shape for %s: %s", historyPath, mat2str(historyShape));
    end

    history = read_float32_tensor(historyPath, [pixelNumPolar, pixelNumZ, historyShape(1)]);
    for iterIdx = 1:iterNum
        snapshotIdx = iterIndices(iterIdx);
        fprintf("  Task %d/%d, iteration %d\n", taskIdx, taskNum, iterValues(iterIdx));
        volumes{taskIdx, iterIdx} = polar_to_cartesian_volume( ...
            history(:, :, snapshotIdx), interpPlan);
    end
end

displayPath = fullfile(runPath, "Display");
ensure_dir(displayPath);

mipFigure = render_mip_comparison( ...
    volumes, taskLabels, iterValues, cfg, manifest);
mipPngPath = fullfile(displayPath, "mip_comparison.png");
mipFigPath = fullfile(displayPath, "mip_comparison.fig");
exportgraphics(mipFigure, mipPngPath, "Resolution", cfg.exportDpi);
savefig(mipFigure, mipFigPath);

orthogonalFigure = render_final_orthogonal( ...
    volumes(:, end), taskLabels, cfg, manifest);
orthogonalPngPath = fullfile(displayPath, "final_orthogonal.png");
orthogonalFigPath = fullfile(displayPath, "final_orthogonal.fig");
exportgraphics(orthogonalFigure, orthogonalPngPath, "Resolution", cfg.exportDpi);
savefig(orthogonalFigure, orthogonalFigPath);

outputPaths = struct( ...
    "mip_png", string(mipPngPath), ...
    "mip_fig", string(mipFigPath), ...
    "orthogonal_png", string(orthogonalPngPath), ...
    "orthogonal_fig", string(orthogonalFigPath));

fprintf("Saved display outputs to %s\n", displayPath);
end


function cfg = default_display_config()
cfg = struct();
cfg.pixelNumX = 100;
cfg.pixelNumY = 100;
cfg.pixelLX = 3;
cfg.pixelLY = 3;
cfg.pixelLZ = 3;
cfg.xCenters = centered_axis(cfg.pixelNumX, cfg.pixelLX);
cfg.yCenters = centered_axis(cfg.pixelNumY, cfg.pixelLY);
cfg.maxIterationRows = 6;
cfg.exportDpi = 220;
cfg.displayPercentile = 99.5;
cfg.colorMap = flipud(gray(1024));
end


function axisCenters = centered_axis(pixelNum, pixelLength)
axisCenters = (-pixelNum * pixelLength / 2 + pixelLength / 2) : ...
    pixelLength : (pixelNum * pixelLength / 2 - pixelLength / 2);
end


function [runPath, polarPath, manifestPath] = resolve_run_paths(folderPath)
folderPath = char(string(folderPath));
if ~exist(folderPath, "dir")
    error("Result directory does not exist: %s", folderPath);
end

manifestHere = fullfile(folderPath, "run_manifest.json");
manifestUnderPolar = fullfile(folderPath, "Polar", "run_manifest.json");
if exist(manifestHere, "file")
    polarPath = folderPath;
    runPath = fileparts(folderPath);
    manifestPath = manifestHere;
elseif exist(manifestUnderPolar, "file")
    runPath = folderPath;
    polarPath = fullfile(folderPath, "Polar");
    manifestPath = manifestUnderPolar;
else
    error("Cannot find run_manifest.json in %s or its Polar subdirectory.", folderPath);
end
end


function validate_manifest(manifest, polarPath)
requiredFields = ["pixel_num", "arguments", "tasks"];
for fieldName = requiredFields
    if ~isfield(manifest, fieldName)
        error("Manifest is missing field '%s': %s", fieldName, polarPath);
    end
end
if ~isfield(manifest.arguments, "e0_list") || ...
        ~isfield(manifest.arguments, "rotate_num") || ...
        ~isfield(manifest.arguments, "pixel_num_z")
    error("Manifest arguments do not contain energy, rotation, and z-size metadata.");
end

tasks = manifest.tasks;
for taskIdx = 1:numel(tasks)
    for fieldName = ["output_file", "history_file", "history_shape", ...
            "iterations", "save_iter_step"]
        if ~isfield(tasks(taskIdx), fieldName)
            error("Task %d is missing manifest field '%s'.", taskIdx, fieldName);
        end
    end
    historyPath = fullfile(polarPath, string(tasks(taskIdx).history_file));
    if ~exist(historyPath, "file")
        error("Cannot find task history file: %s", historyPath);
    end
end
end


function factorPath = resolve_reference_factor_path(repoRoot, manifest)
energyList = double(manifest.arguments.e0_list(:));
energyKeV = round(1000 * energyList(1));
rotateNum = double(manifest.arguments.rotate_num);
suffix = "";
if isfield(manifest.arguments, "factor_dir_suffix")
    suffix = string(manifest.arguments.factor_dir_suffix);
end

factorRoot = fullfile(repoRoot, "Factors");
if isfield(manifest.arguments, "factors_dir")
    configuredRoot = string(manifest.arguments.factors_dir);
    if is_absolute_path(configuredRoot)
        factorRoot = char(configuredRoot);
    else
        factorRoot = fullfile(repoRoot, configuredRoot);
    end
end

dirName = sprintf("%dkeV_RotateNum%d", energyKeV, rotateNum);
if strlength(suffix) > 0
    dirName = dirName + "_" + strip(suffix, "left", "_");
end
factorPath = fullfile(factorRoot, dirName);
if ~exist(factorPath, "dir")
    error("Cannot find reference factor directory: %s", factorPath);
end
end


function result = is_absolute_path(pathValue)
pathValue = char(string(pathValue));
result = ~isempty(regexp(pathValue, "^(?:[A-Za-z]:[\\/]|[\\/]{2}|/)", "once"));
end


function plan = build_interpolation_plan(sourceXY, gridX, gridY)
triangulation = delaunayTriangulation(double(sourceXY));
queryXY = [gridX(:), gridY(:)];
[triangleIds, barycentricWeights] = pointLocation(triangulation, queryXY);
validMask = ~isnan(triangleIds);
validTriangleIds = triangleIds(validMask);

plan = struct();
plan.outputSize = size(gridX);
plan.validMask = validMask;
plan.vertexIds = triangulation.ConnectivityList(validTriangleIds, :);
plan.weights = single(barycentricWeights(validMask, :));
end


function volumeCartesian = polar_to_cartesian_volume(volumePolar, plan)
pixelNumZ = size(volumePolar, 2);
volumeCartesian = zeros( ...
    plan.outputSize(2), plan.outputSize(1), pixelNumZ, "single");

for zIdx = 1:pixelNumZ
    values = volumePolar(:, zIdx);
    interpolated = zeros(prod(plan.outputSize), 1, "single");
    interpolated(plan.validMask) = sum( ...
        values(plan.vertexIds) .* plan.weights, 2);
    volumeCartesian(:, :, zIdx) = reshape(interpolated, plan.outputSize).';
end
end


function [iterValues, iterIndices] = choose_display_iterations(tasks, maxRows)
iterMax = double(tasks(1).iterations);
saveStep = double(tasks(1).save_iter_step);
for taskIdx = 2:numel(tasks)
    if double(tasks(taskIdx).iterations) ~= iterMax || ...
            double(tasks(taskIdx).save_iter_step) ~= saveStep
        error(["Display comparison currently requires all tasks to use the same " ...
            "iteration count and save interval."]);
    end
end

savedValues = saveStep:saveStep:iterMax;
if numel(savedValues) <= maxRows
    iterIndices = 1:numel(savedValues);
else
    targetValues = unique(round(logspace( ...
        log10(saveStep), log10(iterMax), maxRows) / saveStep) * saveStep);
    targetValues = targetValues(targetValues >= saveStep & targetValues <= iterMax);
    targetValues = unique([targetValues, iterMax]);
    if numel(targetValues) > maxRows
        targetValues = targetValues(end - maxRows + 1:end);
    end
    [~, iterIndices] = ismember(targetValues, savedValues);
end
iterValues = savedValues(iterIndices);
end


function label = format_task_label(outputFile)
label = erase(string(outputFile), "Image_");
label = replace(label, "S_(", "S: ");
label = replace(label, ")keV", " keV");
label = replace(label, "S_", "S: ");
label = replace(label, "_", " + ");
end


function figureHandle = render_mip_comparison(volumes, taskLabels, iterValues, cfg, manifest)
[taskNum, iterNum] = size(volumes);
figureHandle = figure("Color", "white", "Position", ...
    [80, 60, 320 * taskNum, 250 * iterNum]);
layout = tiledlayout(iterNum, taskNum, "TileSpacing", "compact", "Padding", "compact");

for iterIdx = 1:iterNum
    for taskIdx = 1:taskNum
        mip = flip(max(volumes{taskIdx, iterIdx}, [], 3).', 1);
        displayMax = robust_display_max(mip, cfg.displayPercentile);
        axisHandle = nexttile(layout);
        imagesc(axisHandle, cfg.yCenters, cfg.xCenters, mip, [0, displayMax]);
        axis(axisHandle, "image");
        colormap(axisHandle, cfg.colorMap);
        if iterIdx == 1
            title(axisHandle, taskLabels(taskIdx), "Interpreter", "none");
        end
        if taskIdx == 1
            ylabel(axisHandle, sprintf("Iter %d\nx (mm)", iterValues(iterIdx)));
        else
            axisHandle.YTickLabel = [];
        end
        if iterIdx == iterNum
            xlabel(axisHandle, "y (mm)");
        else
            axisHandle.XTickLabel = [];
        end
    end
end
title(layout, build_run_title(manifest, "Transverse MIP comparison"), ...
    "Interpreter", "none");
end


function figureHandle = render_final_orthogonal(finalVolumes, taskLabels, cfg, manifest)
taskNum = numel(finalVolumes);
figureHandle = figure("Color", "white", "Position", [100, 80, 960, 270 * taskNum]);
layout = tiledlayout(taskNum, 3, "TileSpacing", "compact", "Padding", "compact");

for taskIdx = 1:taskNum
    volume = finalVolumes{taskIdx};
    xCenterIdx = round((size(volume, 1) + 1) / 2);
    yCenterIdx = round((size(volume, 2) + 1) / 2);

    transverse = flip(max(volume, [], 3).', 1);
    coronal = flip(squeeze(volume(:, yCenterIdx, :)), 1);
    sagittal = flip(squeeze(volume(xCenterIdx, :, :)), 1);
    displayMax = robust_display_max(transverse, cfg.displayPercentile);

    axisTransverse = nexttile(layout);
    imagesc(axisTransverse, cfg.yCenters, cfg.xCenters, transverse, [0, displayMax]);
    axis(axisTransverse, "image");
    colormap(axisTransverse, cfg.colorMap);
    ylabel(axisTransverse, taskLabels(taskIdx), "Interpreter", "none");
    if taskIdx == 1
        title(axisTransverse, "Transverse MIP");
    end

    zCenters = centered_axis(size(volume, 3), cfg.pixelLZ);
    axisCoronal = nexttile(layout);
    imagesc(axisCoronal, zCenters, cfg.xCenters, coronal, [0, displayMax]);
    axis(axisCoronal, "image");
    colormap(axisCoronal, cfg.colorMap);
    if taskIdx == 1
        title(axisCoronal, "Coronal center");
    end

    axisSagittal = nexttile(layout);
    imagesc(axisSagittal, zCenters, cfg.yCenters, sagittal, [0, displayMax]);
    axis(axisSagittal, "image");
    colormap(axisSagittal, cfg.colorMap);
    if taskIdx == 1
        title(axisSagittal, "Sagittal center");
    end
end
title(layout, build_run_title(manifest, "Final orthogonal views"), ...
    "Interpreter", "none");
end


function displayMax = robust_display_max(image, percentileValue)
positiveValues = double(image(image > 0 & isfinite(image)));
if isempty(positiveValues)
    displayMax = 1;
    return;
end
displayMax = prctile(positiveValues, percentileValue);
if ~isfinite(displayMax) || displayMax <= 0
    displayMax = max(positiveValues);
end
end


function titleText = build_run_title(manifest, prefix)
energyKeV = round(1000 * double(manifest.energies_MeV(:))).';
energyText = strjoin(string(energyKeV), " + ");
countText = extract_count_level(manifest.input_cntstat_files(1));
titleText = sprintf("%s | %s keV | %s | OSEM%d", ...
    prefix, energyText, countText, double(manifest.arguments.osem_subset_num));
end


function countText = extract_count_level(inputPath)
tokens = regexp(string(inputPath), "_(\d+(?:\.\d+)?e[+-]?\d+)\.csv$", "tokens", "once");
if isempty(tokens)
    countText = "unknown count";
else
    countText = string(tokens{1});
end
end


function tensor = read_float32_tensor(filePath, tensorShape)
fileInfo = dir(filePath);
expectedBytes = prod(tensorShape) * 4;
if isempty(fileInfo) || fileInfo.bytes ~= expectedBytes
    actualBytes = -1;
    if ~isempty(fileInfo)
        actualBytes = fileInfo.bytes;
    end
    error("Unexpected byte count in %s: expected %d, got %d.", ...
        filePath, expectedBytes, actualBytes);
end

fid = fopen(filePath, "r");
if fid < 0
    error("Failed to open %s.", filePath);
end
cleanupObj = onCleanup(@() fclose(fid));
raw = fread(fid, prod(tensorShape), "single=>single");
tensor = reshape(raw, tensorShape);
end


function data = load_named_array(matPath, csvPath, preferredName)
if exist(matPath, "file")
    loaded = load(matPath);
    if isfield(loaded, preferredName)
        data = loaded.(preferredName);
        return;
    end
    names = fieldnames(loaded);
    if numel(names) ~= 1
        error("MAT file %s has no unique variable to load.", matPath);
    end
    data = loaded.(names{1});
    return;
end
if exist(csvPath, "file")
    data = readmatrix(csvPath);
    return;
end
error("Cannot find %s or %s.", matPath, csvPath);
end


function ensure_dir(pathValue)
if ~exist(pathValue, "dir")
    mkdir(pathValue);
end
end
