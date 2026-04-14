inputFileA = "Sensi_d";
inputFileB = "Sensi_d_old";

outputDir = "Cartesian";
savePreviewFigure = true;

pixelNumX = 100;
pixelNumY = 100;
pixelLX = 3;
pixelLY = 3;

%%
scriptDir = fileparts(mfilename("fullpath"));
if strlength(scriptDir) == 0
    scriptDir = pwd;
end

pathA = fullfile(scriptDir, inputFileA);
pathB = fullfile(scriptDir, inputFileB);
if ~exist(pathA, "file")
    error("Cannot find %s.", pathA);
end
if ~exist(pathB, "file")
    error("Cannot find %s.", pathB);
end

pathOutput = fullfile(scriptDir, outputDir);
if ~exist(pathOutput, "dir")
    mkdir(pathOutput);
end

coorPolar = load_named_array( ...
    fullfile(scriptDir, "coor_polar.mat"), ...
    fullfile(scriptDir, "coor_polar.csv"), ...
    "coor_polar");
rotMat = load_named_array( ...
    fullfile(scriptDir, "RotMat_full.mat"), ...
    fullfile(scriptDir, "RotMat_full.csv"), ...
    "RotMat");

pixelNum = size(rotMat, 1);
pixelNumPolar = size(coorPolar, 1);
pixelNumZ = pixelNum / pixelNumPolar;
if abs(pixelNumZ - round(pixelNumZ)) > 1e-8
    error("pixel_num mismatch: RotMat rows = %d, coor_polar rows = %d.", pixelNum, pixelNumPolar);
end
pixelNumZ = round(pixelNumZ);

[coorCartesianX, coorCartesianY] = meshgrid( ...
    (-pixelNumX * pixelLX / 2 + pixelLX / 2) : pixelLX : (pixelNumX * pixelLX / 2 - pixelLX / 2), ...
    (-pixelNumY * pixelLY / 2 + pixelLY / 2) : pixelLY : (pixelNumY * pixelLY / 2 - pixelLY / 2));

sensiA = read_float32_tensor(pathA, [pixelNumPolar, pixelNumZ]);
sensiB = read_float32_tensor(pathB, [pixelNumPolar, pixelNumZ]);

sensiACartesian = polar_to_cartesian_stack(sensiA, coorPolar, coorCartesianX, coorCartesianY);
sensiBCartesian = polar_to_cartesian_stack(sensiB, coorPolar, coorCartesianX, coorCartesianY);

ratioCartesian = zeros(size(sensiACartesian), "single");
validMask = abs(sensiBCartesian) > 1e-12;
ratioCartesian(validMask) = sensiACartesian(validMask) ./ sensiBCartesian(validMask);

save_tensor_bundle(pathOutput, inputFileA, sensiACartesian);
save_tensor_bundle(pathOutput, inputFileB, sensiBCartesian);
save_tensor_bundle(pathOutput, sprintf("Ratio_%s_over_%s", inputFileA, inputFileB), ratioCartesian);

if savePreviewFigure
    save_preview_figure( ...
        pathOutput, ...
        sensiACartesian, ...
        sensiBCartesian, ...
        ratioCartesian, ...
        inputFileA, ...
        inputFileB, ...
        pixelLX, ...
        pixelLY);
end

fprintf("Saved Cartesian outputs to %s\n", pathOutput);


function data = load_named_array(matPath, csvPath, fieldName)
if exist(matPath, "file")
    loaded = load(matPath);
    if isfield(loaded, fieldName)
        data = loaded.(fieldName);
        return;
    end
    names = fieldnames(loaded);
    if isempty(names)
        error("No variables found in %s.", matPath);
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


function tensor = read_float32_tensor(filePath, tensorShape)
fid = fopen(filePath, "r");
if fid < 0
    error("Failed to open %s.", filePath);
end
cleanupObj = onCleanup(@() fclose(fid));
raw = fread(fid, "float32");
expectedNumel = prod(tensorShape);
if numel(raw) ~= expectedNumel
    error("Unexpected element count in %s: expected %d, got %d.", filePath, expectedNumel, numel(raw));
end
tensor = reshape(raw, tensorShape);
end


function write_float32_tensor(filePath, tensor)
fid = fopen(filePath, "w");
if fid < 0
    error("Failed to open %s for writing.", filePath);
end
cleanupObj = onCleanup(@() fclose(fid));
fwrite(fid, single(tensor), "float32");
end


function imgCartesian = polar_to_cartesian_stack(imgPolar, coorPolar, coorCartesianX, coorCartesianY)
polarSize = size(imgPolar);
if numel(polarSize) == 2
    polarSize(3) = 1;
end

pixelNumX = size(coorCartesianX, 1);
pixelNumY = size(coorCartesianX, 2);
imgCartesian = zeros(pixelNumX, pixelNumY, polarSize(2), polarSize(3), "single");

for iterIdx = 1 : polarSize(3)
    for zIdx = 1 : polarSize(2)
        imgPolarTmp = imgPolar(:, zIdx, iterIdx);
        imgCartesianTmp = griddata( ...
            coorPolar(:, 1), ...
            coorPolar(:, 2), ...
            imgPolarTmp, ...
            coorCartesianX, ...
            coorCartesianY, ...
            "linear").';
        imgCartesian(:, :, zIdx, iterIdx) = single(imgCartesianTmp);
    end
end

imgCartesian(isnan(imgCartesian)) = 0;

if polarSize(3) == 1
    imgCartesian = squeeze(imgCartesian);
end
end


function save_tensor_bundle(outputDir, baseName, tensor)
safeName = regexprep(baseName, "[^\w\-\.]", "_");
matPath = fullfile(outputDir, safeName + ".mat");
binPath = fullfile(outputDir, safeName + ".bin");
tensorShape = size(tensor);

save(matPath, "tensor", "tensorShape", "-v7.3");
write_float32_tensor(binPath, tensor);
end


function save_preview_figure(outputDir, sensiA, sensiB, ratioTensor, labelA, labelB, pixelLX, pixelLY)
pixelNumX = size(sensiA, 1);
pixelNumY = size(sensiA, 2);
pixelNumZ = size(sensiA, 3);
zIdx = round((pixelNumZ + 1) / 2);

imgA = sensiA(:, :, zIdx);
imgB = sensiB(:, :, zIdx);
imgRatio = ratioTensor(:, :, zIdx);

xRange = [-pixelNumX * pixelLX / 2, pixelNumX * pixelLX / 2];
yRange = [-pixelNumY * pixelLY / 2, pixelNumY * pixelLY / 2];

f = figure("Position", [100, 100, 1200, 360]);
tl = tiledlayout(1, 3, "TileSpacing", "compact", "Padding", "compact");

ax1 = nexttile(tl);
imagesc(yRange, xRange, imgA);
axis image;
title(labelA, "Interpreter", "none");
xlabel("y (mm)");
ylabel("x (mm)");
colorbar;

ax2 = nexttile(tl);
imagesc(yRange, xRange, imgB);
axis image;
title(labelB, "Interpreter", "none");
xlabel("y (mm)");
ylabel("x (mm)");
colorbar;

ax3 = nexttile(tl);
imagesc(yRange, xRange, imgRatio);
axis image;
title(sprintf("%s / %s", labelA, labelB), "Interpreter", "none");
xlabel("y (mm)");
ylabel("x (mm)");
colorbar;

colormap(ax1, turbo);
colormap(ax2, turbo);
colormap(ax3, turbo);

saveas(f, fullfile(outputDir, "show_sensi_d_preview.png"));
saveas(f, fullfile(outputDir, "show_sensi_d_preview.fig"));
end
