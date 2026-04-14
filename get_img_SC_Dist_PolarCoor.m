folderPath = uigetdir("./Figure_Dist_SC/");
if isequal(folderPath, 0)
    return;
end

[~, name] = fileparts(folderPath);
pathPolar = fullfile(folderPath, "Polar");
pathCartesian = fullfile(folderPath, "Cartesian");
if ~exist(pathCartesian, "dir")
    mkdir(pathCartesian);
end

rotateNum = parse_single_token(name, "(?:RotNum|RotateNum)(\d+)", "rotate number");
energyKeV = parse_single_token(name, "_(\d+)keV_", "energy");
factorPath = fullfile(".", "Factors", sprintf("%dkeV_RotateNum%d", energyKeV, rotateNum));

rotMat = load_named_array(fullfile(factorPath, "RotMat_full.mat"), fullfile(factorPath, "RotMat_full.csv"), "RotMat");
rotMatInv = load_named_array(fullfile(factorPath, "RotMatInv_full.mat"), fullfile(factorPath, "RotMatInv_full.csv"), "RotMatInv");
coorPolar = load_named_array(fullfile(factorPath, "coor_polar.mat"), fullfile(factorPath, "coor_polar.csv"), "coor_polar");

pixelNum = size(rotMat, 1);
pixelNumPolar = size(coorPolar, 1);
pixelNumCartesianZ = pixelNum / pixelNumPolar;
if abs(pixelNumCartesianZ - round(pixelNumCartesianZ)) > 1e-8
    error("pixel_num mismatch: RotMat rows = %d, coor_polar rows = %d.", pixelNum, pixelNumPolar);
end
pixelNumCartesianZ = round(pixelNumCartesianZ);

iterFile = dir(fullfile(pathPolar, "Image_SC_Iter_*"));
if isempty(iterFile)
    error("Cannot find Image_SC_Iter_* under %s.", pathPolar);
end
if numel(iterFile) > 1
    warning("Found multiple Image_SC_Iter_* files. Using %s.", iterFile(1).name);
end
iterFileName = iterFile(1).name;

tokens = regexp(iterFileName, "Image_SC_Iter_(\d+)_(\d+)$", "tokens", "once");
if isempty(tokens)
    error("Failed to parse iteration info from %s.", iterFileName);
end
iterMax = str2double(tokens{1});
saveCount = str2double(tokens{2});
iterInterval = round(iterMax / saveCount);

%%
sigmaGauss = 0.01;
pixelNumX = 100;
pixelNumY = 100;
pixelLX = 3;
pixelLY = 3;
pixelLZ = 3;
generateCartesian = 1;
mipStartLayer = [];
mipEndLayer = [];
iterShowPreferred = build_iter_show_list(iterMax, iterInterval);

iterShow = unique(iterShowPreferred(iterShowPreferred >= iterInterval & iterShowPreferred <= iterMax));
iterShow = iterShow(mod(iterShow, iterInterval) == 0);
if isempty(iterShow)
    sampleCount = min(6, saveCount);
    iterShow = unique(round(linspace(1, saveCount, sampleCount)) * iterInterval);
end

[coorCartesianX, coorCartesianY] = meshgrid( ...
    (-pixelNumX * pixelLX / 2 + pixelLX / 2) : pixelLX : (pixelNumX * pixelLX / 2 - pixelLX / 2), ...
    (-pixelNumY * pixelLY / 2 + pixelLY / 2) : pixelLY : (pixelNumY * pixelLY / 2 - pixelLY / 2));

cartesianFile = fullfile(pathCartesian, iterFileName);
if generateCartesian == 1 || ~exist(cartesianFile, "file")
    imgScIterPolar = read_float32_tensor(fullfile(pathPolar, iterFileName), [pixelNumPolar, pixelNumCartesianZ, saveCount]);
    imgScIterCartesian = polar_to_cartesian_stack(imgScIterPolar, coorPolar, coorCartesianX, coorCartesianY);
    write_float32_tensor(cartesianFile, imgScIterCartesian);

    finalPolarFile = fullfile(pathPolar, "Image_SC");
    if exist(finalPolarFile, "file")
        imgScPolar = read_float32_tensor(finalPolarFile, [pixelNumPolar, pixelNumCartesianZ]);
        imgScCartesian = polar_to_cartesian_stack(imgScPolar, coorPolar, coorCartesianX, coorCartesianY);
        write_float32_tensor(fullfile(pathCartesian, "Image_SC"), imgScCartesian);
    end
end

imgScIterCartesian = read_float32_tensor(cartesianFile, [pixelNumX, pixelNumY, pixelNumCartesianZ, saveCount]);
[mipStartLayer, mipEndLayer] = normalize_mip_layer_range(mipStartLayer, mipEndLayer, pixelNumCartesianZ);

%%
showCenter = [0, 0, -13];
% showCenter = [0, 0, 0];
showCenterPixel = round(showCenter ./ [pixelLX, pixelLY, pixelLZ] + [pixelNumX, pixelNumY, pixelNumCartesianZ] / 2);
showCenterPixel = max(showCenterPixel, [1, 1, 1]);
showCenterPixel = min(showCenterPixel, [pixelNumX, pixelNumY, pixelNumCartesianZ]);
showCenterPixelData = showCenterPixel;
showCenterPixelData(2) = pixelNumY - showCenterPixel(2) + 1;

cutRange = 5;
rangeU = (1 + cutRange) : (pixelNumX - cutRange);
rangeV = (1 + cutRange) : (pixelNumY - cutRange);

minX = -pixelNumX * pixelLX / 2;
maxX = -minX;
minY = minX;
maxY = maxX;
minZ = -pixelNumCartesianZ * pixelLZ / 2;
maxZ = -minZ;

colorMap = flipud(gray(1024));

f = figure;
f.Position = [100, 100, 700, 250 * length(iterShow)];
t = tiledlayout(length(iterShow), 8);
t.TileSpacing = "none";
t.Padding = "compact";

for idx = 1 : length(iterShow)
    iterValue = iterShow(idx);
    iterId = round(iterValue / iterInterval);
    iterId = max(1, min(saveCount, iterId));

    imgSc = imgScIterCartesian(:, :, :, iterId);

    imgTransverse = flip(imgaussfilt(imgSc(:, :, showCenterPixelData(3)).', sigmaGauss), 1);
    imgCoronal = flip(imgaussfilt(squeeze(imgSc(:, showCenterPixelData(2), :)), sigmaGauss), 1);
    imgSagittal = flip(imgaussfilt(squeeze(imgSc(showCenterPixelData(1), :, :)), sigmaGauss), 1);

    rowMaxColor = max(imgTransverse(rangeU, rangeV), [], "all");
    rowMaxColor = max([rowMaxColor, max(imgCoronal, [], "all"), max(imgSagittal, [], "all")]);
    if rowMaxColor <= 0
        rowMaxColor = 1;
    end

    ax1 = nexttile(t, [1, 4]);
    imagesc([minX, maxX], [minY, maxY], imgTransverse, [0, rowMaxColor]);
    axis equal;
    colormap(colorMap);
    hold on;
    line([showCenter(1), showCenter(1)], [minY, maxY] * 0.75, "Color", "red", "LineStyle", "--", "LineWidth", 0.5);
    line([minX, maxX] * 0.75, [showCenter(2), showCenter(2)], "Color", "blue", "LineStyle", "--", "LineWidth", 0.5);
    if idx == 1
        title("SC Transverse", "Interpreter", "none");
    end
    xlabel("y (mm)");
    ylabel("x (mm)");
    xlim([minY, maxY]);
    ylim([minX, maxX]);

    ax2 = nexttile(t, [1, 2]);
    imagesc([minZ, maxZ], [minX, maxX], imgCoronal, [0, rowMaxColor]);
    axis equal;
    colormap(colorMap);
    hold on;
    line([showCenter(3), showCenter(3)], [minX, maxX] * 0.75, "Color", "black", "LineStyle", "--", "LineWidth", 0.5);
    if idx == 1
        title("SC Coronal", "Interpreter", "none");
    end
    xlabel("z (mm)");
    ylabel("x (mm)");
    xlim([minZ, maxZ]);
    ylim([minX, maxX]);

    ax3 = nexttile(t, [1, 2]);
    imagesc([minZ, maxZ], [minY, maxY], imgSagittal, [0, rowMaxColor]);
    axis equal;
    colormap(colorMap);
    if idx == 1
        title("SC Sagittal", "Interpreter", "none");
    end
    xlabel("z (mm)");
    ylabel("y (mm)");
    xlim([minZ, maxZ]);
    ylim([minY, maxY]);

    cb = colorbar(ax1, "westoutside");
    cb.Label.String = sprintf("Iter=%d", iterValue);
    cb.Label.Interpreter = "none";
    clim(ax1, [0, rowMaxColor]);
    clim(ax2, [0, rowMaxColor]);
    clim(ax3, [0, rowMaxColor]);
end

title(t, sprintf("SC Reconstruction: %s", name), "Interpreter", "none");
saveas(f, fullfile(folderPath, "img_show_sc.png"));
saveas(f, fullfile(folderPath, "img_show_sc.fig"));

fMip = figure;
fMip.Position = [140, 80, 320, 260 * length(iterShow)];
tMip = tiledlayout(length(iterShow), 1);
tMip.TileSpacing = "none";
tMip.Padding = "compact";

for idx = 1 : length(iterShow)
    iterValue = iterShow(idx);
    iterId = round(iterValue / iterInterval);
    iterId = max(1, min(saveCount, iterId));

    imgScMip = get_transverse_mip(imgScIterCartesian(:, :, :, iterId), mipStartLayer, mipEndLayer);
    rowMaxMip = max(imgScMip, [], "all");
    if rowMaxMip <= 0
        rowMaxMip = 1;
    end

    axMip = nexttile(tMip);
    imagesc([minY, maxY], [minX, maxX], imgScMip, [0, rowMaxMip]);
    axis equal;
    colormap(axMip, colorMap);
    xlabel("y (mm)");
    ylabel("x (mm)");
    xlim([minY, maxY]);
    ylim([minX, maxX]);
    if idx == 1
        title("SC Transverse MIP", "Interpreter", "none");
    end

    cbMip = colorbar(axMip, "eastoutside");
    cbMip.Label.String = sprintf("Iter=%d", iterValue);
    cbMip.Label.Interpreter = "none";
    clim(axMip, [0, rowMaxMip]);
end

title(tMip, sprintf("SC Transverse MIP z=%d:%d: %s", mipStartLayer, mipEndLayer, name), "Interpreter", "none");
saveas(fMip, fullfile(folderPath, "mip_sc.png"));
saveas(fMip, fullfile(folderPath, "mip_sc.fig"));


function value = parse_single_token(textValue, expr, label)
tokens = regexp(textValue, expr, "tokens", "once");
if isempty(tokens)
    error("Failed to parse %s from %s.", label, textValue);
end
value = str2double(tokens{1});
end


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


function iterShow = build_iter_show_list(iterMax, iterInterval)
divisors = [50, 20, 10, 2];
iterShow = zeros(1, numel(divisors) + 1);
for idx = 1 : numel(divisors)
    iterValue = floor(iterMax / divisors(idx));
    iterValue = max(iterInterval, floor(iterValue / iterInterval) * iterInterval);
    iterShow(idx) = iterValue;
end
iterShow(end) = iterMax;
iterShow = unique(iterShow(iterShow >= iterInterval & iterShow <= iterMax));
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
fwrite(fid, tensor, "float32");
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


function imgMip = get_transverse_mip(imgVolume, mipStartLayer, mipEndLayer)
imgMip = squeeze(max(imgVolume(:, :, mipStartLayer:mipEndLayer), [], 3)).';
imgMip = flip(imgMip, 1);
end


function [mipStartLayer, mipEndLayer] = normalize_mip_layer_range(mipStartLayer, mipEndLayer, pixelNumCartesianZ)
if isempty(mipStartLayer)
    mipStartLayer = 1;
end
if isempty(mipEndLayer)
    mipEndLayer = pixelNumCartesianZ;
end

mipStartLayer = max(1, min(pixelNumCartesianZ, round(mipStartLayer)));
mipEndLayer = max(1, min(pixelNumCartesianZ, round(mipEndLayer)));
if mipStartLayer > mipEndLayer
    error("Invalid MIP layer range: start=%d, end=%d.", mipStartLayer, mipEndLayer);
end
end
