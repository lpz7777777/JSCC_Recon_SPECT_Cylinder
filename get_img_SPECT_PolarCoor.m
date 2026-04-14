folderPath = uigetdir("./Figure/");
if isequal(folderPath, 0)
    return;
end


generateCartesian = 0;
sigmaGauss = 0.01;
mipStartLayer = 1;
mipEndLayer = 20;

pixelNumX = 100;
pixelNumY = 100;
pixelLX = 3;
pixelLY = 3;
pixelLZ = 3;

[~, name] = fileparts(folderPath);
pathPolar = fullfile(folderPath, "Polar");
pathCartesian = fullfile(folderPath, "Cartesian");
if ~exist(pathCartesian, "dir")
    mkdir(pathCartesian);
end

rotateNum = parse_single_token(name, "(?:RotNum|RotateNum)(\d+)", "rotate number");
energyKeV = parse_single_token(name, "_(\d+)keV", "energy");
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
[mipStartLayer, mipEndLayer] = normalize_mip_layer_range(mipStartLayer, mipEndLayer, pixelNumCartesianZ);

iterInfoSc = parse_iter_file(pathPolar, "Image_SC_Iter_*");
iterInfoScd = parse_iter_file_optional(pathPolar, "Image_SCD_Iter_*");
iterInfoJsccd = parse_iter_file_optional(pathPolar, "Image_JSCCD_Iter_*");
iterInfoJsccsd = parse_iter_file(pathPolar, "Image_JSCCSD_Iter_*");
if iterInfoSc.iterMax ~= iterInfoJsccsd.iterMax || iterInfoSc.saveCount ~= iterInfoJsccsd.saveCount
    error("SC and JSCCSD iteration files are inconsistent.");
end

iterMax = iterInfoSc.iterMax;
saveCount = iterInfoSc.saveCount;
iterInterval = iterInfoSc.iterInterval;
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

cartesianFileSc = fullfile(pathCartesian, iterInfoSc.fileName);
cartesianFileScd = "";
cartesianFileJsccd = "";
cartesianFileJsccsd = fullfile(pathCartesian, iterInfoJsccsd.fileName);
if ~isempty(iterInfoScd)
    cartesianFileScd = fullfile(pathCartesian, iterInfoScd.fileName);
end
if ~isempty(iterInfoJsccd)
    cartesianFileJsccd = fullfile(pathCartesian, iterInfoJsccd.fileName);
end

cartesianJobs = struct( ...
    "iterInfo", {iterInfoSc, iterInfoJsccsd}, ...
    "cartesianFile", {cartesianFileSc, cartesianFileJsccsd});
if ~isempty(iterInfoScd)
    cartesianJobs(end + 1) = struct("iterInfo", iterInfoScd, "cartesianFile", cartesianFileScd);
end
if ~isempty(iterInfoJsccd)
    cartesianJobs(end + 1) = struct("iterInfo", iterInfoJsccd, "cartesianFile", cartesianFileJsccd);
end

needGenerateCartesian = generateCartesian == 1;
for jobIdx = 1 : numel(cartesianJobs)
    if ~exist(cartesianJobs(jobIdx).cartesianFile, "file")
        needGenerateCartesian = true;
        break;
    end
end

if needGenerateCartesian
    for jobIdx = 1 : numel(cartesianJobs)
        imgIterPolar = read_float32_tensor( ...
            fullfile(pathPolar, cartesianJobs(jobIdx).iterInfo.fileName), ...
            [pixelNumPolar, pixelNumCartesianZ, cartesianJobs(jobIdx).iterInfo.saveCount]);
        imgIterCartesian = polar_to_cartesian_stack(imgIterPolar, coorPolar, coorCartesianX, coorCartesianY);
        write_float32_tensor(cartesianJobs(jobIdx).cartesianFile, imgIterCartesian);
    end
end

%%
% showCenter = [30, 0, 0];
showCenter = [0, 0, -13];
% showCenter = [0, 0, 0];
imgScIterCartesian = read_float32_tensor(cartesianFileSc, [pixelNumX, pixelNumY, pixelNumCartesianZ, saveCount]);
imgJsccsdIterCartesian = read_float32_tensor(cartesianFileJsccsd, [pixelNumX, pixelNumY, pixelNumCartesianZ, saveCount]);

showCenterPixel = round(showCenter ./ [pixelLX, pixelLY, pixelLZ] + [pixelNumX, pixelNumY, pixelNumCartesianZ] / 2);
showCenterPixel = max(showCenterPixel, [1, 1, 1]);
showCenterPixel = min(showCenterPixel, [pixelNumX, pixelNumY, pixelNumCartesianZ]);

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
f.Position = [100, 100, 850, 250 * length(iterShow)];
tOuter = tiledlayout(length(iterShow), 17);
tOuter.TileSpacing = "none";
tOuter.Padding = "compact";

for idx = 1 : length(iterShow)
    iterValue = iterShow(idx);
    iterId = round(iterValue / iterInterval);
    iterId = max(1, min(saveCount, iterId));

    imgSc = imgScIterCartesian(:, :, :, iterId);
    imgJsccsd = imgJsccsdIterCartesian(:, :, :, iterId);

    rowDataSc = extract_views(imgSc, showCenterPixel, sigmaGauss);
    rowDataJsccsd = extract_views(imgJsccsd, showCenterPixel, sigmaGauss);

    rowMaxSc = get_display_max(rowDataSc.transverse, rowDataSc.coronal, rowDataSc.sagittal, rangeU, rangeV);
    rowMaxJsccsd = get_display_max(rowDataJsccsd.transverse, rowDataJsccsd.coronal, rowDataJsccsd.sagittal, rangeU, rangeV);

    plot_orthogonal_row( ...
        tOuter, rowDataSc, [minX, maxX], [minY, maxY], [minZ, maxZ], ...
        rowMaxSc, showCenter, "SC", sprintf("Iter=%d", iterValue), colorMap, idx == 1);

    axSpacer = nexttile(tOuter, [1, 1]);
    axSpacer.Visible = "off";

    plot_orthogonal_row( ...
        tOuter, rowDataJsccsd, [minX, maxX], [minY, maxY], [minZ, maxZ], ...
        rowMaxJsccsd, showCenter, "JSCCSD", "", colorMap, idx == 1);
end

title(tOuter, sprintf("Data Name: %s", name), "Interpreter", "none");
saveas(f, fullfile(folderPath, "show.png"));
savefig(f, fullfile(folderPath, "show.fig"));

fMip = figure;
fMip.Position = [140, 80, 500, 260 * length(iterShow)];
tMip = tiledlayout(length(iterShow), 2);
tMip.TileSpacing = "none";
tMip.Padding = "compact";

for idx = 1 : length(iterShow)
    iterValue = iterShow(idx);
    iterId = round(iterValue / iterInterval);
    iterId = max(1, min(saveCount, iterId));

    imgScMip = get_transverse_mip(imgScIterCartesian(:, :, :, iterId), mipStartLayer, mipEndLayer);
    imgJsccsdMip = get_transverse_mip(imgJsccsdIterCartesian(:, :, :, iterId), mipStartLayer, mipEndLayer);
    rowMaxMip = max([imgScMip(:); imgJsccsdMip(:)]);
    if rowMaxMip <= 0
        rowMaxMip = 1;
    end

    axScMip = nexttile(tMip);
    imagesc([minY, maxY], [minX, maxX], imgScMip, [0, rowMaxMip]);
    axis equal;
    colormap(axScMip, colorMap);
    xlabel("y (mm)");
    ylabel("x (mm)");
    xlim([minY, maxY]);
    ylim([minX, maxX]);
    if idx == 1
        title("SC Transverse MIP", "Interpreter", "none");
    end

    axJsccsdMip = nexttile(tMip);
    imagesc([minY, maxY], [minX, maxX], imgJsccsdMip, [0, rowMaxMip]);
    axis equal;
    colormap(axJsccsdMip, colorMap);
    xlabel("y (mm)");
    ylabel("x (mm)");
    xlim([minY, maxY]);
    ylim([minX, maxX]);
    if idx == 1
        title("JSCCSD Transverse MIP", "Interpreter", "none");
    end

    cbMip = colorbar(axJsccsdMip, "eastoutside");
    cbMip.Label.String = sprintf("Iter=%d", iterValue);
    cbMip.Label.Interpreter = "none";
    clim(axScMip, [0, rowMaxMip]);
    clim(axJsccsdMip, [0, rowMaxMip]);
end

title(tMip, sprintf("Transverse MIP z=%d:%d: %s", mipStartLayer, mipEndLayer, name), "Interpreter", "none");
saveas(fMip, fullfile(folderPath, "mip.png"));
savefig(fMip, fullfile(folderPath, "mip.fig"));


function iterInfo = parse_iter_file(folderPath, pattern)
matches = dir(fullfile(folderPath, pattern));
if isempty(matches)
    error("Cannot find %s under %s.", pattern, folderPath);
end
if numel(matches) > 1
    warning("Found multiple files for %s. Using %s.", pattern, matches(1).name);
end

tokens = regexp(matches(1).name, ".*_Iter_(\d+)_(\d+)$", "tokens", "once");
if isempty(tokens)
    error("Failed to parse iteration info from %s.", matches(1).name);
end

iterInfo.fileName = matches(1).name;
iterInfo.iterMax = str2double(tokens{1});
iterInfo.saveCount = str2double(tokens{2});
iterInfo.iterInterval = round(iterInfo.iterMax / iterInfo.saveCount);
end


function iterInfo = parse_iter_file_optional(folderPath, pattern)
matches = dir(fullfile(folderPath, pattern));
if isempty(matches)
    iterInfo = [];
    return;
end
iterInfo = parse_iter_file(folderPath, pattern);
end


function views = extract_views(imgVolume, showCenterPixel, sigmaGauss)
views.transverse = flip(imgaussfilt(imgVolume(:, :, showCenterPixel(3)).', sigmaGauss), 1);
views.coronal = imgaussfilt(squeeze(imgVolume(:, showCenterPixel(2), :)), sigmaGauss);
views.sagittal = imgaussfilt(squeeze(imgVolume(showCenterPixel(1), :, :)), sigmaGauss);
end


function maxValue = get_display_max(imgTransverse, imgCoronal, imgSagittal, rangeU, rangeV)
maxValue = max(imgTransverse(rangeU, rangeV), [], "all");
maxValue = max([maxValue, max(imgCoronal, [], "all"), max(imgSagittal, [], "all")]);
if maxValue <= 0
    maxValue = 1;
end
end


function plot_orthogonal_row(tOuter, rowData, xRange, yRange, zRange, maxColor, showCenter, labelPrefix, colorbarLabel, colorMap, showTitles)
ax1 = nexttile(tOuter, [1, 4]);
imagesc(yRange, xRange, rowData.transverse, [0, maxColor]);
axis equal;
colormap(colorMap);
hold on;
line([showCenter(1), showCenter(1)], yRange * 0.75, "Color", "red", "LineStyle", "--", "LineWidth", 0.5);
line(xRange * 0.75, [showCenter(2), showCenter(2)], "Color", "blue", "LineStyle", "--", "LineWidth", 0.5);
if showTitles
    title(sprintf("%s Transverse", labelPrefix), "Interpreter", "none");
end
xlabel("y (mm)");
ylabel("x (mm)");
xlim(yRange);
ylim(xRange);

ax2 = nexttile(tOuter, [1, 2]);
imagesc(zRange, xRange, rowData.coronal, [0, maxColor]);
axis equal;
colormap(colorMap);
hold on;
line([showCenter(3), showCenter(3)], xRange * 0.75, "Color", "black", "LineStyle", "--", "LineWidth", 0.5);
if showTitles
    title(sprintf("%s Coronal", labelPrefix), "Interpreter", "none");
end
xlabel("z (mm)");
ylabel("x (mm)");
xlim(zRange);
ylim(xRange);

ax3 = nexttile(tOuter, [1, 2]);
imagesc(zRange, yRange, rowData.sagittal, [0, maxColor]);
axis equal;
colormap(colorMap);
if showTitles
    title(sprintf("%s Sagittal", labelPrefix), "Interpreter", "none");
end
xlabel("z (mm)");
ylabel("y (mm)");
xlim(zRange);
ylim(yRange);

cb = colorbar(ax1, "westoutside");
if ~isempty(colorbarLabel)
    cb.Label.String = colorbarLabel;
    cb.Label.Interpreter = "none";
end
clim(ax1, [0, maxColor]);
clim(ax2, [0, maxColor]);
clim(ax3, [0, maxColor]);
end


function value = parse_single_token(textValue, expr, label)
tokens = regexp(textValue, expr, "tokens", "once");
if isempty(tokens)
    error("Failed to parse %s from %s.", label, textValue);
end
value = str2double(tokens{1});
end


function iterShow = build_iter_show_list(iterMax, iterInterval)
divisors = [100, 50, 20, 10, 2];
iterShow = zeros(1, numel(divisors) + 1);
for idx = 1 : numel(divisors)
    iterValue = floor(iterMax / divisors(idx));
    iterValue = max(iterInterval, floor(iterValue / iterInterval) * iterInterval);
    iterShow(idx) = iterValue;
end
iterShow(end) = iterMax;
iterShow = unique(iterShow(iterShow >= iterInterval & iterShow <= iterMax), "stable");
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
