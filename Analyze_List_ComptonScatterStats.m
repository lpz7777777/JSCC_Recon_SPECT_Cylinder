clear;
clc;

repoRoot = fileparts(mfilename("fullpath"));
defaultListRoot = fullfile(repoRoot, "List");

listDir = uigetdir(defaultListRoot, "Select one List directory containing CSV files");
if isequal(listDir, 0)
    return;
end

prompt = {
    "Energy-sum threshold for two interactions (MeV), keep events with e1+e2 >= threshold:";
    "Maximum sampled events per category for scatter plots:";
    "Use only current directory CSV files? Enter 1 for yes, 0 for no:";
    };
dlgTitle = "Compton Scatter Statistics";
dlgDims = [1, 72];
dlgDefault = {"0.46", "30000", "1"};
answer = inputdlg(prompt, dlgTitle, dlgDims, dlgDefault);
if isempty(answer)
    return;
end

energySumThreshold = str2double(answer{1});
maxSamplePerCategory = round(str2double(answer{2}));
useCurrentDirOnly = logical(round(str2double(answer{3})));

if ~isfinite(energySumThreshold)
    error("Invalid energy-sum threshold.");
end
if ~isfinite(maxSamplePerCategory) || maxSamplePerCategory <= 0
    error("Invalid max sampled event count.");
end

detectorPath = resolve_detector_path(listDir, repoRoot);
if isempty(detectorPath)
    [detectorFile, detectorFolder] = uigetfile( ...
        {"Detector.csv", "Detector.csv"; "*.csv", "CSV files (*.csv)"}, ...
        "Select Detector.csv", ...
        fullfile(repoRoot, "Factors"));
    if isequal(detectorFile, 0)
        error("Detector.csv was not selected.");
    end
    detectorPath = fullfile(detectorFolder, detectorFile);
end

[layerByDet, layerLabels, layerCount] = build_layer_map(detectorPath);
frontLayerMask = false(layerCount, 1);
if layerCount >= 2
    frontLayerMask(1:max(1, layerCount - 1)) = true;
else
    error("Detector layer count is less than 2. Cannot separate front/rear layers.");
end
rearLayerIndex = layerCount;

csvFiles = collect_csv_files(listDir, useCurrentDirOnly);
if isempty(csvFiles)
    error("No CSV files were found in the selected List directory.");
end

categoryNames = ["FrontToRear", "SameLayer", "Reverse", "Other"];
categoryLabels = ["Front->Rear", "Same layer", "Reverse", "Other"];
categoryNum = numel(categoryNames);

overallCounts = zeros(1, categoryNum);
overallFilteredEvents = 0;
transitionCounts = zeros(layerCount, layerCount);
sampleStates = repmat(make_sample_state(maxSamplePerCategory), 1, categoryNum);
statsStates = repmat(make_stats_state(), 1, categoryNum);

perFileRows = struct([]);

fprintf("List directory: %s\n", listDir);
fprintf("Detector path:  %s\n", detectorPath);
fprintf("Energy sum threshold: %.6f MeV\n", energySumThreshold);
fprintf("CSV file count: %d\n\n", numel(csvFiles));

for fileIdx = 1:numel(csvFiles)
    filePath = csvFiles(fileIdx).path;
    fileData = readmatrix(filePath, "Delimiter", ",");

    if isempty(fileData)
        row = build_empty_file_row(csvFiles(fileIdx).name, categoryNames);
        perFileRows = [perFileRows; row]; %#ok<AGROW>
        fprintf("[%3d/%3d] %s : empty file\n", fileIdx, numel(csvFiles), csvFiles(fileIdx).name);
        continue;
    end

    if size(fileData, 2) < 4
        error("File %s has fewer than 4 columns.", filePath);
    end

    cpnum1 = round(fileData(:, 1));
    e1 = fileData(:, 2);
    cpnum2 = round(fileData(:, 3));
    e2 = fileData(:, 4);

    valid = isfinite(cpnum1) & isfinite(cpnum2) & isfinite(e1) & isfinite(e2);
    valid = valid & cpnum1 >= 1 & cpnum1 <= numel(layerByDet);
    valid = valid & cpnum2 >= 1 & cpnum2 <= numel(layerByDet);
    valid = valid & (e1 + e2) >= energySumThreshold;

    cpnum1 = cpnum1(valid);
    cpnum2 = cpnum2(valid);
    e1 = e1(valid);
    e2 = e2(valid);

    filteredEventCount = numel(e1);
    fileCounts = zeros(1, categoryNum);
    fileMeans = nan(1, categoryNum * 4);

    if filteredEventCount > 0
        layer1 = layerByDet(cpnum1);
        layer2 = layerByDet(cpnum2);

        isFront1 = frontLayerMask(layer1);
        isFront2 = frontLayerMask(layer2);
        isRear1 = layer1 == rearLayerIndex;
        isRear2 = layer2 == rearLayerIndex;

        maskFrontToRear = isFront1 & isRear2;
        maskSameLayer = layer1 == layer2;
        maskReverse = isRear1 & isFront2;
        maskOther = ~(maskFrontToRear | maskSameLayer | maskReverse);

        classMasks = {
            maskFrontToRear, ...
            maskSameLayer, ...
            maskReverse, ...
            maskOther ...
            };

        overallFilteredEvents = overallFilteredEvents + filteredEventCount;
        transitionCounts = transitionCounts + accumarray([layer1, layer2], 1, [layerCount, layerCount]);

        for catIdx = 1:categoryNum
            mask = classMasks{catIdx};
            count = nnz(mask);
            fileCounts(catIdx) = count;
            overallCounts(catIdx) = overallCounts(catIdx) + count;

            if count == 0
                continue;
            end

            e1Cat = e1(mask);
            e2Cat = e2(mask);
            sumCat = e1Cat + e2Cat;
            ratioCat = e1Cat ./ max(sumCat, eps);

            statsStates(catIdx) = update_stats_state(statsStates(catIdx), e1Cat, e2Cat);
            sampleStates(catIdx) = update_sample_state(sampleStates(catIdx), e1Cat, e2Cat);

            fileMeans(4 * (catIdx - 1) + 1) = mean(e1Cat);
            fileMeans(4 * (catIdx - 1) + 2) = mean(e2Cat);
            fileMeans(4 * (catIdx - 1) + 3) = mean(sumCat);
            fileMeans(4 * (catIdx - 1) + 4) = mean(ratioCat);
        end
    end

    row = build_file_row(csvFiles(fileIdx).name, filteredEventCount, fileCounts, fileMeans, categoryNames);
    perFileRows = [perFileRows; row]; %#ok<AGROW>
    fprintf("[%3d/%3d] %s : kept %d events\n", fileIdx, numel(csvFiles), csvFiles(fileIdx).name, filteredEventCount);
end

outputPrefix = sprintf("compton_scatter_stats_sumthr_%s", sanitize_number_string(energySumThreshold));
summaryTable = build_summary_table(categoryNames, categoryLabels, overallCounts, overallFilteredEvents, statsStates);
perFileTable = struct2table(perFileRows);

transitionRatio = zeros(size(transitionCounts));
if overallFilteredEvents > 0
    transitionRatio = transitionCounts / overallFilteredEvents;
end

summaryPath = fullfile(listDir, [outputPrefix, "_summary.csv"]);
perFilePath = fullfile(listDir, [outputPrefix, "_per_file.csv"]);
transitionCountPath = fullfile(listDir, [outputPrefix, "_layer_transition_counts.csv"]);
transitionRatioPath = fullfile(listDir, [outputPrefix, "_layer_transition_ratio.csv"]);
reportPath = fullfile(listDir, [outputPrefix, "_report.txt"]);
overviewPngPath = fullfile(listDir, [outputPrefix, "_overview.png"]);
overviewFigPath = fullfile(listDir, [outputPrefix, "_overview.fig"]);
transitionPngPath = fullfile(listDir, [outputPrefix, "_layer_transition.png"]);
transitionFigPath = fullfile(listDir, [outputPrefix, "_layer_transition.fig"]);

writetable(summaryTable, summaryPath);
writetable(perFileTable, perFilePath);
writematrix(transitionCounts, transitionCountPath);
writematrix(transitionRatio, transitionRatioPath);
write_report(reportPath, listDir, detectorPath, energySumThreshold, categoryLabels, overallCounts, overallFilteredEvents, statsStates, layerLabels, transitionCounts);

figOverview = build_overview_figure(categoryLabels, overallCounts, overallFilteredEvents, perFileTable, sampleStates, energySumThreshold);
saveas(figOverview, overviewPngPath);
savefig(figOverview, overviewFigPath);

figTransition = build_transition_figure(transitionCounts, layerLabels, overallFilteredEvents);
saveas(figTransition, transitionPngPath);
savefig(figTransition, transitionFigPath);

fprintf("\nFinished.\n");
fprintf("Summary CSV:      %s\n", summaryPath);
fprintf("Per-file CSV:     %s\n", perFilePath);
fprintf("Report TXT:       %s\n", reportPath);
fprintf("Overview figure:  %s\n", overviewPngPath);
fprintf("Transition figure:%s\n", transitionPngPath);


function detectorPath = resolve_detector_path(listDir, repoRoot)
detectorPath = '';

listDirInfo = dir(listDir);
if isempty(listDirInfo)
    return;
end

parentDir = fileparts(listDir);
[~, parentName] = fileparts(parentDir);
candidate = fullfile(repoRoot, "Factors", parentName, "Detector.csv");
if isfile(candidate)
    detectorPath = candidate;
    return;
end

pathParts = split(string(listDir), filesep);
rotateMask = contains(pathParts, "RotateNum");
if any(rotateMask)
    rotateName = pathParts(find(rotateMask, 1, "last"));
    candidate = fullfile(repoRoot, "Factors", rotateName, "Detector.csv");
    if isfile(candidate)
        detectorPath = candidate;
        return;
    end
end
end


function [layerByDet, layerLabels, layerCount] = build_layer_map(detectorPath)
detectorData = readmatrix(detectorPath, "Delimiter", ",");
if size(detectorData, 2) < 4
    error("Detector.csv must contain at least 4 columns.");
end

detectorPos = detectorData(:, 2:4);
detectorYAbs = abs(detectorPos(:, 2));
tolerance = 1e-4;

layerValues = uniquetol(detectorYAbs, tolerance, "DataScale", 1);
layerValues = sort(layerValues(:), "ascend");
layerCount = numel(layerValues);

layerByDet = zeros(size(detectorYAbs));
for idx = 1:numel(detectorYAbs)
    [distance, nearestLayer] = min(abs(detectorYAbs(idx) - layerValues));
    if distance > max(tolerance, tolerance * max(layerValues))
        error("Detector layer assignment failed at detector index %d.", idx);
    end
    layerByDet(idx) = nearestLayer;
end

layerLabels = strings(layerCount, 1);
for idx = 1:layerCount
    layerLabels(idx) = sprintf("L%d |y|=%.4f", idx, layerValues(idx));
end
end


function csvFiles = collect_csv_files(listDir, useCurrentDirOnly)
if useCurrentDirOnly
    dirInfo = dir(fullfile(listDir, "*.csv"));
else
    dirInfo = dir(fullfile(listDir, "**", "*.csv"));
end

dirInfo = dirInfo(~[dirInfo.isdir]);
if isempty(dirInfo)
    csvFiles = struct("name", {}, "path", {});
    return;
end

csvFiles = repmat(struct("name", "", "path", ""), numel(dirInfo), 1);
sortKeys = zeros(numel(dirInfo), 1);

for idx = 1:numel(dirInfo)
    csvFiles(idx).name = dirInfo(idx).name;
    csvFiles(idx).path = fullfile(dirInfo(idx).folder, dirInfo(idx).name);
    [~, baseName] = fileparts(dirInfo(idx).name);
    numericKey = str2double(baseName);
    if isnan(numericKey)
        numericKey = inf;
    end
    sortKeys(idx) = numericKey;
end

[~, order] = sortrows([isinf(sortKeys), sortKeys, (1:numel(sortKeys)).']);
csvFiles = csvFiles(order);
end


function row = build_empty_file_row(fileName, categoryNames)
zeroCounts = zeros(1, numel(categoryNames));
nanMeans = nan(1, numel(categoryNames) * 4);
row = build_file_row(fileName, 0, zeroCounts, nanMeans, categoryNames);
end


function row = build_file_row(fileName, filteredEventCount, fileCounts, fileMeans, categoryNames)
row = struct();
row.file_name = string(fileName);
row.filtered_event_count = filteredEventCount;

if filteredEventCount > 0
    ratios = fileCounts / filteredEventCount;
else
    ratios = zeros(size(fileCounts));
end

for idx = 1:numel(categoryNames)
    prefix = char(lower(categoryNames(idx)));
    row.([prefix, "_count"]) = fileCounts(idx);
    row.([prefix, "_ratio"]) = ratios(idx);
    row.([prefix, "_mean_e1"]) = fileMeans(4 * (idx - 1) + 1);
    row.([prefix, "_mean_e2"]) = fileMeans(4 * (idx - 1) + 2);
    row.([prefix, "_mean_esum"]) = fileMeans(4 * (idx - 1) + 3);
    row.([prefix, "_mean_e1_fraction"]) = fileMeans(4 * (idx - 1) + 4);
end
end


function state = make_stats_state()
state.count = 0;
state.sumE1 = 0;
state.sumE2 = 0;
state.sumESum = 0;
state.sumRatio = 0;
state.sumDiff = 0;
state.sumE1Sq = 0;
state.sumE2Sq = 0;
state.sumESumSq = 0;
state.sumRatioSq = 0;
state.sumDiffSq = 0;
state.sumE1E2 = 0;
state.countE1Greater = 0;
state.countE2Greater = 0;
state.countEqual = 0;
end


function state = update_stats_state(state, e1, e2)
eSum = e1 + e2;
ratio = e1 ./ max(eSum, eps);
diffValue = e1 - e2;

state.count = state.count + numel(e1);
state.sumE1 = state.sumE1 + sum(e1);
state.sumE2 = state.sumE2 + sum(e2);
state.sumESum = state.sumESum + sum(eSum);
state.sumRatio = state.sumRatio + sum(ratio);
state.sumDiff = state.sumDiff + sum(diffValue);
state.sumE1Sq = state.sumE1Sq + sum(e1 .^ 2);
state.sumE2Sq = state.sumE2Sq + sum(e2 .^ 2);
state.sumESumSq = state.sumESumSq + sum(eSum .^ 2);
state.sumRatioSq = state.sumRatioSq + sum(ratio .^ 2);
state.sumDiffSq = state.sumDiffSq + sum(diffValue .^ 2);
state.sumE1E2 = state.sumE1E2 + sum(e1 .* e2);
state.countE1Greater = state.countE1Greater + nnz(e1 > e2);
state.countE2Greater = state.countE2Greater + nnz(e2 > e1);
state.countEqual = state.countEqual + nnz(e1 == e2);
end


function state = make_sample_state(maxCount)
state.e1 = zeros(0, 1);
state.e2 = zeros(0, 1);
state.maxCount = maxCount;
end


function state = update_sample_state(state, e1, e2)
if isempty(e1)
    return;
end

if numel(e1) > state.maxCount
    chooseIdx = randperm(numel(e1), state.maxCount);
    e1 = e1(chooseIdx);
    e2 = e2(chooseIdx);
end

state.e1 = [state.e1; e1(:)]; %#ok<AGROW>
state.e2 = [state.e2; e2(:)]; %#ok<AGROW>

if numel(state.e1) > state.maxCount
    chooseIdx = randperm(numel(state.e1), state.maxCount);
    state.e1 = state.e1(chooseIdx);
    state.e2 = state.e2(chooseIdx);
end
end


function summaryTable = build_summary_table(categoryNames, categoryLabels, overallCounts, totalCount, statsStates)
rowCount = numel(categoryNames);

category = strings(rowCount, 1);
count = zeros(rowCount, 1);
ratio = zeros(rowCount, 1);
meanE1 = nan(rowCount, 1);
stdE1 = nan(rowCount, 1);
meanE2 = nan(rowCount, 1);
stdE2 = nan(rowCount, 1);
meanESum = nan(rowCount, 1);
stdESum = nan(rowCount, 1);
meanRatio = nan(rowCount, 1);
stdRatio = nan(rowCount, 1);
meanDiff = nan(rowCount, 1);
stdDiff = nan(rowCount, 1);
fracE1Greater = nan(rowCount, 1);
fracE2Greater = nan(rowCount, 1);
corrE1E2 = nan(rowCount, 1);

for idx = 1:rowCount
    category(idx) = categoryLabels(idx);
    count(idx) = overallCounts(idx);
    if totalCount > 0
        ratio(idx) = overallCounts(idx) / totalCount;
    end

    state = statsStates(idx);
    if state.count == 0
        continue;
    end

    meanE1(idx) = state.sumE1 / state.count;
    meanE2(idx) = state.sumE2 / state.count;
    meanESum(idx) = state.sumESum / state.count;
    meanRatio(idx) = state.sumRatio / state.count;
    meanDiff(idx) = state.sumDiff / state.count;

    stdE1(idx) = sqrt(max(state.sumE1Sq / state.count - meanE1(idx) ^ 2, 0));
    stdE2(idx) = sqrt(max(state.sumE2Sq / state.count - meanE2(idx) ^ 2, 0));
    stdESum(idx) = sqrt(max(state.sumESumSq / state.count - meanESum(idx) ^ 2, 0));
    stdRatio(idx) = sqrt(max(state.sumRatioSq / state.count - meanRatio(idx) ^ 2, 0));
    stdDiff(idx) = sqrt(max(state.sumDiffSq / state.count - meanDiff(idx) ^ 2, 0));

    fracE1Greater(idx) = state.countE1Greater / state.count;
    fracE2Greater(idx) = state.countE2Greater / state.count;

    covE1E2 = state.sumE1E2 / state.count - meanE1(idx) * meanE2(idx);
    denom = stdE1(idx) * stdE2(idx);
    if denom > 0
        corrE1E2(idx) = covE1E2 / denom;
    end
end

summaryTable = table( ...
    category, count, ratio, ...
    meanE1, stdE1, meanE2, stdE2, meanESum, stdESum, ...
    meanRatio, stdRatio, meanDiff, stdDiff, ...
    fracE1Greater, fracE2Greater, corrE1E2);
end


function write_report(reportPath, listDir, detectorPath, energySumThreshold, categoryLabels, overallCounts, totalCount, statsStates, layerLabels, transitionCounts)
fileId = fopen(reportPath, "w");
if fileId < 0
    error("Failed to open report file: %s", reportPath);
end

cleanupObj = onCleanup(@() fclose(fileId));

fprintf(fileId, "Compton scatter statistics report\n");
fprintf(fileId, "List directory: %s\n", listDir);
fprintf(fileId, "Detector path: %s\n", detectorPath);
fprintf(fileId, "Energy sum threshold: %.6f MeV\n", energySumThreshold);
fprintf(fileId, "Filtered event count: %d\n\n", totalCount);

for idx = 1:numel(categoryLabels)
    state = statsStates(idx);
    fprintf(fileId, "[%s]\n", categoryLabels(idx));
    fprintf(fileId, "  count = %d\n", overallCounts(idx));
    if totalCount > 0
        fprintf(fileId, "  ratio = %.6f\n", overallCounts(idx) / totalCount);
    else
        fprintf(fileId, "  ratio = NaN\n");
    end

    if state.count == 0
        fprintf(fileId, "  no events\n\n");
        continue;
    end

    meanE1 = state.sumE1 / state.count;
    meanE2 = state.sumE2 / state.count;
    meanESum = state.sumESum / state.count;
    meanRatio = state.sumRatio / state.count;
    meanDiff = state.sumDiff / state.count;

    stdE1 = sqrt(max(state.sumE1Sq / state.count - meanE1 ^ 2, 0));
    stdE2 = sqrt(max(state.sumE2Sq / state.count - meanE2 ^ 2, 0));
    stdESum = sqrt(max(state.sumESumSq / state.count - meanESum ^ 2, 0));
    stdRatio = sqrt(max(state.sumRatioSq / state.count - meanRatio ^ 2, 0));
    stdDiff = sqrt(max(state.sumDiffSq / state.count - meanDiff ^ 2, 0));

    fprintf(fileId, "  mean e1 = %.6f, std e1 = %.6f\n", meanE1, stdE1);
    fprintf(fileId, "  mean e2 = %.6f, std e2 = %.6f\n", meanE2, stdE2);
    fprintf(fileId, "  mean e1+e2 = %.6f, std e1+e2 = %.6f\n", meanESum, stdESum);
    fprintf(fileId, "  mean e1/(e1+e2) = %.6f, std = %.6f\n", meanRatio, stdRatio);
    fprintf(fileId, "  mean (e1-e2) = %.6f, std = %.6f\n", meanDiff, stdDiff);
    fprintf(fileId, "  P(e1 > e2) = %.6f\n", state.countE1Greater / state.count);
    fprintf(fileId, "  P(e2 > e1) = %.6f\n", state.countE2Greater / state.count);

    if meanRatio < 0.45
        trendText = "second interaction tends to receive more deposited energy";
    elseif meanRatio > 0.55
        trendText = "first interaction tends to receive more deposited energy";
    else
        trendText = "two interactions tend to share energy more evenly";
    end
    fprintf(fileId, "  trend = %s\n\n", trendText);
end

fprintf(fileId, "Layer transition counts\n");
for rowIdx = 1:size(transitionCounts, 1)
    for colIdx = 1:size(transitionCounts, 2)
        fprintf(fileId, "  %s -> %s : %d\n", layerLabels(rowIdx), layerLabels(colIdx), transitionCounts(rowIdx, colIdx));
    end
end
end


function fig = build_overview_figure(categoryLabels, overallCounts, totalCount, perFileTable, sampleStates, energySumThreshold)
fig = figure("Position", [100, 80, 1400, 900], "Color", "w");
t = tiledlayout(2, 2, "TileSpacing", "compact", "Padding", "compact");
title(t, sprintf("Compton Scatter Statistics  (e1+e2 >= %.3f MeV)", energySumThreshold), "FontSize", 16);

colorMap = [0.05, 0.35, 0.70; 0.15, 0.60, 0.25; 0.85, 0.33, 0.10; 0.50, 0.50, 0.50];

ax1 = nexttile;
if totalCount > 0
    ratio = overallCounts / totalCount;
else
    ratio = zeros(size(overallCounts));
end
bar(ax1, categorical(cellstr(categoryLabels)), ratio * 100, 0.65, "FaceColor", "flat");
for idx = 1:numel(categoryLabels)
    ax1.Children.CData(idx, :) = colorMap(idx, :);
end
ylabel(ax1, "Percentage (%)");
title(ax1, "Overall category proportion");
grid(ax1, "on");
ax1.GridAlpha = 0.2;
ax1.Box = "off";

ax2 = nexttile;
ratioColumns = {
    "fronttorear_ratio", ...
    "samelayer_ratio", ...
    "reverse_ratio", ...
    "other_ratio" ...
    };
stackData = zeros(height(perFileTable), numel(ratioColumns));
for idx = 1:numel(ratioColumns)
    stackData(:, idx) = perFileTable.(ratioColumns{idx});
end
bar(ax2, stackData, "stacked", "BarWidth", 0.92);
for idx = 1:numel(ratioColumns)
    ax2.Children(numel(ratioColumns) - idx + 1).FaceColor = colorMap(idx, :);
end
xlabel(ax2, "List file index");
ylabel(ax2, "Proportion");
title(ax2, "Per-file category proportion");
legend(ax2, categoryLabels, "Location", "eastoutside");
grid(ax2, "on");
ax2.GridAlpha = 0.2;
ax2.Box = "off";
ylim(ax2, [0, 1]);

ax3 = nexttile;
hold(ax3, "on");
maxEnergy = 0;
for idx = 1:numel(sampleStates)
    if isempty(sampleStates(idx).e1)
        continue;
    end
    maxEnergy = max(maxEnergy, max([sampleStates(idx).e1; sampleStates(idx).e2]));
    scatter(ax3, sampleStates(idx).e1, sampleStates(idx).e2, 10, ...
        "MarkerFaceColor", colorMap(idx, :), ...
        "MarkerEdgeColor", "none", ...
        "MarkerFaceAlpha", 0.18);
end
if maxEnergy <= 0
    maxEnergy = 1;
end
plot(ax3, [0, maxEnergy], [0, maxEnergy], "k--", "LineWidth", 1.2);
xlabel(ax3, "e1 (MeV)");
ylabel(ax3, "e2 (MeV)");
title(ax3, "Sampled e1 vs e2");
legend(ax3, categoryLabels, "Location", "eastoutside");
grid(ax3, "on");
ax3.GridAlpha = 0.2;
ax3.Box = "off";

ax4 = nexttile;
hold(ax4, "on");
edges = linspace(0, 1, 41);
for idx = 1:numel(sampleStates)
    if isempty(sampleStates(idx).e1)
        continue;
    end
    ratioSample = sampleStates(idx).e1 ./ max(sampleStates(idx).e1 + sampleStates(idx).e2, eps);
    histogram(ax4, ratioSample, edges, "Normalization", "probability", ...
        "DisplayStyle", "stairs", "LineWidth", 2, "EdgeColor", colorMap(idx, :));
end
xlabel(ax4, "e1 / (e1 + e2)");
ylabel(ax4, "Probability");
title(ax4, "Energy sharing ratio");
legend(ax4, categoryLabels, "Location", "eastoutside");
grid(ax4, "on");
ax4.GridAlpha = 0.2;
ax4.Box = "off";
end


function fig = build_transition_figure(transitionCounts, layerLabels, totalCount)
fig = figure("Position", [180, 120, 780, 620], "Color", "w");
imagesc(transitionCounts);
axis image;
colormap(parula(256));
colorbar;
set(gca, "XTick", 1:numel(layerLabels), "XTickLabel", layerLabels, ...
    "YTick", 1:numel(layerLabels), "YTickLabel", layerLabels, ...
    "TickLabelInterpreter", "none", "FontSize", 11);
xtickangle(20);
xlabel("Second interaction layer");
ylabel("First interaction layer");
title("Layer-to-layer transition counts");

for rowIdx = 1:size(transitionCounts, 1)
    for colIdx = 1:size(transitionCounts, 2)
        if totalCount > 0
            labelText = sprintf("%d\n%.2f%%", transitionCounts(rowIdx, colIdx), 100 * transitionCounts(rowIdx, colIdx) / totalCount);
        else
            labelText = sprintf("%d", transitionCounts(rowIdx, colIdx));
        end
        text(colIdx, rowIdx, labelText, "HorizontalAlignment", "center", "FontSize", 10, "Color", "w");
    end
end
end


function token = sanitize_number_string(value)
token = strrep(sprintf("%.6f", value), ".", "p");
token = regexprep(token, "p?0+$", "");
if endsWith(token, "p")
    token = token(1:end-1);
end
if isempty(token)
    token = "0";
end
end
