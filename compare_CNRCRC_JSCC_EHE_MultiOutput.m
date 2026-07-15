function outputPaths = compare_CNRCRC_JSCC_EHE_MultiOutput(outputDir)
%COMPARE_CNRCRC_JSCC_EHE_MULTIOUTPUT Compare contrast-phantom CRC and CNR.
%
% Reads the completed JSCC 5000-iteration and EHE 100-iteration dual-energy
% CntStat runs at 1e9, 1e10, and 1e11 emitted photons. Metrics follow
% CNRCRC_SC_Dist.m: 0.85-radius hot ROIs, a background ROI excluding
% 1.15-radius rods, z layers 7:14, CRC=(Ch/Cb-1)/(6-1), and
% CNR=(Ch-Cb)/SDb. Only rods emitted at each energy are reported.

if nargin < 1 || strlength(string(outputDir)) == 0
    outputDir = fullfile(fileparts(mfilename("fullpath")), ...
        "Analysis_CNRCRC_JSCC_vs_EHE");
end
outputDir = char(string(outputDir));
ensure_dir(outputDir);

repoRoot = fileparts(mfilename("fullpath"));
cfg = default_config(repoRoot);
roi = build_roi_masks(cfg);

records = repmat(empty_record(), 0, 1);
series = repmat(empty_series(), numel(cfg.detectors), numel(cfg.countLevels), ...
    numel(cfg.energiesKeV));

for detectorIdx = 1:numel(cfg.detectors)
    detector = cfg.detectors(detectorIdx);
    for countIdx = 1:numel(cfg.countLevels)
        countLevel = cfg.countLevels(countIdx);
        runPath = find_run(detector.root, countLevel, detector.iterMax(countIdx));
        [polarPath, manifest] = read_run(runPath);
        factorPath = resolve_factor_path(repoRoot, manifest);
        coorPolar = load_named_array(fullfile(factorPath, "coor_polar.mat"), ...
            fullfile(factorPath, "coor_polar.csv"), "coor_polar");
        interpPlan = build_interpolation_plan(coorPolar(:, 1:2), ...
            cfg.gridX, cfg.gridY);

        for energyIdx = 1:numel(cfg.energiesKeV)
            energyKeV = cfg.energiesKeV(energyIdx);
            task = find_metric_task(manifest.tasks, energyKeV);
            history = read_history(polarPath, task, size(coorPolar, 1), ...
                cfg.pixelNumZ, double(manifest.pixel_num));
            iterations = double(task.save_iter_step):double(task.save_iter_step):double(task.iterations);
            rodIds = find(cfg.rodEnergyKeV == energyKeV);
            [crc, cnr, hotMean, backgroundMean, backgroundStd] = ...
                calculate_metrics(history, interpPlan, roi, rodIds, cfg);

            s = struct();
            s.detector = detector.name;
            s.countLevel = countLevel;
            s.energyKeV = energyKeV;
            s.runPath = string(runPath);
            s.iterations = iterations;
            s.rodIds = rodIds;
            s.diametersMm = cfg.rodDiametersMm(rodIds);
            s.crc = crc;
            s.cnr = cnr;
            s.hotMean = hotMean;
            s.backgroundMean = backgroundMean;
            s.backgroundStd = backgroundStd;
            series(detectorIdx, countIdx, energyIdx) = s;

            for rodLocalIdx = 1:numel(rodIds)
                for iterIdx = 1:numel(iterations)
                    record = empty_record();
                    record.Detector = detector.name;
                    record.CountLevel = countLevel;
                    record.EnergyKeV = energyKeV;
                    record.RodId = rodIds(rodLocalIdx);
                    record.DiameterMm = cfg.rodDiametersMm(rodIds(rodLocalIdx));
                    record.Iteration = iterations(iterIdx);
                    record.CRC = crc(rodLocalIdx, iterIdx);
                    record.CNR = cnr(rodLocalIdx, iterIdx);
                    record.HotMean = hotMean(rodLocalIdx, iterIdx);
                    record.BackgroundMean = backgroundMean(iterIdx);
                    record.BackgroundStd = backgroundStd(iterIdx);
                    records(end + 1, 1) = record; %#ok<AGROW>
                end
            end
            fprintf("Loaded %s %s %d keV: %d saved iterations from %s\n", ...
                detector.name, countLevel, energyKeV, numel(iterations), runPath);
        end
    end
end

curveTable = struct2table(records);
curveCsv = fullfile(outputDir, "cnrcrc_curves.csv");
writetable(curveTable, curveCsv);
summaryTable = build_summary_table(series);
summaryCsv = fullfile(outputDir, "cnrcrc_summary.csv");
writetable(summaryTable, summaryCsv);
groupSummaryTable = build_group_summary_table(series);
groupSummaryCsv = fullfile(outputDir, "cnrcrc_group_summary.csv");
writetable(groupSummaryTable, groupSummaryCsv);

crcOverview = render_overview(series, cfg, "crc");
crcPng = fullfile(outputDir, "crc_all_counts.png");
crcFig = fullfile(outputDir, "crc_all_counts.fig");
exportgraphics(crcOverview, crcPng, "Resolution", cfg.exportDpi);
savefig(crcOverview, crcFig);
close(crcOverview);

cnrOverview = render_overview(series, cfg, "cnr");
cnrPng = fullfile(outputDir, "cnr_all_counts.png");
cnrFig = fullfile(outputDir, "cnr_all_counts.fig");
exportgraphics(cnrOverview, cnrPng, "Resolution", cfg.exportDpi);
savefig(cnrOverview, cnrFig);
close(cnrOverview);

countPng = strings(numel(cfg.countLevels), 1);
countFig = strings(numel(cfg.countLevels), 1);
for countIdx = 1:numel(cfg.countLevels)
    f = render_count_comparison(series, cfg, countIdx);
    stem = "cnrcrc_" + cfg.countLevels(countIdx);
    countPng(countIdx) = string(fullfile(outputDir, stem + ".png"));
    countFig(countIdx) = string(fullfile(outputDir, stem + ".fig"));
    exportgraphics(f, countPng(countIdx), "Resolution", cfg.exportDpi);
    savefig(f, countFig(countIdx));
    close(f);
end

matPath = fullfile(outputDir, "cnrcrc_comparison.mat");
save(matPath, "series", "curveTable", "summaryTable", "groupSummaryTable", ...
    "cfg", "-v7.3");
outputPaths = struct("output_dir", string(outputDir), ...
    "crc_overview_png", string(crcPng), "cnr_overview_png", string(cnrPng), ...
    "count_png", countPng, "curve_csv", string(curveCsv), ...
    "summary_csv", string(summaryCsv), ...
    "group_summary_csv", string(groupSummaryCsv), "mat", string(matPath));
fprintf("Saved JSCC/EHE CRC-CNR comparison to %s\n", outputDir);
end

function cfg = default_config(repoRoot)
cfg = struct();
cfg.pixelNumX = 100;
cfg.pixelNumY = 100;
cfg.pixelNumZ = 20;
cfg.pixelLengthX = 3;
cfg.pixelLengthY = 3;
cfg.gridXCenters = centered_axis(cfg.pixelNumX, cfg.pixelLengthX);
cfg.gridYCenters = centered_axis(cfg.pixelNumY, cfg.pixelLengthY);
[cfg.gridX, cfg.gridY] = meshgrid(cfg.gridXCenters, cfg.gridYCenters);
cfg.backgroundRadiusMm = 120;
cfg.rodDiametersMm = 10:4:30;
cfg.rodEnergyKeV = [218, 440, 218, 440, 218, 440];
cfg.rodCenterRadiusMm = 60;
cfg.activityContrast = 6;
cfg.zIndices = 7:14;
cfg.energiesKeV = [440, 218];
cfg.countLevels = ["1e9", "1e10", "1e11"];
cfg.detectors = [ ...
    struct("name", "JSCC", "root", fullfile(repoRoot, ...
        "Figure_Local_SC_MultiOutput"), "iterMax", [5000, 10000, 20000]), ...
    struct("name", "EHE", "root", fullfile(repoRoot, ...
        "Figure_Local_SC_MultiOutput_EHE"), "iterMax", [100, 100, 100])];
cfg.colors = [ ...
    0.00, 0.45, 0.70
    0.90, 0.62, 0.00
    0.00, 0.62, 0.45
    0.80, 0.47, 0.65
    0.34, 0.71, 0.91
    0.84, 0.37, 0.00];
cfg.energyLineStyles = containers.Map({218, 440}, {"-", "--"});
cfg.crcLimits = [0, 1];
cfg.cnrLimits = [-2, 18];
cfg.exportDpi = 220;
end

function roi = build_roi_masks(cfg)
[x, y] = ndgrid(cfg.gridXCenters, cfg.gridYCenters);
background2d = x.^2 + y.^2 <= (0.85 * cfg.backgroundRadiusMm)^2;
roi.hot = cell(1, numel(cfg.rodDiametersMm));
for rodIdx = 1:numel(cfg.rodDiametersMm)
    theta = (rodIdx - 1) * pi / 3;
    cx = cfg.rodCenterRadiusMm * cos(theta);
    cy = cfg.rodCenterRadiusMm * sin(theta);
    radius = cfg.rodDiametersMm(rodIdx) / 2;
    roi.hot{rodIdx} = (x - cx).^2 + (y - cy).^2 <= (0.85 * radius)^2;
    exclusion = (x - cx).^2 + (y - cy).^2 <= (1.15 * radius)^2;
    background2d(exclusion) = false;
end
roi.background2d = background2d;
if ~any(background2d, "all") || any(cellfun(@(mask) ~any(mask, "all"), roi.hot))
    error("At least one configured ROI is empty on the Cartesian grid.");
end
end

function runPath = find_run(rootPath, countLevel, iterMax)
pattern = sprintf("ME_R20_E218-440_*_C%s_*_O1_SI%d_XTALK_BG1_*", ...
    countLevel, iterMax);
matches = dir(fullfile(rootPath, pattern));
matches = matches([matches.isdir]);
if numel(matches) ~= 1
    error("Expected one run matching %s under %s, found %d.", ...
        pattern, rootPath, numel(matches));
end
runPath = fullfile(matches(1).folder, matches(1).name);
end

function [polarPath, manifest] = read_run(runPath)
polarPath = fullfile(runPath, "Polar");
manifestPath = fullfile(polarPath, "run_manifest.json");
if ~exist(manifestPath, "file")
    error("Run manifest is missing: %s", manifestPath);
end
manifest = jsondecode(fileread(manifestPath));
end

function task = find_metric_task(tasks, energyKeV)
if energyKeV == 440
    expectedType = "Direct440";
else
    expectedType = "CrossTalkCorrected218";
end
matches = arrayfun(@(item) strcmp(string(item.type), expectedType), tasks);
if nnz(matches) ~= 1
    error("Expected one task of type %s, found %d.", expectedType, nnz(matches));
end
task = tasks(matches);
end

function history = read_history(polarPath, task, pixelNumPolar, pixelNumZ, pixelNum)
shape = double(task.history_shape(:)).';
if ~isequal(shape, [double(task.iterations) / double(task.save_iter_step), pixelNum])
    error("Manifest history shape is inconsistent for %s.", task.history_file);
end
history = read_float32_tensor(fullfile(polarPath, string(task.history_file)), ...
    [pixelNumPolar, pixelNumZ, shape(1)]);
end

function [crc, cnr, hotMean, backgroundMean, backgroundStd] = ...
        calculate_metrics(history, interpPlan, roi, rodIds, cfg)
snapshotNum = size(history, 3);
rodNum = numel(rodIds);
hotMean = zeros(rodNum, snapshotNum);
backgroundMean = zeros(1, snapshotNum);
backgroundStd = zeros(1, snapshotNum);
for snapshotIdx = 1:snapshotNum
    volume = polar_to_cartesian_volume(history(:, :, snapshotIdx), interpPlan);
    backgroundValues = volume(repmat(roi.background2d, 1, 1, cfg.pixelNumZ) & ...
        reshape(ismember(1:cfg.pixelNumZ, cfg.zIndices), 1, 1, []));
    backgroundMean(snapshotIdx) = mean(backgroundValues, "double");
    backgroundStd(snapshotIdx) = std(double(backgroundValues));
    for rodLocalIdx = 1:rodNum
        hotVolume = volume(:, :, cfg.zIndices);
        hotValues = hotVolume(repmat(roi.hot{rodIds(rodLocalIdx)}, ...
            1, 1, numel(cfg.zIndices)));
        hotMean(rodLocalIdx, snapshotIdx) = mean(hotValues, "double");
    end
end
crc = (hotMean - backgroundMean) ./ backgroundMean / (cfg.activityContrast - 1);
cnr = (hotMean - backgroundMean) ./ backgroundStd;
end

function tableOut = build_summary_table(series)
rows = repmat(struct("Detector", "", "CountLevel", "", "EnergyKeV", 0, ...
    "DiameterMm", 0, "PeakCNR", 0, "PeakCNRIteration", 0, ...
    "CRCAtPeakCNR", 0, "FinalCRC", 0, "FinalCNR", 0), 0, 1);
for idx = 1:numel(series)
    s = series(idx);
    for rodIdx = 1:numel(s.rodIds)
        [peakCnr, peakIdx] = max(s.cnr(rodIdx, :), [], "omitnan");
        row = struct("Detector", s.detector, "CountLevel", s.countLevel, ...
            "EnergyKeV", s.energyKeV, "DiameterMm", s.diametersMm(rodIdx), ...
            "PeakCNR", peakCnr, "PeakCNRIteration", s.iterations(peakIdx), ...
            "CRCAtPeakCNR", s.crc(rodIdx, peakIdx), ...
            "FinalCRC", s.crc(rodIdx, end), "FinalCNR", s.cnr(rodIdx, end));
        rows(end + 1, 1) = row; %#ok<AGROW>
    end
end
tableOut = struct2table(rows);
end

function tableOut = build_group_summary_table(series)
rows = repmat(struct("Detector", "", "CountLevel", "", "EnergyKeV", 0, ...
    "BestMeanCNRIteration", 0, "PeakMeanCNR", 0, ...
    "MeanCRCAtBestCNR", 0, "FinalMeanCRC", 0, "FinalMeanCNR", 0), 0, 1);
for idx = 1:numel(series)
    s = series(idx);
    meanCnr = mean(s.cnr, 1, "omitnan");
    [peakMeanCnr, peakIdx] = max(meanCnr, [], "omitnan");
    row = struct("Detector", s.detector, "CountLevel", s.countLevel, ...
        "EnergyKeV", s.energyKeV, ...
        "BestMeanCNRIteration", s.iterations(peakIdx), ...
        "PeakMeanCNR", peakMeanCnr, ...
        "MeanCRCAtBestCNR", mean(s.crc(:, peakIdx), "omitnan"), ...
        "FinalMeanCRC", mean(s.crc(:, end), "omitnan"), ...
        "FinalMeanCNR", mean(s.cnr(:, end), "omitnan"));
    rows(end + 1, 1) = row; %#ok<AGROW>
end
tableOut = struct2table(rows);
end

function f = render_overview(series, cfg, metricName)
metricLabel = upper(metricName);
f = figure("Color", "white", "Position", [80, 40, 1200, 940]);
layout = tiledlayout(numel(cfg.countLevels), numel(cfg.detectors), ...
    "TileSpacing", "compact", "Padding", "compact");
for countIdx = 1:numel(cfg.countLevels)
    for detectorIdx = 1:numel(cfg.detectors)
        ax = nexttile(layout, ...
            (countIdx - 1) * numel(cfg.detectors) + detectorIdx);
        pair = squeeze(series(detectorIdx, countIdx, :));
        handles = plot_energy_pair(ax, pair, metricName, cfg);
        title(ax, sprintf("%s, %s emitted photons", ...
            cfg.detectors(detectorIdx).name, cfg.countLevels(countIdx)));
        ylabel(ax, metricLabel);
        if countIdx == numel(cfg.countLevels)
            xlabel(ax, "Iteration");
        end
        if countIdx == 1
            legend(ax, handles, energy_rod_labels(pair), "Location", "best", ...
                "Box", "off", "FontSize", 8, "NumColumns", 2);
        end
    end
end
title(layout, sprintf("Contrast phantom %s: JSCC count-dependent iterations vs EHE 100 iterations", ...
    metricLabel));
end

function f = render_count_comparison(series, cfg, countIdx)
f = figure("Color", "white", "Position", [120, 60, 1200, 680]);
layout = tiledlayout(2, numel(cfg.detectors), ...
    "TileSpacing", "compact", "Padding", "compact");
for metricIdx = 1:2
    metricNames = ["crc", "cnr"];
    metricName = metricNames(metricIdx);
    for detectorIdx = 1:numel(cfg.detectors)
        ax = nexttile(layout, ...
            (metricIdx - 1) * numel(cfg.detectors) + detectorIdx);
        pair = squeeze(series(detectorIdx, countIdx, :));
        handles = plot_energy_pair(ax, pair, metricName, cfg);
        title(ax, sprintf("%s: corrected 218 + direct 440 keV", ...
            cfg.detectors(detectorIdx).name));
        ylabel(ax, upper(metricName));
        xlabel(ax, "Iteration");
        if metricIdx == 1
            legend(ax, handles, energy_rod_labels(pair), "Location", "best", ...
                "Box", "off", "FontSize", 8, "NumColumns", 2);
        end
    end
end
title(layout, sprintf("%s emitted photons: JSCC vs EHE CRC/CNR", cfg.countLevels(countIdx)));
end

function handles = plot_energy_pair(ax, pair, metricName, cfg)
hold(ax, "on");
handles = gobjects(0);
for energyIdx = 1:numel(pair)
    s = pair(energyIdx);
    values = s.(metricName);
    lineStyle = cfg.energyLineStyles(s.energyKeV);
    for rodIdx = 1:size(values, 1)
        markerIndices = unique(round(linspace(1, numel(s.iterations), ...
            min(7, numel(s.iterations)))));
        rodId = s.rodIds(rodIdx);
        handles(end + 1) = plot(ax, s.iterations, values(rodIdx, :), ...
            "LineWidth", 1.8, "LineStyle", lineStyle, ...
            "Color", cfg.colors(rodId, :), "Marker", "o", "MarkerSize", 3.5, ...
            "MarkerIndices", markerIndices); %#ok<AGROW>
    end
end
grid(ax, "on");
ax.Box = "on";
ax.LineWidth = 1;
ax.FontSize = 10;
xlim(ax, [0, s.iterations(end)]);
if metricName == "crc"
    yline(ax, 1, ":", "Color", [0.3, 0.3, 0.3], "HandleVisibility", "off");
    ylim(ax, cfg.crcLimits);
else
    ylim(ax, cfg.cnrLimits);
end
end

function labels = energy_rod_labels(pair)
labels = strings(0);
for energyIdx = 1:numel(pair)
    s = pair(energyIdx);
    labels = [labels, compose("%d keV, D=%g mm", ...
        s.energyKeV, s.diametersMm)]; %#ok<AGROW>
end
end

function factorPath = resolve_factor_path(repoRoot, manifest)
energyKeV = round(1000 * double(manifest.arguments.e0_list(1)));
suffix = string(manifest.arguments.factor_dir_suffix);
name = sprintf("%dkeV_RotateNum%d", energyKeV, manifest.arguments.rotate_num);
if strlength(suffix) > 0
    name = name + "_" + strip(suffix, "left", "_");
end
factorPath = fullfile(repoRoot, "Factors", name);
if ~exist(factorPath, "dir")
    error("Reference Factor directory not found: %s", factorPath);
end
end

function plan = build_interpolation_plan(sourceXY, gridX, gridY)
tri = delaunayTriangulation(double(sourceXY));
queryXY = [gridX(:), gridY(:)];
[triangleIds, weights] = pointLocation(tri, queryXY);
valid = ~isnan(triangleIds);
plan.outputSize = size(gridX);
plan.validMask = valid;
plan.vertexIds = tri.ConnectivityList(triangleIds(valid), :);
plan.weights = single(weights(valid, :));
end

function volume = polar_to_cartesian_volume(volumePolar, plan)
volume = zeros(plan.outputSize(2), plan.outputSize(1), size(volumePolar, 2), "single");
for zIdx = 1:size(volumePolar, 2)
    values = volumePolar(:, zIdx);
    interpolated = zeros(prod(plan.outputSize), 1, "single");
    interpolated(plan.validMask) = sum(values(plan.vertexIds) .* plan.weights, 2);
    volume(:, :, zIdx) = reshape(interpolated, plan.outputSize).';
end
end

function arr = load_named_array(matPath, csvPath, preferredName)
if exist(matPath, "file")
    data = load(matPath);
    if isfield(data, preferredName)
        arr = data.(preferredName);
    else
        names = fieldnames(data);
        if numel(names) ~= 1
            error("Cannot resolve variable %s in %s.", preferredName, matPath);
        end
        arr = data.(names{1});
    end
elseif exist(csvPath, "file")
    arr = readmatrix(csvPath);
else
    error("Cannot find %s or %s.", matPath, csvPath);
end
end

function tensor = read_float32_tensor(filePath, shape)
fid = fopen(filePath, "r");
if fid < 0
    error("Failed to open %s.", filePath);
end
cleanup = onCleanup(@() fclose(fid));
raw = fread(fid, "single=>single");
if numel(raw) ~= prod(shape)
    error("Unexpected float count in %s: expected %d, got %d.", ...
        filePath, prod(shape), numel(raw));
end
tensor = reshape(raw, shape);
end

function values = centered_axis(pixelNum, pixelLength)
values = (-pixelNum * pixelLength / 2 + pixelLength / 2):pixelLength: ...
    (pixelNum * pixelLength / 2 - pixelLength / 2);
end

function record = empty_record()
record = struct("Detector", "", "CountLevel", "", "EnergyKeV", 0, ...
    "RodId", 0, "DiameterMm", 0, "Iteration", 0, "CRC", 0, "CNR", 0, ...
    "HotMean", 0, "BackgroundMean", 0, "BackgroundStd", 0);
end

function value = empty_series()
value = struct("detector", "", "countLevel", "", "energyKeV", 0, ...
    "runPath", "", "iterations", [], "rodIds", [], "diametersMm", [], ...
    "crc", [], "cnr", [], "hotMean", [], "backgroundMean", [], ...
    "backgroundStd", []);
end

function ensure_dir(pathValue)
if ~exist(pathValue, "dir")
    mkdir(pathValue);
end
end
