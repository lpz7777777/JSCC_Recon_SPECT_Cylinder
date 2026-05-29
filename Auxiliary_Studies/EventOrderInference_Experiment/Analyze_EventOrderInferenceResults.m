clear;
clc;

repoRoot = fileparts(mfilename("fullpath"));
defaultDir = repoRoot;

selectedResultDirs = strings(0, 1);

startDir = uigetdir(defaultDir, "Select one result folder");
if isequal(startDir, 0)
    return;
end
selectedResultDirs = string(startDir);

multiSelectAnswer = questdlg( ...
    "Do you want to add more result folders for comparison?", ...
    "Add More Folders", ...
    "Yes", "No", "No");

while strcmpi(multiSelectAnswer, "Yes")
    nextDir = uigetdir(defaultDir, "Select another result folder");
    if isequal(nextDir, 0)
        break;
    end
    selectedResultDirs(end + 1, 1) = string(nextDir); %#ok<AGROW>
    selectedResultDirs = unique(selectedResultDirs);
    multiSelectAnswer = questdlg( ...
        "Add one more result folder?", ...
        "Add More Folders", ...
        "Yes", "No", "No");
end

if isempty(selectedResultDirs)
    return;
end

allMethods = struct([]);
analysisLabels = strings(0, 1);

for fileIdx = 1:numel(selectedResultDirs)
    resultDir = char(selectedResultDirs(fileIdx));
    jsonPath = fullfile(resultDir, "visualization_metrics.json");
    if ~isfile(jsonPath)
        warning("visualization_metrics.json not found in %s. Skipped.", resultDir);
        continue;
    end
    rawText = fileread(jsonPath);
    analysis = jsondecode(rawText);

    if ~isfield(analysis, "methods") || isempty(analysis.methods)
        warning("No methods found in %s. Skipped.", jsonPath);
        continue;
    end

    analysisLabels(end + 1, 1) = string(analysis.analysis_label); %#ok<AGROW>
    methods = analysis.methods;
    if iscell(methods)
        methods = [methods{:}];
    elseif isstruct(methods)
        methods = methods(:);
    else
        warning("Unexpected methods format in %s. Skipped.", jsonPath);
        continue;
    end

    for methodIdx = 1:numel(methods)
        method = methods(methodIdx);
        method.source_json = string(jsonPath);
        method.analysis_label = string(analysis.analysis_label);
        if ~isfield(method, "display_name") || strlength(string(method.display_name)) == 0
            method.display_name = string(analysis.analysis_label) + " / " + string(method.name);
        end
        allMethods = [allMethods; method]; %#ok<AGROW>
    end
end

if isempty(allMethods)
    error("No valid visualization metrics were loaded.");
end

timestampTag = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
if numel(allMethods) == 1
    baseTag = sanitize_label(string(allMethods(1).analysis_label) + "_" + string(allMethods(1).name));
else
    baseTag = "compare_" + sanitize_label(strjoin(unique(analysisLabels), "_"));
end

outputPrefix = "event_order_inference_" + baseTag + "_" + timestampTag;
outputDir = char(selectedResultDirs(1));

summaryTable = build_summary_table(allMethods);
rejectionTable = build_rejection_table(allMethods);

summaryCsvPath = fullfile(outputDir, outputPrefix + "_summary.csv");
rejectionCsvPath = fullfile(outputDir, outputPrefix + "_rejection.csv");
reportTxtPath = fullfile(outputDir, outputPrefix + "_report.txt");
overviewPngPath = fullfile(outputDir, outputPrefix + "_overview.png");
overviewFigPath = fullfile(outputDir, outputPrefix + "_overview.fig");
rejectionPngPath = fullfile(outputDir, outputPrefix + "_rejection.png");
rejectionFigPath = fullfile(outputDir, outputPrefix + "_rejection.fig");
softPngPath = fullfile(outputDir, outputPrefix + "_soft.png");
softFigPath = fullfile(outputDir, outputPrefix + "_soft.fig");

writetable(summaryTable, summaryCsvPath);
writetable(rejectionTable, rejectionCsvPath);
write_report(reportTxtPath, allMethods);

figOverview = build_overview_figure(allMethods);
saveas(figOverview, overviewPngPath);
savefig(figOverview, overviewFigPath);

figRejection = build_rejection_figure(allMethods);
saveas(figRejection, rejectionPngPath);
savefig(figRejection, rejectionFigPath);

figSoft = build_soft_figure(allMethods);
saveas(figSoft, softPngPath);
savefig(figSoft, softFigPath);

fprintf("Finished.\n");
fprintf("Summary CSV:   %s\n", summaryCsvPath);
fprintf("Rejection CSV: %s\n", rejectionCsvPath);
fprintf("Report TXT:    %s\n", reportTxtPath);
fprintf("Overview PNG:  %s\n", overviewPngPath);
fprintf("Rejection PNG: %s\n", rejectionPngPath);
fprintf("Soft PNG:      %s\n", softPngPath);


function summaryTable = build_summary_table(methods)
rows = struct([]);
for idx = 1:numel(methods)
    method = methods(idx);
    row.analysis_label = string(method.analysis_label);
    row.method_name = string(method.name);
    row.display_name = string(method.display_name);
    row.event_count_mean = get_stat_mean(method.overall, "event_count");
    row.event_count_std = get_stat_std(method.overall, "event_count");
    row.accuracy_mean = get_stat_mean(method.overall, "accuracy");
    row.accuracy_std = get_stat_std(method.overall, "accuracy");
    row.balanced_accuracy_mean = get_stat_mean(method.overall, "balanced_accuracy");
    row.balanced_accuracy_std = get_stat_std(method.overall, "balanced_accuracy");
    row.front_first_accuracy_mean = get_stat_mean(method.overall, "front_first_accuracy");
    row.front_first_accuracy_std = get_stat_std(method.overall, "front_first_accuracy");
    row.rear_first_accuracy_mean = get_stat_mean(method.overall, "rear_first_accuracy");
    row.rear_first_accuracy_std = get_stat_std(method.overall, "rear_first_accuracy");
    row.truth_front_first_count_mean = get_stat_mean(method.overall, "truth_front_first_count");
    row.truth_rear_first_count_mean = get_stat_mean(method.overall, "truth_rear_first_count");
    row.correct_front_first_count_mean = get_stat_mean(method.overall, "correct_front_first_count");
    row.correct_rear_first_count_mean = get_stat_mean(method.overall, "correct_rear_first_count");
    row.pred_front_first_ratio_mean = get_stat_mean(method.overall, "pred_front_first_ratio");
    row.truth_front_first_ratio_mean = get_stat_mean(method.overall, "truth_front_first_ratio");

    if isfield(method, "soft_metrics")
        row.brier_mean = get_stat_mean(method.soft_metrics, "classifier_brier_score");
        row.logloss_mean = get_stat_mean(method.soft_metrics, "classifier_log_loss");
        row.avg_true_prob_mean = get_stat_mean(method.soft_metrics, "classifier_avg_true_class_prob");
    else
        row.brier_mean = NaN;
        row.logloss_mean = NaN;
        row.avg_true_prob_mean = NaN;
    end

    rows = [rows; row]; %#ok<AGROW>
end

summaryTable = struct2table(rows);
end


function rejectionTable = build_rejection_table(methods)
rows = struct([]);
for idx = 1:numel(methods)
    method = methods(idx);
    if ~isfield(method, "rejection_sweep") || isempty(method.rejection_sweep)
        continue;
    end

    sweep = method.rejection_sweep;
    for sweepIdx = 1:numel(sweep)
        item = sweep(sweepIdx);
        row.analysis_label = string(method.analysis_label);
        row.method_name = string(method.name);
        row.display_name = string(method.display_name);
        row.reject_fraction_target = item.reject_fraction_target;
        row.reject_fraction_actual_mean = get_stat_mean(item, "reject_fraction_actual");
        row.retained_count_mean = get_stat_mean(item, "retained_count");
        row.retained_accuracy_mean = get_stat_mean(item, "accuracy");
        row.retained_balanced_accuracy_mean = get_stat_mean(item, "balanced_accuracy");
        row.retained_front_first_accuracy_mean = get_stat_mean(item, "front_first_accuracy");
        row.retained_rear_first_accuracy_mean = get_stat_mean(item, "rear_first_accuracy");
        row.retained_truth_front_first_count_mean = get_stat_mean(item, "truth_front_first_count");
        row.retained_truth_rear_first_count_mean = get_stat_mean(item, "truth_rear_first_count");
        row.retained_correct_front_first_count_mean = get_stat_mean(item, "correct_front_first_count");
        row.retained_correct_rear_first_count_mean = get_stat_mean(item, "correct_rear_first_count");
        rows = [rows; row]; %#ok<AGROW>
    end
end

rejectionTable = struct2table(rows);
end


function write_report(outputPath, methods)
fid = fopen(outputPath, "w");
if fid < 0
    error("Failed to open %s for writing.", outputPath);
end
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, "Event Order Inference Visualization Report\n");
fprintf(fid, "Generated at: %s\n\n", char(datetime("now")));

for idx = 1:numel(methods)
    method = methods(idx);
    fprintf(fid, "Method: %s\n", char(string(method.display_name)));
    fprintf(fid, "  event_count_mean          = %.6f\n", get_stat_mean(method.overall, "event_count"));
    fprintf(fid, "  accuracy_mean             = %.6f\n", get_stat_mean(method.overall, "accuracy"));
    fprintf(fid, "  balanced_accuracy_mean    = %.6f\n", get_stat_mean(method.overall, "balanced_accuracy"));
    fprintf(fid, "  front_first_accuracy_mean = %.6f\n", get_stat_mean(method.overall, "front_first_accuracy"));
    fprintf(fid, "  rear_first_accuracy_mean  = %.6f\n", get_stat_mean(method.overall, "rear_first_accuracy"));
    fprintf(fid, "  truth_front_count_mean    = %.6f\n", get_stat_mean(method.overall, "truth_front_first_count"));
    fprintf(fid, "  truth_rear_count_mean     = %.6f\n", get_stat_mean(method.overall, "truth_rear_first_count"));
    fprintf(fid, "  correct_front_count_mean  = %.6f\n", get_stat_mean(method.overall, "correct_front_first_count"));
    fprintf(fid, "  correct_rear_count_mean   = %.6f\n", get_stat_mean(method.overall, "correct_rear_first_count"));
    if isfield(method, "soft_metrics")
        fprintf(fid, "  brier_mean                = %.6f\n", get_stat_mean(method.soft_metrics, "classifier_brier_score"));
        fprintf(fid, "  logloss_mean              = %.6f\n", get_stat_mean(method.soft_metrics, "classifier_log_loss"));
        fprintf(fid, "  avg_true_prob_mean        = %.6f\n", get_stat_mean(method.soft_metrics, "classifier_avg_true_class_prob"));
    end
    fprintf(fid, "\n");
end
end


function fig = build_overview_figure(methods)
displayNames = string({methods.display_name});
accuracy = arrayfun(@(m) get_stat_mean(m.overall, "accuracy"), methods);
balanced = arrayfun(@(m) get_stat_mean(m.overall, "balanced_accuracy"), methods);
frontAcc = arrayfun(@(m) get_stat_mean(m.overall, "front_first_accuracy"), methods);
rearAcc = arrayfun(@(m) get_stat_mean(m.overall, "rear_first_accuracy"), methods);
truthFront = arrayfun(@(m) get_stat_mean(m.overall, "truth_front_first_count"), methods);
truthRear = arrayfun(@(m) get_stat_mean(m.overall, "truth_rear_first_count"), methods);
correctFront = arrayfun(@(m) get_stat_mean(m.overall, "correct_front_first_count"), methods);
correctRear = arrayfun(@(m) get_stat_mean(m.overall, "correct_rear_first_count"), methods);
eventCount = arrayfun(@(m) get_stat_mean(m.overall, "event_count"), methods);

fig = figure("Color", "w", "Name", "Event Order Inference Overview", "Position", [120, 80, 1600, 900]);
tiledlayout(fig, 2, 3, "TileSpacing", "compact", "Padding", "compact");

nexttile;
bar(categorical(displayNames), accuracy);
ylabel("Accuracy");
title("Overall Accuracy");
grid on;
ylim([0.7, 1.0]);
set(gca, "TickLabelInterpreter", "none");

nexttile;
bar(categorical(displayNames), balanced);
ylabel("Balanced Accuracy");
title("Balanced Accuracy");
grid on;
ylim([0.7, 1.0]);
set(gca, "TickLabelInterpreter", "none");

nexttile;
bar(categorical(displayNames), [frontAcc(:), rearAcc(:)], "grouped");
ylabel("Accuracy");
title("Class-wise Accuracy");
legend("Front-first truth", "Rear-first truth", "Location", "best", "Interpreter", "none");
grid on;
ylim([0.7, 1.0]);
set(gca, "TickLabelInterpreter", "none");

nexttile;
bar(categorical(displayNames), [truthFront(:), truthRear(:)], "stacked");
ylabel("Event Count");
title("Truth Counts");
legend("Front-first truth", "Rear-first truth", "Location", "best", "Interpreter", "none");
grid on;
set(gca, "TickLabelInterpreter", "none");

nexttile;
bar(categorical(displayNames), [correctFront(:), correctRear(:)], "stacked");
ylabel("Correct Count");
title("Correct Counts by Truth Class");
legend("Correct front-first", "Correct rear-first", "Location", "best", "Interpreter", "none");
grid on;
set(gca, "TickLabelInterpreter", "none");

nexttile;
bar(categorical(displayNames), eventCount);
ylabel("Event Count");
title("Retained Event Count");
grid on;
set(gca, "TickLabelInterpreter", "none");
end


function fig = build_rejection_figure(methods)
fig = figure("Color", "w", "Name", "Event Order Inference Rejection", "Position", [140, 100, 1600, 900]);
tiledlayout(fig, 2, 3, "TileSpacing", "compact", "Padding", "compact");

nexttile;
hold on;
for idx = 1:numel(methods)
    plot_rejection_curve(methods(idx), "accuracy", "Overall Accuracy");
end
hold off;
legend(string({methods.display_name}), "Location", "best", "Interpreter", "none");
grid on;
ylim([0.75, 1.0]);
set(gca, "TickLabelInterpreter", "none");

nexttile;
hold on;
for idx = 1:numel(methods)
    plot_rejection_curve(methods(idx), "front_first_accuracy", "Front-first Accuracy");
end
hold off;
legend(string({methods.display_name}), "Location", "best", "Interpreter", "none");
grid on;
ylim([0.75, 1.0]);
set(gca, "TickLabelInterpreter", "none");

nexttile;
hold on;
for idx = 1:numel(methods)
    plot_rejection_curve(methods(idx), "rear_first_accuracy", "Rear-first Accuracy");
end
hold off;
legend(string({methods.display_name}), "Location", "best", "Interpreter", "none");
grid on;
ylim([0.75, 1.0]);
set(gca, "TickLabelInterpreter", "none");

nexttile;
hold on;
for idx = 1:numel(methods)
    plot_rejection_curve(methods(idx), "balanced_accuracy", "Balanced Accuracy");
end
hold off;
legend(string({methods.display_name}), "Location", "best", "Interpreter", "none");
grid on;
ylim([0.75, 1.0]);
set(gca, "TickLabelInterpreter", "none");

nexttile;
hold on;
for idx = 1:numel(methods)
    sweep = methods(idx).rejection_sweep;
    x = arrayfun(@(s) s.reject_fraction_target, sweep);
    y = arrayfun(@(s) get_stat_mean(s, "retained_count"), sweep);
    plot(x, y, "-o", "LineWidth", 1.6, "MarkerSize", 6);
end
hold off;
xlabel("Rejected Fraction");
ylabel("Retained Event Count");
title("Retained Count");
legend(string({methods.display_name}), "Location", "best", "Interpreter", "none");
grid on;
set(gca, "TickLabelInterpreter", "none");

nexttile;
hold on;
for idx = 1:numel(methods)
    sweep = methods(idx).rejection_sweep;
    x = arrayfun(@(s) s.reject_fraction_target, sweep);
    totalCount = get_stat_mean(methods(idx).overall, "event_count");
    y = arrayfun(@(s) get_stat_mean(s, "retained_count"), sweep) ./ max(totalCount, eps);
    plot(x, y, "-o", "LineWidth", 1.6, "MarkerSize", 6);
end
hold off;
xlabel("Rejected Fraction");
ylabel("Retained Fraction");
title("Retained Fraction");
legend(string({methods.display_name}), "Location", "best", "Interpreter", "none");
grid on;
ylim([0.5, 1.05]);
set(gca, "TickLabelInterpreter", "none");
end


function fig = build_soft_figure(methods)
hasSoft = arrayfun(@(m) isfield(m, "soft_metrics"), methods);
softMethods = methods(hasSoft);

fig = figure("Color", "w", "Name", "Event Order Inference Soft Metrics", "Position", [160, 120, 1600, 900]);
if isempty(softMethods)
    axes(fig);
    axis off;
    text(0.5, 0.5, "No soft metrics available in the selected files.", ...
        "HorizontalAlignment", "center", "FontSize", 14);
    return;
end

tiledlayout(fig, 2, 2, "TileSpacing", "compact", "Padding", "compact");
displayNames = string({softMethods.display_name});
brierVals = arrayfun(@(m) get_stat_mean(m.soft_metrics, "classifier_brier_score"), softMethods);
logLossVals = arrayfun(@(m) get_stat_mean(m.soft_metrics, "classifier_log_loss"), softMethods);
trueProbVals = arrayfun(@(m) get_stat_mean(m.soft_metrics, "classifier_avg_true_class_prob"), softMethods);

nexttile;
bar(categorical(displayNames), brierVals);
ylabel("Brier Score");
title("Soft Metric: Brier Score");
grid on;
set(gca, "TickLabelInterpreter", "none");

nexttile;
bar(categorical(displayNames), logLossVals);
ylabel("Log Loss");
title("Soft Metric: Log Loss");
grid on;
set(gca, "TickLabelInterpreter", "none");

nexttile;
bar(categorical(displayNames), trueProbVals);
ylabel("Average True-class Probability");
title("Soft Metric: Avg True-class Probability");
grid on;
ylim([0.5, 1.0]);
set(gca, "TickLabelInterpreter", "none");

nexttile;
hold on;
plot([0, 1], [0, 1], "k--", "LineWidth", 1.2);
for idx = 1:numel(softMethods)
    bins = softMethods(idx).calibration_bins;
    x = arrayfun(@(b) b.mean_pred_prob, bins);
    y = arrayfun(@(b) b.empirical_front_first_ratio, bins);
    valid = isfinite(x) & isfinite(y);
    plot(x(valid), y(valid), "-o", "LineWidth", 1.6, "MarkerSize", 6);
end
hold off;
xlabel("Mean Predicted P(front-first)");
ylabel("Empirical Front-first Ratio");
title("Calibration Curve");
legend(["Ideal", displayNames], "Location", "best", "Interpreter", "none");
grid on;
axis([0, 1, 0, 1]);
set(gca, "TickLabelInterpreter", "none");
end


function plot_rejection_curve(method, fieldName, plotTitle)
sweep = method.rejection_sweep;
x = arrayfun(@(s) s.reject_fraction_target, sweep);
y = arrayfun(@(s) get_stat_mean(s, fieldName), sweep);
plot(x, y, "-o", "LineWidth", 1.6, "MarkerSize", 6);
xlabel("Rejected Fraction");
ylabel(plotTitle);
title(plotTitle + " vs Reject Fraction");
end


function value = get_stat_mean(statStruct, fieldName)
value = get_stat_value(statStruct, fieldName, "mean");
end


function value = get_stat_std(statStruct, fieldName)
value = get_stat_value(statStruct, fieldName, "std");
end


function value = get_stat_value(statStruct, fieldName, statName)
if ~isfield(statStruct, fieldName)
    value = NaN;
    return;
end
fieldValue = statStruct.(fieldName);
if isstruct(fieldValue) && isfield(fieldValue, statName)
    value = fieldValue.(statName);
elseif isnumeric(fieldValue) && isscalar(fieldValue)
    value = fieldValue;
else
    value = NaN;
end
end


function out = sanitize_label(textIn)
textIn = string(textIn);
out = regexprep(textIn, "[^a-zA-Z0-9_]+", "_");
out = regexprep(out, "_+", "_");
out = regexprep(out, "^_|_$", "");
if strlength(out) == 0
    out = "result";
end
end
