scriptDir = fileparts(mfilename("fullpath"));
folderPath = uigetdir(fullfile(scriptDir, "Result"));
if isequal(folderPath, 0)
    return;
end

betaPath = fullfile(folderPath, "SinglePhoton", "beta_values");
summaryPath = fullfile(folderPath, "SinglePhoton", "summary.json");
if ~exist(summaryPath, "file")
    error("summary.json not found under %s", fullfile(folderPath, "SinglePhoton"));
end

summary = jsondecode(fileread(summaryPath));
factorLabels = string(summary.config.factor_labels);
basePath = fullfile(folderPath, "SinglePhoton");
figure("Position", [100, 100, 620, 500]);
ax = axes();
energyTags = strings(size(factorLabels));
systemTags = strings(size(factorLabels));
for i = 1 : numel(factorLabels)
    label = factorLabels(i);
    tokens = regexp(char(label), '^([^_]+)(.*)$', 'tokens', 'once');
    if isempty(tokens)
        energyTags(i) = label;
        systemTags(i) = "";
    else
        energyTags(i) = string(tokens{1});
        systemTags(i) = string(tokens{2});
    end
end

uniqueEnergyTags = unique(energyTags, "stable");
lineStyles = {"-", ":", "--", "-."};

hold on;
for i = 1 : numel(factorLabels)
    label = factorLabels(i);
    varPath = fullfile(basePath, sprintf("Var_mean_%s", label));
    crcPath = fullfile(basePath, sprintf("CRC_mean_%s", label));

    fid = fopen(varPath, "r");
    varMean = fread(fid, "float32");
    fclose(fid);

    fid = fopen(crcPath, "r");
    crcMean = fread(fid, "float32");
    fclose(fid);

    validMask = (varMean > 0) & (crcMean > 0);
    if ~any(validMask)
        warning("No positive CRC/VAR samples for %s; skipped in log-scale plot.", label);
        continue;
    end

    energyIdx = find(uniqueEnergyTags == energyTags(i), 1, "first");
    lineStyle = lineStyles{mod(energyIdx - 1, numel(lineStyles)) + 1};
    if contains(systemTags(i), "SPECTEHENaI", "IgnoreCase", true)
        lineColor = [0.77, 0.33, 0.16];
    else
        lineColor = [0.48, 0.69, 0.40];
    end

    plot(ax, varMean(validMask), crcMean(validMask), ...
        "LineStyle", lineStyle, ...
        "Color", lineColor, ...
        "Marker", ".", ...
        "LineWidth", 1.2, ...
        "MarkerSize", 18, ...
        "DisplayName", label);
end
ax.XScale = "log";
ax.YScale = "log";
grid on;
% grid minor;
xlabel("VAR");
ylabel("CRC");
legend("Interpreter", "none", "Location", "best");
title("CRC-VAR Curve", "Interpreter", "none");
