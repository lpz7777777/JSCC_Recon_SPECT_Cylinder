interp_factor = 2;
PixelNumX = 100;
PixelNumY = 100;
PixelNumZ = 20;
PixelNumX_new = 2^interp_factor * (PixelNumX-1) + 1;
PixelNumY_new = 2^interp_factor * (PixelNumY-1) + 1;
PixelNumZ_new = 2^interp_factor * (PixelNumZ-1) + 1;

PixelLengthX = 3;
PixelLengthY = 3;

back_rod_r = 120;
rod_r = 5:2:15;
r_tocenter = 60;
num_rods = numel(rod_r);
legend_labels = build_diameter_labels(rod_r);

axis_font_size = 18;
title_font_size = 21;
legend_font_size = 15;

C = 6;

%% Select multiple folders
folderList = {};
while true
    if isempty(folderList)
        folderPath = uigetdir("./Figure_Dist_JSCCSD/", "Select run folder (Cancel to finish)");
    else
        folderPath = uigetdir("./Figure_Dist_JSCCSD/", ...
            sprintf("Select run folder #%d (Cancel to finish)", numel(folderList) + 1));
    end
    if isequal(folderPath, 0)
        break;
    end
    folderList{end+1} = folderPath; %#ok<AGROW>
    fprintf("Added folder: %s\n", folderPath);
end

num_runs = numel(folderList);
if num_runs == 0
    disp("No folders selected. Exiting.");
    return;
end
fprintf("Total folders selected: %d\n", num_runs);

%% Build masks (same for all runs)
start_id_z = 7;
end_id_z = 14;

Mask_b = zeros(PixelNumX, PixelNumY, PixelNumZ);
for i = 1 : PixelNumX
    for j = 1 : PixelNumY
        X = (i - 1/2 - PixelNumX/2) * PixelLengthX;
        Y = (j - 1/2 - PixelNumY/2) * PixelLengthY;

        if X^2 + Y^2 <= (back_rod_r*0.85)^2
            Mask_b(i, j, start_id_z:end_id_z) = 1;
        end
    end
end

rod_masks = cell(num_rods, 1);
for Id_R = 1 : num_rods
    R = rod_r(Id_R);
    Mask = zeros(PixelNumX, PixelNumY, PixelNumZ);
    Mask_tmp = zeros(PixelNumX, PixelNumY, PixelNumZ);

    theta_tmp = (Id_R-1) * pi/3;
    x_tmp = r_tocenter * cos(theta_tmp);
    y_tmp = r_tocenter * sin(theta_tmp);

    for i = 1 : PixelNumX
        for j = 1 : PixelNumY
            X = (i - 1/2 - PixelNumX/2) * PixelLengthX;
            Y = (j - 1/2 - PixelNumY/2) * PixelLengthY;

            if (X-x_tmp)^2 + (Y-y_tmp)^2 <= (0.85*R)^2
                Mask(i, j, start_id_z:end_id_z) = 1;
            end

            if (X-x_tmp)^2 + (Y-y_tmp)^2 <= (1.15*R)^2
                Mask_tmp(i, j, start_id_z:end_id_z) = 1;
            end
        end
    end

    Mask_b = Mask_b - Mask_tmp;
    rod_masks{Id_R} = logical(reshape(Mask, [], 1));
end
Mask_b = logical(reshape(Mask_b, [], 1));

%% Compute CRC/CNR for each run
crc_all = [];
cnr_all = [];
iterations = [];

for run_idx = 1 : num_runs
    folderPath = folderList{run_idx};
    Path = fullfile(folderPath, "Cartesian");

    iterInfoJsccsd = parse_iter_file(Path, "Image_JSCCSD_Iter_*");
    iter_max = iterInfoJsccsd.iterMax;
    iter_interval = iterInfoJsccsd.iterInterval;
    save_count = iterInfoJsccsd.saveCount;
    iterations_run = iter_interval:iter_interval:iter_max;

    img_jsccsd_iter = read_float32_tensor( ...
        fullfile(Path, iterInfoJsccsd.fileName), ...
        [PixelNumX * PixelNumY * PixelNumZ, save_count]);

    % img_jsccsd_iter_new = zeros(PixelNumX_new, PixelNumY_new, PixelNumZ_new, size(img_jsccsd_iter, 2));
    %
    % for i = 1 : size(img_jsccsd_iter, 2)
    %     img_jsccsd_iter_new(:, :, :, i) = interp3(reshape(img_jsccsd_iter(:, i), PixelNumX, PixelNumY, PixelNumZ), interp_factor);
    % end
    %
    % img_jsccsd_iter = reshape(img_jsccsd_iter_new, [], size(img_jsccsd_iter, 2));
    % PixelNumX = PixelNumX_new;
    % PixelNumY = PixelNumY_new;
    % PixelNumZ = PixelNumZ_new;

    mean_jsccsd = zeros(num_rods, save_count);
    std_jsccsd = zeros(num_rods, save_count);
    mean_jsccsd_b = zeros(1, save_count);
    std_jsccsd_b = zeros(1, save_count);

    for Id_R = 1 : num_rods
        img_jsccsd_iter_masked = img_jsccsd_iter(rod_masks{Id_R}, :);
        mean_jsccsd(Id_R, :) = mean(img_jsccsd_iter_masked, 1);
        std_jsccsd(Id_R, :) = std(img_jsccsd_iter_masked, 1);
    end

    img_jsccsd_iter_masked_b = img_jsccsd_iter(Mask_b, :);
    mean_jsccsd_b = mean(img_jsccsd_iter_masked_b, 1);
    std_jsccsd_b = std(img_jsccsd_iter_masked_b, 1);

    crc_jsccsd = (mean_jsccsd - mean_jsccsd_b) ./ mean_jsccsd_b / (C - 1);
    cnr_jsccsd = (mean_jsccsd - mean_jsccsd_b) ./ std_jsccsd_b;

    if isempty(crc_all)
        crc_all = zeros(num_runs, num_rods, save_count);
        cnr_all = zeros(num_runs, num_rods, save_count);
        iterations = iterations_run;
    end

    crc_all(run_idx, :, :) = crc_jsccsd;
    cnr_all(run_idx, :, :) = cnr_jsccsd;

    fprintf("Run %d/%d processed: %s\n", run_idx, num_runs, folderPath);
end

%% Average across runs
crc_mean = squeeze(mean(crc_all, 1));   % [num_rods, save_count]
crc_std = squeeze(std(crc_all, 0, 1));  % [num_rods, save_count]
cnr_mean = squeeze(mean(cnr_all, 1));   % [num_rods, save_count]
cnr_std = squeeze(std(cnr_all, 0, 1));  % [num_rods, save_count]

%% Plot errorbar version
f = figure;
t = tiledlayout(2, 1);
f.Position = [50, 50, 500, 800];
t.TileSpacing = "tight";
t.Padding = "tight";
Color = build_polar_palette(num_rods);

marker_step = max(1, round(numel(iterations) / 6));
idx_markers = marker_step:marker_step:length(iterations);
offset_scale = marker_step * 2;

ax = nexttile;
hold on
h_crc = gobjects(num_rods, 1);
for i = 1 : num_rods
    Color_tmp = Color(i, :);
    x_offset = (i - (num_rods + 1) / 2) * offset_scale;
    x_markers = iterations(idx_markers) + x_offset;

    h_crc(i) = plot(iterations, crc_mean(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2);

    y_at_shifted = interp1(iterations, crc_mean(i, :), x_markers);
    std_at_shifted = interp1(iterations, crc_std(i, :), x_markers);
    errorbar(x_markers, y_at_shifted, std_at_shifted, ...
        "Color", Color_tmp, ...
        "LineWidth", 1, ...
        "CapSize", 6, ...
        "LineStyle", "none", ...
        "Marker", ".", ...
        "MarkerSize", 14);
end
lgd = legend(h_crc, legend_labels, "FontSize", legend_font_size, "Location", "southeast");
lgd.NumColumns = 2;
lgd.Box = "off";
ylim([0 1]);
xlim([0 iterations(end)]);
xlabel("Iteration");
ylabel("CRC");
title(sprintf("JSCCSD CRC (mean \\pm std, n=%d)", num_runs));
style_axes(ax, axis_font_size, title_font_size);

ax = nexttile;
hold on
h_cnr = gobjects(num_rods, 1);
for i = 1 : num_rods
    Color_tmp = Color(i, :);
    x_offset = (i - (num_rods + 1) / 2) * offset_scale;
    x_markers = iterations(idx_markers) + x_offset;

    h_cnr(i) = plot(iterations, cnr_mean(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2);

    y_at_shifted = interp1(iterations, cnr_mean(i, :), x_markers);
    std_at_shifted = interp1(iterations, cnr_std(i, :), x_markers);
    errorbar(x_markers, y_at_shifted, std_at_shifted, ...
        "Color", Color_tmp, ...
        "LineWidth", 1, ...
        "CapSize", 6, ...
        "LineStyle", "none", ...
        "Marker", ".", ...
        "MarkerSize", 14);
end
lgd = legend(h_cnr, legend_labels, "FontSize", legend_font_size);
lgd.NumColumns = 2;
lgd.Box = "off";
ylim([0 8]);
xlim([0 iterations(end)]);
xlabel("Iteration");
ylabel("CNR");
title(sprintf("JSCCSD CNR (mean \\pm std, n=%d)", num_runs));
style_axes(ax, axis_font_size, title_font_size);

saveas(f, fullfile(fileparts(folderList{1}), "cnrcrc_jsccsd_average.png"));
savefig(f, fullfile(fileparts(folderList{1}), "cnrcrc_jsccsd_average.fig"));

%% Plot ribbon version (dashed upper/lower bounds)
f2 = figure;
t2 = tiledlayout(2, 1);
f2.Position = [50, 50, 500, 800];
t2.TileSpacing = "tight";
t2.Padding = "tight";

ax2_crc = nexttile;
hold on
h2_crc = gobjects(num_rods, 1);
for i = 1 : num_rods
    Color_tmp = Color(i, :);

    h2_crc(i) = plot(iterations, crc_mean(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2);

    plot(iterations, crc_mean(i, :) + crc_std(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 1, ...
        "LineStyle", "--");

    plot(iterations, crc_mean(i, :) - crc_std(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 1, ...
        "LineStyle", "--");
end
lgd2 = legend(h2_crc, legend_labels, "FontSize", legend_font_size, "Location", "southeast");
lgd2.NumColumns = 2;
lgd2.Box = "off";
ylim([0 1]);
xlim([0 iterations(end)]);
xlabel("Iteration");
ylabel("CRC");
title(sprintf("JSCCSD CRC (mean \\pm std, n=%d)", num_runs));
style_axes(ax2_crc, axis_font_size, title_font_size);

ax2_cnr = nexttile;
hold on
h2_cnr = gobjects(num_rods, 1);
for i = 1 : num_rods
    Color_tmp = Color(i, :);

    h2_cnr(i) = plot(iterations, cnr_mean(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2);

    plot(iterations, cnr_mean(i, :) + cnr_std(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 1, ...
        "LineStyle", "--");

    plot(iterations, cnr_mean(i, :) - cnr_std(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 1, ...
        "LineStyle", "--");
end
lgd2c = legend(h2_cnr, legend_labels, "FontSize", legend_font_size);
lgd2c.NumColumns = 2;
lgd2c.Box = "off";
ylim([0 15]);
xlim([0 iterations(end)]);
xlabel("Iteration");
ylabel("CNR");
title(sprintf("JSCCSD CNR (mean \\pm std, n=%d)", num_runs));
style_axes(ax2_cnr, axis_font_size, title_font_size);

saveas(f2, fullfile(fileparts(folderList{1}), "cnrcrc_jsccsd_average_ribbon.png"));
savefig(f2, fullfile(fileparts(folderList{1}), "cnrcrc_jsccsd_average_ribbon.fig"));


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


function labels = build_diameter_labels(rod_r)
diameters = 2 * rod_r(:).';
labels = strings(1, numel(diameters));
for idx = 1 : numel(diameters)
    labels(idx) = sprintf("D=%gmm", diameters(idx));
end
end


function style_axes(ax, axisFontSize, titleFontSize)
ax.Box = "on";
ax.LineWidth = 1.5;
ax.FontSize = axisFontSize;
ax.TitleFontSizeMultiplier = 1.0;
ax.Title.FontSize = titleFontSize;
ax.XLabel.FontSize = axisFontSize;
ax.YLabel.FontSize = axisFontSize;
end


function palette = build_polar_palette(numColors)
anchorPalette = [
    0.77, 0.33, 0.16
    0.89, 0.49, 0.18
    0.98, 0.71, 0.20
    0.99, 0.86, 0.34
    0.79, 0.86, 0.46
    0.48, 0.69, 0.40
];

if numColors <= size(anchorPalette, 1)
    idx = round(linspace(1, size(anchorPalette, 1), numColors));
    palette = anchorPalette(idx, :);
    return;
end

xAnchor = linspace(0, 1, size(anchorPalette, 1));
xTarget = linspace(0, 1, numColors);
palette = interp1(xAnchor, anchorPalette, xTarget, "linear");
end