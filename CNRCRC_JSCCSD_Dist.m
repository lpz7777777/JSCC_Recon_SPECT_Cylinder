interp_factor = 2;
PixelNumX = 100;
PixelNumY = 100;
PixelNumZ = 20;
PixelNumX_new = 2^interp_factor * (PixelNumX-1) + 1;
PixelNumY_new = 2^interp_factor * (PixelNumY-1) + 1;
PixelNumZ_new = 2^interp_factor * (PixelNumZ-1) + 1;
% PixelLengthX = 3/2^interp_factor;
% PixelLengthY = 3/2^interp_factor;
% PixelLengthZ = 3/2^interp_factor;

PixelLengthX = 3;
PixelLengthY = 3;

back_rod_r = 120;
% rod_r = 8:2:18;
rod_r = 5:2:15;
r_tocenter = 60;
num_rods = numel(rod_r);
legend_labels = build_diameter_labels(rod_r);

% axis_font_size = 18;
% title_font_size = 21;
% legend_font_size = 15;

axis_font_size = 21;
title_font_size = 24;
legend_font_size = 18;

C = 6;

folderPath = uigetdir("./Figure_Dist_JSCCSD/");
if isequal(folderPath, 0)
    return
end
Path = fullfile(folderPath, "Cartesian");

iterInfoJsccsd = parse_iter_file(Path, "Image_JSCCSD_Iter_*");
iter_max = iterInfoJsccsd.iterMax;
iter_interval = iterInfoJsccsd.iterInterval;
save_count = iterInfoJsccsd.saveCount;
iterations = iter_interval:iter_interval:iter_max;

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

mean_jsccsd = zeros(num_rods, size(img_jsccsd_iter, 2));
std_jsccsd = zeros(num_rods, size(img_jsccsd_iter, 2));

% PixelNumX = PixelNumX_new;
% PixelNumY = PixelNumY_new;
% PixelNumZ = PixelNumZ_new;

mean_jsccsd_b = zeros(1, size(img_jsccsd_iter, 2));

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

    Mask = logical(reshape(Mask, [], 1));
    img_jsccsd_iter_masked = img_jsccsd_iter(Mask, :);

    mean_jsccsd(Id_R, :) = mean(img_jsccsd_iter_masked, 1);
    std_jsccsd(Id_R, :) = std(img_jsccsd_iter_masked, 1);
end

Mask_b = logical(reshape(Mask_b, [], 1));

img_jsccsd_iter_masked = img_jsccsd_iter(Mask_b, :);
mean_jsccsd_b = mean(img_jsccsd_iter_masked, 1);
std_jsccsd_b = std(img_jsccsd_iter_masked, 1);

%%
crc_jsccsd = (mean_jsccsd - mean_jsccsd_b) ./ mean_jsccsd_b / (C - 1);
cnr_jsccsd = (mean_jsccsd - mean_jsccsd_b) ./ std_jsccsd_b;

crc_ymax = max(crc_jsccsd(:));
cnr_ymax = max(cnr_jsccsd(:));

if ~isfinite(crc_ymax) || crc_ymax <= 0
    crc_ymax = 1;
else
    crc_ymax = ceil(crc_ymax * 20) / 20;
end

if ~isfinite(cnr_ymax) || cnr_ymax <= 0
    cnr_ymax = 1;
else
    cnr_ymax = ceil(cnr_ymax * 10) / 10;
end

f = figure;
t = tiledlayout(2, 1);
f.Position = [50, 50, 500, 800];
t.TileSpacing = "tight";
t.Padding = "tight";
Color = build_polar_palette(num_rods);

MarkerList = repmat({'.'}, 1, num_rods);
marker_step = max(1, round(numel(iterations) / 6));
idx_markers = marker_step:marker_step:length(iterations);

ax = nexttile;
hold on
for i = 1 : num_rods
    Color_tmp = Color(i, :);
    plot(iterations, crc_jsccsd(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...
        "MarkerIndices", idx_markers);
end
lgd = legend(legend_labels, "FontSize", legend_font_size, "Location", "southeast");
lgd.NumColumns = 2;
lgd.Box = "off";
% ylim([0 crc_ymax]);
ylim([0 1]);
xlim([0 iter_max]);
xlabel("Iteration");
ylabel("CRC");
title("JSCCSD CRC");
style_axes(ax, axis_font_size, title_font_size);

ax = nexttile;
hold on
for i = 1 : num_rods
    Color_tmp = Color(i, :);
    plot(iterations, cnr_jsccsd(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...
        "MarkerIndices", idx_markers);
end
lgd = legend(legend_labels, "FontSize", legend_font_size);
lgd.NumColumns = 2;
lgd.Box = "off";
% ylim([0 cnr_ymax]);
ylim([0 15]);
xlim([0 iter_max]);
xlabel("Iteration");
ylabel("CNR");
title("JSCCSD CNR");
style_axes(ax, axis_font_size, title_font_size);

saveas(f, fullfile(folderPath, "cnrcrc_jsccsd.png"));
savefig(f, fullfile(folderPath, "cnrcrc_jsccsd.fig"));

%% Visual Comparison at Optimal Iteration
avg_cnr_jsccsd = mean(cnr_jsccsd, 1);
[~, best_idx_jsccsd] = max(avg_cnr_jsccsd);
best_iter_jsccsd = iterations(best_idx_jsccsd);

fprintf('Optimal Iteration JSCCSD: %d\n', best_iter_jsccsd);

img_jsccsd_best = reshape(img_jsccsd_iter(:, best_idx_jsccsd), [PixelNumX, PixelNumY, PixelNumZ]);
z_slice = floor(PixelNumZ/2);

%%
colorMap = flipud(gray(1024));
f_best = figure;
f_best.Position = [200, 200, 450, 400];

imagesc(img_jsccsd_best(:,:,z_slice));
colormap(colorMap);
axis image; colorbar;
title(sprintf('JSCCSD (Best Iter: %d)', best_iter_jsccsd));
xlabel('X'); ylabel('Y');

saveas(f_best, fullfile(folderPath, "best_iter_jsccsd.png"));
savefig(f_best, fullfile(folderPath, "best_iter_jsccsd.fig"));


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
