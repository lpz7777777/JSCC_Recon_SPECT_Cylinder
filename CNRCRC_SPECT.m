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
rod_r = 8:2:18;
r_tocenter = 60;
num_rods = numel(rod_r);
legend_labels = build_diameter_labels(rod_r);
% axis_font_size = 12;
% title_font_size = 14;
% legend_font_size = 11;

axis_font_size = 18;
title_font_size = 21;
legend_font_size = 15;

C = 6;

folderPath = uigetdir("./Figure/");
if isequal(folderPath, 0)
    return
end
Path = fullfile(folderPath, "Cartesian");

iterInfoSc = parse_iter_file(Path, "Image_SC_Iter_*");
iterInfoJsccsd = parse_iter_file(Path, "Image_JSCCSD_Iter_*");
if iterInfoSc.iterMax ~= iterInfoJsccsd.iterMax || ...
        iterInfoSc.saveCount ~= iterInfoJsccsd.saveCount || ...
        iterInfoSc.iterInterval ~= iterInfoJsccsd.iterInterval
    error("SC and JSCCSD iteration files are inconsistent.");
end

iter_max = iterInfoSc.iterMax;
iter_interval = iterInfoSc.iterInterval;
save_count = iterInfoSc.saveCount;
iterations = iter_interval:iter_interval:iter_max;

img_sc_iter = read_float32_tensor( ...
    fullfile(Path, iterInfoSc.fileName), ...
    [PixelNumX * PixelNumY * PixelNumZ, save_count]);
img_jsccsd_iter = read_float32_tensor( ...
    fullfile(Path, iterInfoJsccsd.fileName), ...
    [PixelNumX * PixelNumY * PixelNumZ, save_count]);


% img_sc_iter_new = zeros(PixelNumX_new, PixelNumY_new, PixelNumZ_new, size(img_sc_iter, 2));
% img_jsccsd_iter_new = zeros(PixelNumX_new, PixelNumY_new, PixelNumZ_new, size(img_sc_iter, 2));
% 
% for i = 1 : size(img_sc_iter, 2)
%     img_sc_iter_new(:, :, :, i) = interp3(reshape(img_sc_iter(:, i), PixelNumX, PixelNumY, PixelNumZ), interp_factor);
%     img_jsccsd_iter_new(:, :, :, i) = interp3(reshape(img_jsccsd_iter(:, i), PixelNumX, PixelNumY, PixelNumZ), interp_factor);
% end
% 
% img_sc_iter = reshape(img_sc_iter_new, [], size(img_sc_iter, 2));
% img_jsccsd_iter = reshape(img_jsccsd_iter_new, [], size(img_jsccsd_iter, 2));

mean_sc = zeros(num_rods, size(img_sc_iter, 2));
mean_jsccsd = mean_sc;
std_sc = zeros(num_rods, size(img_sc_iter, 2));
std_jsccsd = mean_sc;

% PixelNumX = PixelNumX_new;
% PixelNumY = PixelNumY_new;
% PixelNumZ = PixelNumZ_new;

mean_sc_b = zeros(1, size(img_sc_iter, 2));
mean_jsccsd_b = mean_sc_b;

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
    
    img_sc_iter_masked = img_sc_iter(Mask, :);
    img_jsccsd_iter_masked = img_jsccsd_iter(Mask, :);

    mean_sc(Id_R, :) = mean(img_sc_iter_masked, 1);
    mean_jsccsd(Id_R, :) = mean(img_jsccsd_iter_masked, 1);
    std_sc(Id_R, :) = std(img_sc_iter_masked, 1);
    std_jsccsd(Id_R, :) = std(img_jsccsd_iter_masked, 1);
end

Mask_b = logical(reshape(Mask_b, [], 1));

img_sc_iter_masked = img_sc_iter(Mask_b, :);
img_jsccsd_iter_masked = img_jsccsd_iter(Mask_b, :);

mean_sc_b = mean(img_sc_iter_masked, 1);
mean_jsccsd_b = mean(img_jsccsd_iter_masked, 1);
std_sc_b = std(img_sc_iter_masked, 1);
std_jsccsd_b = std(img_jsccsd_iter_masked, 1);

%%
% 假設前面的數據計算部分保持不變
crc_sc = (mean_sc - mean_sc_b) ./ mean_sc_b / (C - 1);
cnr_sc = (mean_sc - mean_sc_b) ./ std_sc_b;

crc_jsccsd = (mean_jsccsd - mean_jsccsd_b) ./ mean_jsccsd_b / (C - 1);
cnr_jsccsd = (mean_jsccsd - mean_jsccsd_b) ./ std_jsccsd_b;

crc_ymax = max([crc_sc(:); crc_jsccsd(:)]);
cnr_ymax = max([cnr_sc(:); cnr_jsccsd(:)]);

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
t = tiledlayout(2, 2);
f.Position = [50, 50, 1000, 800];
t.TileSpacing = "tight";
t.Padding = "tight";
Color = build_polar_palette(num_rods);

% 定義 6 種不同的 Marker 形狀，以便區分不同的 D 值
MarkerList = repmat({'.'}, 1, num_rods); 
% 計算 Marker 的索引位置：從第 1 個點開始，每隔 100 個點畫一個
% 注意：這裡假設 iterations 的長度與數據列長度一致
marker_step = max(1, round(numel(iterations) / 6));
idx_markers = marker_step:marker_step:length(iterations); 

% Plot 1: SC CRC
ax = nexttile;
hold on
for i = 1 : num_rods
    Color_tmp = Color(i, :);
    
    % plot(iterations, crc_sc(i, :), ...
    %     "Color", Color_tmp, ...
    %     "Marker", MarkerList{i}, ...           % 設定標記形狀
    %     "MarkerSize", 20, ...           % 設定標記形狀     
    %     "MarkerIndices", idx_markers);         % 設定僅在特定索引處顯示標記

    plot(iterations, crc_sc(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...           % 設定標記形狀
        "MarkerSize", 20, ...           % 設定標記形狀     
        "MarkerIndices", idx_markers);         % 設定僅在特定索引處顯示標記
end
legend(legend_labels, "FontSize", legend_font_size);
% grid on
% grid minor
ylim([0 crc_ymax]);
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CRC");
title("SC CRC");
style_axes(ax, axis_font_size, title_font_size);

% Plot 2: JSCC CRC
ax = nexttile;
hold on
for i = 1 : num_rods
    Color_tmp = Color(i, :);

    % plot(iterations, crc_jsccsd(i, :), ...
    %     "Color", Color_tmp, ...
    %     "Marker", MarkerList{i}, ...
    %     "MarkerSize", 20, ...           % 設定標記形狀     
    %     "MarkerIndices", idx_markers);

    plot(iterations, crc_jsccsd(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...           % 設定標記形狀     
        "MarkerIndices", idx_markers);
end
legend(legend_labels, "FontSize", legend_font_size);
% grid on
% grid minor
ylim([0 crc_ymax]);
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CRC");
title("JSCC CRC");
style_axes(ax, axis_font_size, title_font_size);

% Plot 3: SC CNR
ax = nexttile;
hold on
for i = 1 : num_rods
    Color_tmp = Color(i, :);

    % plot(iterations, cnr_sc(i, :), ...
    %     "Color", Color_tmp, ...
    %     "Marker", MarkerList{i}, ...
    %     "MarkerSize", 20, ...           % 設定標記形狀     
    %     "MarkerIndices", idx_markers);

    plot(iterations, cnr_sc(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...           % 設定標記形狀     
        "MarkerIndices", idx_markers);
end
lgd = legend(legend_labels, "FontSize", legend_font_size);
lgd.NumColumns = 2;
lgd.Box = "off";
ylim([0 cnr_ymax]);
% grid on
% grid minor
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CNR");
title("SC CNR");
style_axes(ax, axis_font_size, title_font_size);

% Plot 4: JSCC CNR
ax = nexttile;
hold on
for i = 1 : num_rods
    Color_tmp = Color(i, :);

    % plot(iterations, cnr_jsccsd(i, :), ...
    %     "Color", Color_tmp, ...
    %     "Marker", MarkerList{i}, ...
    %     "MarkerSize", 20, ...           % 設定標記形狀     
    %     "MarkerIndices", idx_markers);

    plot(iterations, cnr_jsccsd(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...           % 設定標記形狀     
        "MarkerIndices", idx_markers);
end
lgd = legend(legend_labels, "FontSize", legend_font_size);
lgd.NumColumns = 2;
lgd.Box = "off";
% lgd.NumColumns = 2;
% legend('boxoff')
ylim([0 cnr_ymax]);
% grid on
% grid minor
% xlim([0, 4000]);
xlabel("Iteration");
ylabel("CNR");
title("JSCC CNR");
style_axes(ax, axis_font_size, title_font_size);

%%
f_cnr_compare = figure;
f_cnr_compare.Position = [120, 120, 1100, 500];
hold on

legend_entries = strings(1, 2 * num_rods);
for i = 1 : num_rods
    Color_tmp = Color(i, :);

    plot(iterations, cnr_sc(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2, ...
        "LineStyle", "--");
    legend_entries(2*i-1) = sprintf("SC %s", legend_labels(i));

    plot(iterations, cnr_jsccsd(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2, ...
        "LineStyle", "-", ...
        "Marker", "o", ...
        "MarkerSize", 6, ...
        "MarkerIndices", idx_markers);
    legend_entries(2*i) = sprintf("JSCC %s", legend_labels(i));
end

lgd = legend(legend_entries, "Location", "northeast");
lgd.NumColumns = 2;
lgd.FontSize = legend_font_size;
ylim([0 cnr_ymax]);
grid on
% grid minor
xlabel("Iteration");
ylabel("CNR");
title("CNR Comparison: SC vs JSCC");
style_axes(gca, axis_font_size, title_font_size);

saveas(f, fullfile(folderPath, "cnrcrc.png"));
savefig(f, fullfile(folderPath, "cnrcrc.fig"));
saveas(f_cnr_compare, fullfile(folderPath, "cnr_compare.png"));
savefig(f_cnr_compare, fullfile(folderPath, "cnr_compare.fig"));

%% Visual Comparison at Optimal Iteration
avg_cnr_sc = mean(cnr_sc, 1);
avg_cnr_jsccsd = mean(cnr_jsccsd, 1);

[~, best_idx_sc] = max(avg_cnr_sc);
[~, best_idx_jsccsd] = max(avg_cnr_jsccsd);

best_iter_sc = iterations(best_idx_sc);
best_iter_jsccsd = iterations(best_idx_jsccsd);

fprintf('Optimal Iteration SC: %d\n', best_iter_sc);
fprintf('Optimal Iteration JSCC: %d\n', best_iter_jsccsd);

% Reshape for visualization
img_sc_best = reshape(img_sc_iter(:, best_idx_sc), [PixelNumX, PixelNumY, PixelNumZ]);
img_jsccsd_best = reshape(img_jsccsd_iter(:, best_idx_jsccsd), [PixelNumX, PixelNumY, PixelNumZ]);

z_slice = floor(PixelNumZ/2);

%%
colorMap = flipud(gray(1024));
f_best = figure;
f_best.Position = [200, 200, 900, 400];

subplot(1, 2, 1);
imagesc(img_sc_best(:,:,z_slice));
colormap(colorMap);
axis image; colorbar;
title(sprintf('SC (Best Iter: %d)', best_iter_sc));
xlabel('X'); ylabel('Y');

subplot(1, 2, 2);
imagesc(img_jsccsd_best(:,:,z_slice));
colormap(colorMap);
axis image; colorbar;
title(sprintf('JSCC (Best Iter: %d)', best_iter_jsccsd));
xlabel('X'); ylabel('Y');

saveas(f_best, fullfile(folderPath, "best_iter.png"));
savefig(f_best, fullfile(folderPath, "best_iter.fig"));


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

