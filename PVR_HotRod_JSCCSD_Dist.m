%% PVR_HotRod_JSCCSD_Dist.m
% =========================================================================
% 功能：计算 JSCCSD 重建结果的 Hot-Rod 模体峰值-谷值比（PVR）
%       仿照 PVR_HotRod_SC_Dist.m，针对 Figure_Dist_JSCCSD 目录下的数据
% =========================================================================
%
% 算法流程：
%   1. 用户选择 Figure_Dist_JSCCSD 目录
%   2. 读取各迭代的 JSCCSD 重建图像
%   3. 构建Hot-Rod模体几何结构（6组等腰三角形排列的热棒阵列）
%   4. 通过图像匹配自动对齐模体几何位置（旋转+翻转搜索）
%   5. 对每组阵列，计算峰值均值和谷值均值
%   6. 计算 PVR = peak_mean / valley_mean
%   7. 绘制 PVR 随迭代次数变化的曲线
%   8. 保存结果图像和数据文件
%
% 输入：Figure_Dist_JSCCSD/Cartesian/Image_JSCCSD_Iter_<max>_<count>（float32 二进制）
% 输出：pvr_hotrod_jsccsd.png/fig, pvr_hotrod_jsccsd_curve.csv 等

PixelNumX = 100;    % 直角坐标图像X方向像素数
PixelNumY = 100;    % 直角坐标图像Y方向像素数
PixelNumZ = 20;     % Z方向层数
PixelLengthX = 3;   % X方向像素尺寸（mm）
PixelLengthY = 3;   % Y方向像素尺寸（mm）
PixelLengthZ = 3;   % Z方向像素尺寸（mm）

% Hot-Rod 模体几何参数（与 GenProj_SPECT_PolarCoor.m 中一致）
rod_num = [10, 6, 6, 3, 3, 3];       % 每组阵列的热棒数量
rod_r = 5:2:15;                         % 每组阵列的热棒半径（mm）
back_rod_r = 200;                       % 阵列中心到模体中心的距离（mm）

% 备选参数（更细的热棒）
% back_rod_r = 175;
% rod_r = 2.5:1:7.5;
% rod_num = [21, 15, 10, 10, 6, 6];

center_factor = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4];  % 阵列中心的径向位置因子
rod_h = 30;       % 热棒高度（mm）

% 取平均的Z层范围（避开边缘层，减少部分容积效应）
start_id_z = 7;
end_id_z = 14;

% 字体大小设置
axis_font_size = 18;
title_font_size = 21;
legend_font_size = 15;

%% ==================== 选择数据目录 ====================
% 弹出文件夹选择对话框，选择 Figure_Dist_JSCCSD 目录
folderPath = uigetdir("./Figure_Dist_JSCCSD/");
if isequal(folderPath, 0)
    return
end

%% ==================== 读取迭代数据 ====================
Path = fullfile(folderPath, "Cartesian");

% 解析文件名中的迭代信息
iterInfoJSCCSD = parse_iter_file(Path, "Image_JSCCSD_Iter_*");
iter_max = iterInfoJSCCSD.iterMax;           % 最大迭代次数
iter_interval = iterInfoJSCCSD.iterInterval; % 迭代保存间隔
save_count = iterInfoJSCCSD.saveCount;       % 保存次数
iterations = iter_interval:iter_interval:iter_max;  % 迭代次数序列

% 读取所有迭代的重建图像（float32 二进制格式）
img_jsccsd_iter = read_float32_tensor( ...
    fullfile(Path, iterInfoJSCCSD.fileName), ...
    [PixelNumX * PixelNumY * PixelNumZ, save_count]);

%% ==================== 构建坐标网格 ====================
x_coords = ((1 : PixelNumX) - 0.5 - PixelNumX / 2) * PixelLengthX;
y_coords = ((1 : PixelNumY) - 0.5 - PixelNumY / 2) * PixelLengthY;
[y_grid, x_grid] = meshgrid(y_coords, x_coords);

%% ==================== 构建Hot-Rod几何结构并自动对齐 ====================
% 构建6组等腰三角形排列的热棒阵列几何结构
array_geom = build_hotrod_array_geometry(back_rod_r, rod_r, rod_num, center_factor);

% 使用最后一次迭代的图像作为参考，自动搜索最佳旋转和翻转对齐
ref_img_volume = reshape(img_jsccsd_iter(:, end), [PixelNumX, PixelNumY, PixelNumZ]);
ref_img_plane = mean(ref_img_volume(:, :, start_id_z:end_id_z), 3);
[array_geom, pose_info] = align_hotrod_geometry(array_geom, ref_img_plane, x_grid, y_grid);

array_num = numel(array_geom);
legend_labels = build_array_labels(array_geom);

%% ==================== 计算PVR ====================
pvr_jsccsd = zeros(array_num, save_count);
peak_mean_jsccsd = zeros(array_num, save_count);
valley_mean_jsccsd = zeros(array_num, save_count);

for iter_idx = 1 : save_count
    % 提取当前迭代的3D图像，并对指定Z层取平均得到2D切面
    img_volume = reshape(img_jsccsd_iter(:, iter_idx), [PixelNumX, PixelNumY, PixelNumZ]);
    img_plane = mean(img_volume(:, :, start_id_z:end_id_z), 3);

    for array_idx = 1 : array_num
        % 在峰值位置（热棒中心）采样圆盘ROI的均值
        peak_values = sample_disk_means( ...
            img_plane, ...
            x_grid, ...
            y_grid, ...
            array_geom(array_idx).peak_xy, ...
            array_geom(array_idx).peak_roi_radius_mm);

        % 在谷值位置（热棒之间的中点）采样圆盘ROI的均值
        valley_values = sample_disk_means( ...
            img_plane, ...
            x_grid, ...
            y_grid, ...
            array_geom(array_idx).valley_xy, ...
            array_geom(array_idx).valley_roi_radius_mm);

        peak_mean_jsccsd(array_idx, iter_idx) = mean(peak_values);
        valley_mean_jsccsd(array_idx, iter_idx) = mean(valley_values);
        % PVR = 峰值均值 / 谷值均值（谷值过小时截断避免除零）
        pvr_jsccsd(array_idx, iter_idx) = peak_mean_jsccsd(array_idx, iter_idx) / max(valley_mean_jsccsd(array_idx, iter_idx), 1e-12);
    end
end

%% ==================== 绘制PVR曲线 ====================
% 自动计算Y轴范围
pvr_ymax = max(pvr_jsccsd(:));
if ~isfinite(pvr_ymax) || pvr_ymax <= 0
    pvr_ymax = 1;
else
    pvr_ymax = ceil(pvr_ymax * 10) / 10;
end

pvr_positive = pvr_jsccsd(pvr_jsccsd > 0 & isfinite(pvr_jsccsd));
if isempty(pvr_positive)
    pvr_ymin = 1e-3;
else
    pvr_ymin = min(pvr_positive) * 0.9;
end

f = figure;
t = tiledlayout(1, 1);
f.Position = [50, 50, 900, 600];
t.TileSpacing = "tight";
t.Padding = "tight";
Color = build_polar_palette(array_num);
MarkerList = repmat({'.'}, 1, array_num);
marker_step = max(1, round(numel(iterations) / 6));
idx_markers = marker_step:marker_step:length(iterations);

ax = nexttile;
hold on
for i = 1 : array_num
    Color_tmp = Color(i, :);
    plot(iterations, pvr_jsccsd(i, :), ...
        "Color", Color_tmp, ...
        "LineWidth", 2, ...
        "Marker", MarkerList{i}, ...
        "MarkerSize", 20, ...
        "MarkerIndices", idx_markers);
end
% 绘制Rayleigh判据参考线（分辨极限PVR=1.23）
rayleigh_handle = yline(1.23, "--", "Rayleigh = 1.23", ...
    "Color", [0.20, 0.20, 0.20], ...
    "LineWidth", 1.5, ...
    "LabelHorizontalAlignment", "left", ...
    "LabelVerticalAlignment", "bottom");
lgd = legend(legend_labels, "FontSize", legend_font_size, "Location", "northeast");
lgd.Box = "off";
set(ax, "YScale", "log");
ylim([pvr_ymin pvr_ymax]);
xlim([0 iter_max]);
xlabel("Iteration");
ylabel("Average Peak-Valley Ratio");
title("Hot-Rod Array Average Peak-Valley Ratio");
style_axes(ax, axis_font_size, title_font_size);

title(t, sprintf("JSCCSD Hot-Rod PVR, z=%d:%d, rot=%g deg, flipX=%d, flipY=%d: %s", ...
    start_id_z, end_id_z, pose_info.rotation_deg, pose_info.flip_x, pose_info.flip_y, folderPath), ...
    "Interpreter", "none");
saveas(f, fullfile(folderPath, "pvr_hotrod_jsccsd.png"));
savefig(f, fullfile(folderPath, "pvr_hotrod_jsccsd.fig"));

%% ==================== 保存数据文件 ====================
% 阵列参数表
result_table = [(1:array_num).', [array_geom(:).rod_radius_mm].', [array_geom(:).rod_count].'];
writematrix(result_table, fullfile(folderPath, "pvr_hotrod_jsccsd_arrays.csv"));
% PVR曲线数据
writematrix([iterations(:), pvr_jsccsd.'], fullfile(folderPath, "pvr_hotrod_jsccsd_curve.csv"));
% 峰值均值数据
writematrix([iterations(:), peak_mean_jsccsd.'], fullfile(folderPath, "pvr_hotrod_jsccsd_peak_mean.csv"));
% 谷值均值数据
writematrix([iterations(:), valley_mean_jsccsd.'], fullfile(folderPath, "pvr_hotrod_jsccsd_valley_mean.csv"));
% 对齐姿态数据
writematrix([pose_info.rotation_deg, pose_info.flip_x, pose_info.flip_y, pose_info.score], ...
    fullfile(folderPath, "pvr_hotrod_jsccsd_pose.csv"));

%% ==================== 绘制调试图（采样位置叠加） ====================
% 找到平均PVR最高的迭代作为最优迭代
[~, best_iter_idx] = max(mean(pvr_jsccsd, 1));
best_img_volume = reshape(img_jsccsd_iter(:, best_iter_idx), [PixelNumX, PixelNumY, PixelNumZ]);
best_img_plane = mean(best_img_volume(:, :, start_id_z:end_id_z), 3);

f_debug = figure;
f_debug.Position = [80, 80, 1000, 850];
imagesc(y_coords, x_coords, best_img_plane);
axis equal;
axis tight;
colormap(flipud(gray(1024)));
hold on;
for array_idx = 1 : array_num
    color_tmp = Color(array_idx, :);
    % 标记峰值采样位置（圆圈）
    plot(array_geom(array_idx).peak_xy(:, 2), array_geom(array_idx).peak_xy(:, 1), ...
        "o", "Color", color_tmp, "MarkerSize", 6, "LineWidth", 1.2);
    % 标记谷值采样位置（十字）
    plot(array_geom(array_idx).valley_xy(:, 2), array_geom(array_idx).valley_xy(:, 1), ...
        "+", "Color", color_tmp, "MarkerSize", 6, "LineWidth", 1.2);
end
colorbar;
xlabel("y (mm)");
ylabel("x (mm)");
title(sprintf("JSCCSD Hot-Rod Peak(o)/Valley(+) Sampling Positions, Iter=%d", iterations(best_iter_idx)));
style_axes(gca, axis_font_size, title_font_size);
saveas(f_debug, fullfile(folderPath, "pvr_hotrod_jsccsd_debug_positions.png"));
savefig(f_debug, fullfile(folderPath, "pvr_hotrod_jsccsd_debug_positions.fig"));

fprintf('JSCCSD Hot-Rod PVR 计算完成！\n');
fprintf('最优迭代: %d (平均PVR = %.4f)\n', iterations(best_iter_idx), mean(pvr_jsccsd(:, best_iter_idx)));


%% ==================== 辅助函数 ====================

function iterInfo = parse_iter_file(folderPath, pattern)
% 从文件名中解析迭代信息
% 例如 "Image_JSCCSD_Iter_5000_100" → iterMax=5000, saveCount=100, iterInterval=50
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
% 读取float32二进制文件并reshape为指定形状
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


function array_geom = build_hotrod_array_geometry(back_rod_r, rod_r, rod_num, center_factor)
% 构建6组等腰三角形排列的热棒阵列几何结构
% 每组阵列包含：
%   - peak_xy: 热棒中心（峰值）坐标 [N×2]
%   - valley_xy: 热棒间中点（谷值）坐标 [M×2]
%   - peak_roi_radius_mm / valley_roi_radius_mm: ROI采样圆盘半径
    array_num = numel(rod_r);
    array_geom = repmat(struct( ...
        "peak_xy", zeros(0, 2), ...
        "valley_xy", zeros(0, 2), ...
        "rod_radius_mm", 0, ...
        "rod_count", 0, ...
        "peak_roi_radius_mm", 0, ...
        "valley_roi_radius_mm", 0), 1, array_num);

    for i = 1 : array_num
        % 每组阵列的中心位置（均匀分布在圆周上）
        theta_tmp = (i - 1) * pi / 3 + pi / 6;
        x_tmp = back_rod_r * center_factor(i) * cos(theta_tmp);
        y_tmp = back_rod_r * center_factor(i) * sin(theta_tmp);
        rod_r_tmp = rod_r(i);
        rod_num_tmp = rod_num(i);
        d_tmp = rod_r_tmp * 4;  % 热棒间距 = 4倍半径

        % 构建等腰三角形排列的热棒位置
        [x_local, y_local] = build_isosceles_triangle_rods(rod_num_tmp, d_tmp, mod(i, 2) == 0);
        peak_xy = zeros(numel(x_local), 2);
        for j = 1 : numel(x_local)
            [x_global, y_global] = map_local_hotrod_point(x_local(j) + x_tmp, y_local(j) + y_tmp);
            peak_xy(j, :) = [x_global, y_global];
        end

        % 计算相邻热棒的中点作为谷值采样位置
        valley_xy = build_valley_midpoints(x_local, y_local, d_tmp, x_tmp, y_tmp);
        array_geom(i).peak_xy = peak_xy;
        array_geom(i).valley_xy = valley_xy;
        array_geom(i).rod_radius_mm = rod_r_tmp;
        array_geom(i).rod_count = rod_num_tmp;
        % ROI采样半径设为热棒半径的一半，减少像素对齐误差
        array_geom(i).peak_roi_radius_mm = 0.5 * rod_r_tmp;
        array_geom(i).valley_roi_radius_mm = 0.5 * rod_r_tmp;
    end
end


function valley_xy = build_valley_midpoints(x_local, y_local, d_tmp, x_center, y_center)
% 计算相邻热棒的中点（谷值采样位置）
    pair_count = 0;
    valley_xy = zeros(0, 2);
    tol = max(1e-6, d_tmp * 1e-3);

    for i = 1 : numel(x_local)
        for j = (i + 1) : numel(x_local)
            dist = hypot(x_local(i) - x_local(j), y_local(i) - y_local(j));
            if abs(dist - d_tmp) <= tol
                pair_count = pair_count + 1;
                x_mid = (x_local(i) + x_local(j)) / 2 + x_center;
                y_mid = (y_local(i) + y_local(j)) / 2 + y_center;
                [x_global, y_global] = map_local_hotrod_point(x_mid, y_mid);
                valley_xy(pair_count, :) = [x_global, y_global]; %#ok<AGROW>
            end
        end
    end

    if isempty(valley_xy)
        error("Failed to build valley midpoints for one hot-rod array.");
    end

    valley_xy = unique(round(valley_xy, 6), "rows");
end


function [x_global, y_global] = map_local_hotrod_point(x_in, y_in)
% 将局部坐标映射到全局坐标（旋转变换）
    r_tmp = hypot(x_in, y_in);
    theta_tmp_1 = -atan2(y_in, x_in) + pi / 2;
    x_global = r_tmp * cos(theta_tmp_1);
    y_global = r_tmp * sin(theta_tmp_1);
end


function values = sample_disk_means(img_plane, x_grid, y_grid, centers_xy, radius_mm)
% 在指定中心位置采样圆盘ROI的均值
% 使用双线性插值在圆盘内均匀采样点，然后取平均
    values = zeros(size(centers_xy, 1), 1);
    sample_step_mm = max(0.5, radius_mm / 3);

    for idx = 1 : size(centers_xy, 1)
        [x_offsets, y_offsets] = meshgrid(-radius_mm:sample_step_mm:radius_mm, -radius_mm:sample_step_mm:radius_mm);
        disk_mask = (x_offsets .^ 2 + y_offsets .^ 2) <= radius_mm ^ 2;
        x_query = centers_xy(idx, 1) + x_offsets(disk_mask);
        y_query = centers_xy(idx, 2) + y_offsets(disk_mask);
        samples = interp2(y_grid, x_grid, img_plane, y_query, x_query, "linear", 0);
        values(idx) = mean(samples);
    end
end


function [array_geom_out, best_pose] = align_hotrod_geometry(array_geom_in, img_plane, x_grid, y_grid)
% 自动对齐模体几何结构（搜索最佳旋转角度和翻转组合）
% 搜索空间：0°~355°（步长5°）× 4种翻转组合 = 288种姿态
    rotation_candidates = 0:5:355;
    flip_candidates = [
        0, 0
        1, 0
        0, 1
        1, 1
    ];

    best_score = -inf;
    best_pose = struct("rotation_deg", 0, "flip_x", 0, "flip_y", 0, "score", -inf);
    array_geom_out = array_geom_in;

    for flip_idx = 1 : size(flip_candidates, 1)
        flip_x = logical(flip_candidates(flip_idx, 1));
        flip_y = logical(flip_candidates(flip_idx, 2));
        for rotation_deg = rotation_candidates
            array_geom_tmp = apply_pose_to_array_geom(array_geom_in, rotation_deg, flip_x, flip_y);
            score = score_peak_alignment(array_geom_tmp, img_plane, x_grid, y_grid);
            if score > best_score
                best_score = score;
                array_geom_out = array_geom_tmp;
                best_pose.rotation_deg = rotation_deg;
                best_pose.flip_x = double(flip_x);
                best_pose.flip_y = double(flip_y);
                best_pose.score = score;
            end
        end
    end
end


function score = score_peak_alignment(array_geom, img_plane, x_grid, y_grid)
% 评估对齐质量：峰值位置图像值的均值越大越好
    score_sum = 0;
    count_sum = 0;
    for idx = 1 : numel(array_geom)
        peak_values = sample_disk_means( ...
            img_plane, ...
            x_grid, ...
            y_grid, ...
            array_geom(idx).peak_xy, ...
            array_geom(idx).peak_roi_radius_mm);
        score_sum = score_sum + sum(peak_values);
        count_sum = count_sum + numel(peak_values);
    end
    score = score_sum / max(count_sum, 1);
end


function array_geom_out = apply_pose_to_array_geom(array_geom_in, rotation_deg, flip_x, flip_y)
% 对阵列几何结构施加旋转和翻转变换
    array_geom_out = array_geom_in;
    for idx = 1 : numel(array_geom_in)
        array_geom_out(idx).peak_xy = transform_points(array_geom_in(idx).peak_xy, rotation_deg, flip_x, flip_y);
        array_geom_out(idx).valley_xy = transform_points(array_geom_in(idx).valley_xy, rotation_deg, flip_x, flip_y);
    end
end


function points_out = transform_points(points_in, rotation_deg, flip_x, flip_y)
% 对点集施加翻转和旋转变换
    points_out = points_in;
    if flip_x
        points_out(:, 1) = -points_out(:, 1);
    end
    if flip_y
        points_out(:, 2) = -points_out(:, 2);
    end

    theta = deg2rad(rotation_deg);
    rot_mat = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    points_out = (rot_mat * points_out.').';
end


function [x_rod, y_rod] = build_isosceles_triangle_rods(rod_num_tmp, d_tmp, flip_y)
% 构建等腰三角形排列的热棒位置
% 等腰三角形：第1行1个，第2行2个，...，第R行R个
% 总数 N = R*(R+1)/2，所以 R = (sqrt(8N+1)-1)/2
    row_num = (sqrt(8 * rod_num_tmp + 1) - 1) / 2;
    if abs(row_num - round(row_num)) > 1e-8
        error("rod_num=%d cannot form a 1+2+...+R isosceles triangle array.", rod_num_tmp);
    end

    row_num = round(row_num);
    vertical_step = sqrt(3) / 2 * d_tmp;
    x_rod = [];
    y_rod = [];

    for row = 1 : row_num
        rod_count_in_row = row;
        y_row = (2 * (row_num - 1) / 3 - (row - 1)) * vertical_step;
        x_positions = (-(rod_count_in_row - 1) / 2 : 1 : (rod_count_in_row - 1) / 2) * d_tmp;

        x_rod = [x_rod, x_positions];
        y_rod = [y_rod, y_row * ones(1, rod_count_in_row)];
    end

    if flip_y
        y_rod = -y_rod;
    end
end


function labels = build_array_labels(array_geom)
% 构建图例标签（阵列编号、直径、数量）
    labels = strings(1, numel(array_geom));
    for idx = 1 : numel(array_geom)
        labels(idx) = sprintf("Array %d: D=%g mm, N=%d", ...
            idx, 2 * array_geom(idx).rod_radius_mm, array_geom(idx).rod_count);
    end
end


function style_axes(ax, axisFontSize, titleFontSize)
% 统一设置坐标轴样式
    ax.Box = "on";
    ax.LineWidth = 1.5;
    ax.FontSize = axisFontSize;
    ax.TitleFontSizeMultiplier = 1.0;
    ax.Title.FontSize = titleFontSize;
    ax.XLabel.FontSize = axisFontSize;
    ax.YLabel.FontSize = axisFontSize;
end


function palette = build_polar_palette(numColors)
% 构建颜色映射表（暖色调渐变）
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