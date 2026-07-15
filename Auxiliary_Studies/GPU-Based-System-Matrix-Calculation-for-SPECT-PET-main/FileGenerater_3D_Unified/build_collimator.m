function col_params = build_collimator(cfg, col_coeff, energy_keV)
% BUILD_COLLIMATOR 构建准直器参数，按 cfg.geometry_type 分派。
%
% 输入：
%   cfg          - config_geometry() 返回的配置结构体
%   col_coeff    - [mu_total, mu_PE, mu_Compton]，准直器材料（Pb/W）的衰减系数
%   energy_keV   - 当前能量
% 输出：
%   col_params   - float32 行向量，写 Params_Collimator.dat 的完整内容
%                  布局：[100字段头, (cx,y1,y2,cz,radius,mu_t,mu_pe,mu_co,flag) × numHoles]
%
% 与原 generateCollimator.m / generateCollimator_conventionalSPECT.m 的关键区别：
%   1. JSCC 版补全衰减系数（原版 collimator_params(16..18)=0）
%   2. 准直器层头部存板材系数；实际孔道是空气/真空，孔记录的系数保持为 0

    switch cfg.geometry_type
        case 'JSCC'
            col_params = build_collimator_jscc(cfg, col_coeff, energy_keV);
        case 'ConventionalSPECT'
            col_params = build_collimator_conv(cfg, col_coeff, energy_keV);
        otherwise
            error('build_collimator: unknown geometry_type "%s".', cfg.geometry_type);
    end
end


% --------------------------------------------------------------------------- %
%  JSCC 平面随机孔准直器（Poisson-disk 采样，从 randomPoints.mat 加载）         %
% --------------------------------------------------------------------------- %
function col_params = build_collimator_jscc(cfg, col_coeff, ~)
    c = cfg.jscc.collimator;
    y_center = c.y_center;
    t = c.thickness;

    if c.num_points == 0
        % 无孔模式：完整板（真空时等效无准直器）
        hole_params = zeros(0, 9);
    else
        % 从 randomPoints.mat 加载孔中心（500 个点，[x,y] 坐标）
        rp_file = 'randomPoints.mat';
        if ~isempty(dir(rp_file))
            loaded = load(rp_file);
            randomPoints = loaded.randomPoints;
        else
            error('build_collimator: randomPoints.mat 未找到。');
        end
        % 组装每孔 9 字段记录
        n = size(randomPoints, 1);
        hole_params = zeros(n, 9);
        hole_params(:, 1) = randomPoints(:, 2);
        hole_params(:, 2) = y_center - t/2;
        hole_params(:, 3) = y_center + t/2;
        hole_params(:, 4) = randomPoints(:, 1);
        hole_params(:, 5) = c.radius;
        hole_params(:, 6:8) = 0;  % Air/vacuum inside the aperture
        hole_params(:, 9) = 0;
    end

    n = size(hole_params, 1);

    % 100 字段头
    header = zeros(1, 100);
    header(1) = 1;                  % numLayers
    header(11) = n;                 % numHoles
    header(12) = c.width;
    header(13) = t;
    header(14) = c.height;
    header(15) = 0;
    header(16) = col_coeff(1);      % mu_total（补全：原版这里为 0）
    header(17) = col_coeff(2);      % mu_PE
    header(18) = col_coeff(3);      % mu_Compton

    col_params = single([header, reshape(hole_params.', 1, [])]);
end


% --------------------------------------------------------------------------- %
%  传统 Siemens Symbia EHE 三角晶格平行孔准直器                                 %
% --------------------------------------------------------------------------- %
function col_params = build_collimator_conv(cfg, col_coeff, ~)
    c = cfg.conv.collimator;
    col_t = cfg.conv.collimator_thickness;

    num_x = c.hole_rows;          % 25（z 方向）
    num_y = c.hole_cols;          % 50（x 方向）
    radius = c.hole_diameter / 2;
    pitch = c.hole_diameter + c.septal_thickness;
    column_pitch = sqrt(3)/2 * pitch;    % 三角晶格列间距
    row_pitch = pitch;                    % 行间距

    % 生成三角晶格孔中心
    points = [];
    for id_x = 1:num_x
        col_offset = 0;
        if mod(id_x, 2) == 0
            col_offset = row_pitch / 2;
        end
        for id_y = 1:num_y
            px = (id_x - 1/2 - num_x/2) * column_pitch;
            py = (id_y - 1/2 - num_y/2) * row_pitch + col_offset;
            points = cat(1, points, [px, py]);
        end
    end
    % 中心化
    points(:, 1) = points(:, 1) - mean(points(:, 1));
    points(:, 2) = points(:, 2) - mean(points(:, 2));

    % 组装每孔 9 字段记录
    n = size(points, 1);
    hole_params = zeros(n, 9);
    hole_params(:, 1) = points(:, 2);                  % cx（原代码用 randomPoints(:,2)）
    hole_params(:, 2) = -col_t/2;                      % y1
    hole_params(:, 3) = col_t/2;                       % y2
    hole_params(:, 4) = points(:, 1);                  % cz（原代码用 randomPoints(:,1)）
    hole_params(:, 5) = radius;
    hole_params(:, 6:8) = 0;                           % 孔内为空气/真空，不是 Pb
    hole_params(:, 9) = 0;

    % 100 字段头
    header = zeros(1, 100);
    header(1) = 1;
    header(11) = n;
    header(12) = c.width;
    header(13) = col_t;
    header(14) = c.height;
    header(15) = 0;
    header(16) = col_coeff(1);
    header(17) = col_coeff(2);
    header(18) = col_coeff(3);

    col_params = single([header, reshape(hole_params.', 1, [])]);
end
