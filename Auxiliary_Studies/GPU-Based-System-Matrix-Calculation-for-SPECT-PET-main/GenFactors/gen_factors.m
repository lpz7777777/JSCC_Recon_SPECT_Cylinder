function gen_factors(energy_keV, sysmat_file, params_detector_file, outdir, crystal_matrix_file)
% GEN_FACTORS 把 GPU 引擎生成的 .sysmat 转成重建用的 Factors/ 目录。
%
% 流程：
%   1. 读 .sysmat (11520 晶体 × 52020 体素)
%   2. 用 Params_Detector 的 flag==1 过滤出闪烁晶体行 → SysMat_tmp
%   3. 笛卡尔→极坐标转换 (1280 点/层，匹配现有 511keV 配置)
%   4. 生成 coor_polar, RotMat/RotMatInv, Detector.csv, SysMat_polar
%
% 输入：
%   energy_keV          - 能量（用于命名/日志）
%   sysmat_file         - 输入 .sysmat 路径（如 SysMat_withScatter_*.sysmat）
%   params_detector_file- Params_Detector.dat 路径（用于 flag 过滤 + Detector.csv）
%   outdir              - 输出目录（如 Factors/218keV_RotateNum20/）
%   crystal_matrix_file - CrystalMatrix_*.mat 路径（备用，当前用 Params_Detector 的 flag）

    fprintf('=====================================\n');
    fprintf('gen_factors: %d keV\n', energy_keV);
    fprintf('  输入矩阵: %s\n', sysmat_file);
    fprintf('  输出目录: %s\n', outdir);
    fprintf('=====================================\n');

    if ~exist(outdir, 'dir'); mkdir(outdir); end

    %% ---- 几何参数（与现有 511keV Factors 一致）----
    s_x_axis = -150:6:150;          % 51
    s_y_axis = -150:6:150;          % 51
    s_z_axis = -28.5:3:28.5;        % 20
    r_value = 6:6:150;              % 25 个半径
    theta_num_value = 20:20:80;     % 半径依赖的角度数（分 4 段，每段 6 半径）
    rotate_num = 20;                % 旋转数

    %% ---- 读 Params_Detector，确定闪烁晶体过滤掩码 ----
    fid = fopen(params_detector_file, 'r');
    d = fread(fid, 'float32');
    fclose(fid);
    n_det = d(1);
    det = reshape(d(2:end), 12, n_det).';
    scin_mask = det(:, 12) == 1;    % flag==1 为闪烁体
    n_scin = sum(scin_mask);
    fprintf('  总晶体=%d, 闪烁体(flag=1)=%d\n', n_det, n_scin);

    %% ---- 读 .sysmat 并过滤闪烁体行 ----
    % .sysmat 布局：[numProjection × numImagebin × numRotation] = [11520 × 52020 × 1]
    % reshape 成 [numImagebinX, numImagebinY, numImagebinZ, numProjection]
    fid = fopen(sysmat_file, 'r');
    SysMat = fread(fid, 'float32');
    fclose(fid);
    fprintf('  原始矩阵元素数=%d\n', numel(SysMat));
    SysMat = reshape(SysMat, length(s_x_axis), length(s_y_axis), length(s_z_axis), []);
    % 清洗残余 NaN/Inf（保险：新版引擎已修复，但旧矩阵或边界情况可能残留）
    n_bad = sum(~isfinite(SysMat(:)));
    if n_bad > 0
        fprintf('  清洗 %d 个 NaN/Inf (%.2f%%) -> 0\n', n_bad, 100*n_bad/numel(SysMat));
        SysMat(~isfinite(SysMat)) = 0;
    end
    % 过滤：只保留闪烁晶体（第4维）
    SysMat = SysMat(:, :, :, scin_mask);
    fprintf('  过滤后维度: [%d,%d,%d,%d]\n', size(SysMat));

    % 存 SysMat_tmp（过滤后版本）
    tmp_file = fullfile(outdir, 'SysMat_tmp');
    fid = fopen(tmp_file, 'w');
    fwrite(fid, SysMat, 'float32');
    fclose(fid);
    fprintf('  SysMat_tmp 已写入\n');

    %% ---- 极坐标网格（1280 点/层）----
    [coor_cartesian_x, coor_cartesian_y] = meshgrid(s_x_axis, s_y_axis);
    coor_polar = [];
    theta_per_r = zeros(1, length(r_value));
    for id_r = 1:length(r_value)
        r = r_value(id_r);
        theta_num = theta_num_value(ceil(id_r / (length(r_value)/length(theta_num_value))));
        theta_per_r(id_r) = theta_num;
        for id_theta = 1:theta_num
            theta = (id_theta - 1) * 360 / theta_num;
            x = r * cosd(theta);
            y = r * sind(theta);
            coor_polar = cat(1, coor_polar, [x, y]);
        end
    end
    n_polar_per_layer = size(coor_polar, 1);
    fprintf('  极坐标点数/层=%d, 总点数=%d\n', n_polar_per_layer, n_polar_per_layer*length(s_z_axis));

    % 存 coor_polar + coor_polar_full
    save(fullfile(outdir, 'coor_polar.mat'), 'coor_polar');
    writematrix(coor_polar, fullfile(outdir, 'coor_polar.csv'));
    coor_polar_full = [];
    for id_z = 1:length(s_z_axis)
        z = s_z_axis(id_z);
        coor_polar_full = cat(1, coor_polar_full, cat(2, coor_polar, z*ones(n_polar_per_layer,1)));
    end
    save(fullfile(outdir, 'coor_polar_full.mat'), 'coor_polar_full');
    writematrix(coor_polar_full, fullfile(outdir, 'coor_polar_full.csv'));

    %% ---- 旋转矩阵 RotMat / RotMatInv ----
    % 对每个半径，圆周移位间隔 = theta_num / rotate_num
    RotMat = [];
    for id_rotate = 1:rotate_num
        RotMat_tmp = [];
        for id_r = 1:length(r_value)
            theta_num = theta_per_r(id_r);
            interval = theta_num / rotate_num;
            RotMat_tmp_r = zeros(1, theta_num);
            for id_theta = 1:theta_num
                RotMat_tmp_r(id_theta) = mod((id_rotate-1)*interval + id_theta - 1, theta_num) + 1;
            end
            RotMat_tmp_r = RotMat_tmp_r + length(RotMat_tmp);
            RotMat_tmp = cat(2, RotMat_tmp, RotMat_tmp_r);
        end
        RotMat = cat(1, RotMat, RotMat_tmp);
    end
    RotMat = RotMat.';
    % 求逆
    RotMatInv = [];
    for i = 1:rotate_num
        [~, inv_idx] = sort(RotMat(:, i));
        RotMatInv = cat(2, RotMatInv, inv_idx);
    end
    save(fullfile(outdir, 'RotMat.mat'), 'RotMat');
    save(fullfile(outdir, 'RotMatInv.mat'), 'RotMatInv');
    writematrix(RotMat, fullfile(outdir, 'RotMat.csv'));
    writematrix(RotMatInv, fullfile(outdir, 'RotMatInv.csv'));
    % _full（按 z 层偏移）
    RotMat_full = []; RotMatInv_full = [];
    for id_z = 1:length(s_z_axis)
        RotMat_full = cat(1, RotMat_full, RotMat + size(RotMat_full, 1));
        RotMatInv_full = cat(1, RotMatInv_full, RotMatInv + size(RotMatInv_full, 1));
    end
    save(fullfile(outdir, 'RotMat_full.mat'), 'RotMat_full');
    save(fullfile(outdir, 'RotMatInv_full.mat'), 'RotMatInv_full');
    writematrix(RotMat_full, fullfile(outdir, 'RotMat_full.csv'));
    writematrix(RotMatInv_full, fullfile(outdir, 'RotMatInv_full.csv'));
    fprintf('  RotMat/RotMatInv 已生成 (%d×%d)\n', size(RotMat,1), size(RotMat,2));

    %% ---- 笛卡尔→极坐标转换 SysMat_polar ----
    % SysMat_polar[polar, z, crystal] <- interp2(SysMat[x,y,z,crystal])
    n_z = length(s_z_axis);
    n_crystal = size(SysMat, 4);
    SysMat_polar = zeros(n_polar_per_layer, n_z, n_crystal, 'single');
    for id_z = 1:n_z
        for id_crystal = 1:n_crystal
            layer = squeeze(SysMat(:, :, id_z, id_crystal));
            SysMat_polar(:, id_z, id_crystal) = interp2(coor_cartesian_x, coor_cartesian_y, ...
                                                         layer.', coor_polar(:,1), coor_polar(:,2), 'linear');
        end
    end
    % 排列成 [crystal, polar, z]
    SysMat_polar = permute(SysMat_polar, [3, 1, 2]);
    fid = fopen(fullfile(outdir, 'SysMat_polar'), 'w');
    fwrite(fid, SysMat_polar, 'float32');
    fclose(fid);
    fprintf('  SysMat_polar 已写入 (%d×%d×%d)\n', size(SysMat_polar));

    %% ---- Detector.csv（仅 index, x, y, z）----
    det_scin = det(scin_mask, :);
    Detector = [(1:n_scin)', det_scin(:, 1:3)];
    % 补列名行
    header = 'index,x,y,z';
    fid = fopen(fullfile(outdir, 'Detector.csv'), 'w');
    fprintf(fid, '%s\n', header);
    fclose(fid);
    writematrix(Detector, fullfile(outdir, 'Detector.csv'), 'WriteMode', 'append');
    fprintf('  Detector.csv 已写入 (%d 晶体)\n', n_scin);

    fprintf('  完成: %s\n', outdir);
end
