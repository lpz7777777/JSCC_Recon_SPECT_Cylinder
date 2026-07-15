function det_params = build_detector(cfg, coeffs, energy_keV, energy_res)
% BUILD_DETECTOR 构建探测器参数，按 cfg.geometry_type 分派。
%
% 输入：
%   cfg          - config_geometry() 返回的配置结构体
%   coeffs       - 结构体，含 .scintillator（标签1，闪烁体）和 .highz（标签2，高Z）
%                  各为 [mu_total, mu_PE, mu_Compton]
%   energy_keV   - 当前能量
%   energy_res   - 当前能量处的 FWHM 能量分辨率（分数，已按 1/√E 标度好）
%                  填入 Detector[id*12+10]（散射引擎必需）
% 输出：
%   det_params   - float32 行向量，写 Params_Detector.dat 的完整内容
%                  布局：[count, (x,y,z,sx,sy,sz,mu_t,mu_pe,mu_co,eres,roty,flag) × count]

    switch cfg.geometry_type
        case 'JSCC'
            det_params = build_detector_jscc(cfg, coeffs, energy_keV, energy_res);
        case 'ConventionalSPECT'
            det_params = build_detector_conv(cfg, coeffs.scintillator, energy_keV, energy_res);
        otherwise
            error('build_detector: unknown geometry_type "%s".', cfg.geometry_type);
    end
end


% --------------------------------------------------------------------------- %
%  JSCC 多层 depth-of-interaction 探测器                                       %
% --------------------------------------------------------------------------- %
function det_params = build_detector_jscc(cfg, coeffs, ~, energy_res)
    det_coeff = coeffs.scintillator;
    highz_coeff = coeffs.highz;

    cm_file = cfg.jscc.crystal_matrix_file;
    if ~isempty(dir(cm_file))     % 存在性检查（dir 返回 struct 或空）
        loaded = load(cm_file);
        CrystalMatrix = loaded.CrystalMatrix;
    else
        error('build_detector: CrystalMatrix 文件未找到: %s', cm_file);
    end

    unit_num_x = size(CrystalMatrix, 1);   % 32
    unit_num_y = size(CrystalMatrix, 2);   % 64
    unit_num_z = size(CrystalMatrix, 3);   % 31

    pitch_x = cfg.jscc.pitch(1);   % 4.2（X 方向间距）
    pitch_y = cfg.jscc.pitch(2);   % 4.2（Y 方向层间距）
    pitch_z = cfg.jscc.pitch(3);   % 3（Z 方向间距）

    % --- 前 3 层（Y = base, base+pitch_z, base+2*pitch_z）---
    % 注意：原 generateDet_3D.m:68 有越界 bug（第二维写成 unit_num_x），
    % 这里用正确的 (unit_num_x, unit_num_y, unit_num_z-1) 动态尺寸。
    nlayer_front = unit_num_z - 1;
    pos1 = zeros(unit_num_x, unit_num_y, nlayer_front, 3);
    for id_x = 1:unit_num_x
        for id_y = 1:unit_num_y
            for id_z = 1:nlayer_front
                % 轴置换与原代码一致：id_y->X, id_z->Y(深度), id_x->Z
                pos1(id_x, id_y, id_z, 1) = pitch_y * (id_y - unit_num_y/2 - 1/2);
                pos1(id_x, id_y, id_z, 2) = pitch_z * (id_z - 1) + cfg.jscc.layer_y_base;
                pos1(id_x, id_y, id_z, 3) = pitch_x * (id_x - unit_num_x/2 - 1/2);
            end
        end
    end
    pos1 = reshape(pos1, [], 3);

    % 用 CrystalMatrix 非零掩码筛选（前 nlayer_front 层）
    cm_front = reshape(CrystalMatrix, [], 1);
    cm_front = cm_front(1:size(pos1, 1));
    tf = (cm_front ~= 0);
    pos1 = pos1(tf, :);
    cm_front = cm_front(tf);

    % --- 第 4 层（细分层，Y = back_layer_y）---
    % 尺寸 2×6×2，半间距 back_layer_half_pitch，晶体数 2*unit_num_x × 2*unit_num_y
    % 轴置换与前 3 层一致：id_y->X, id_x->Z
    nx_back = 2 * unit_num_x;     % Z 方向晶体数（由 id_x 驱动）
    ny_back = 2 * unit_num_y;     % X 方向晶体数（由 id_y 驱动）
    hp = cfg.jscc.back_layer_half_pitch;
    pos2 = zeros(nx_back, ny_back, 3);
    for id_x = 1:nx_back
        for id_y = 1:ny_back
            pos2(id_x, id_y, 1) = hp * (id_y - ny_back/2 - 1/2);   % X（修正：原代码误用 nx_back）
            pos2(id_x, id_y, 2) = cfg.jscc.back_layer_y;            % Y
            pos2(id_x, id_y, 3) = hp * (id_x - nx_back/2 - 1/2);   % Z（修正：原代码误用 ny_back）
        end
    end
    pos2 = reshape(pos2, [], 3);
    cm_back = ones(size(pos2, 1), 1);    % 第 4 层全为闪烁体（标签 1）

    % --- 合并 ---
    pos = cat(1, pos1, pos2);
    cm_all = cat(1, cm_front, cm_back);

    n_front = size(pos1, 1);

    % --- 组装 12 字段/晶体记录 ---
    % CUDA 引擎约定（PESysMatGen.cu:533-535）：det[+4]=wDet(X), det[+5]=tDet(Y朝FOV), det[+6]=hDet(Z)
    % config 的 unit_size / back_layer_size = [X, Y_朝FOV, Z]，故 1:1 赋值
    det = zeros(size(pos, 1), 12);
    det(:, 1:3) = pos;
    det(1:n_front, 4) = cfg.jscc.unit_size(1);            % X 方向宽度
    det(1:n_front, 5) = cfg.jscc.unit_size(2);            % Y 方向厚度（朝 FOV）
    det(1:n_front, 6) = cfg.jscc.unit_size(3);            % Z 方向高度
    det(n_front+1:end, 4) = cfg.jscc.back_layer_size(1);  % 第 4 层 X
    det(n_front+1:end, 5) = cfg.jscc.back_layer_size(2);  % 第 4 层 Y（朝 FOV，6mm）
    det(n_front+1:end, 6) = cfg.jscc.back_layer_size(3);  % 第 4 层 Z

    % 衰减系数按 CrystalMatrix 标签查表
    % 标签 1 -> 闪烁体 det_coeff；标签 2 -> 高 Z 材料 highz_coeff（如 Pb/W 屏蔽层）
    coeff_lookup = cat(1, det_coeff, highz_coeff);
    label_idx = max(1, min(size(coeff_lookup, 1), round(cm_all)));
    det(:, 7) = coeff_lookup(label_idx, 1);
    det(:, 8) = coeff_lookup(label_idx, 2);
    det(:, 9) = coeff_lookup(label_idx, 3);

    det(:, 10) = energy_res;                  % 能量分辨率（已按 1/√E 标度到当前能量）
    det(:, 11) = 0;                           % Y 轴旋转角
    det(:, 12) = cm_all;                      % flag 存 CrystalMatrix 材料标签（1=闪烁体, >1=屏蔽体）

    det_params = single([size(pos, 1), reshape(det.', 1, [])]);
end


% --------------------------------------------------------------------------- %
%  传统平行孔 SPECT 探测器（34×68×1 NaI 平面阵列）                              %
% --------------------------------------------------------------------------- %
function det_params = build_detector_conv(cfg, det_coeff, ~, energy_res)
    nx = cfg.conv.unit_num(1);   % 34
    ny = cfg.conv.unit_num(2);   % 68
    nz = cfg.conv.unit_num(3);   % 1
    % unit_size = [X, Y_朝FOV, Z]，pitch = [px_X, py_Y, pz_Z]
    size_x = cfg.conv.unit_size(1);   % 4（横向）
    size_y = cfg.conv.unit_size(2);   % 10（朝 FOV 厚度）
    size_z = cfg.conv.unit_size(3);   % 4（轴向）
    pitch_x = cfg.conv.pitch(1); % 4
    pitch_y = cfg.conv.pitch(2); % 4
    pitch_z = cfg.conv.pitch(3); % 3

    collimator_t = cfg.conv.collimator_thickness;
    detector_center_y = size_y/2 + collimator_t/2 + cfg.conv.detector_gap_y;

    pos = zeros(nx, ny, nz, 3);
    for id_x = 1:nx
        for id_y = 1:ny
            for id_z = 1:nz
                pos(id_x, id_y, id_z, 1) = pitch_y * (id_y - ny/2 - 1/2);
                pos(id_x, id_y, id_z, 2) = pitch_z * (id_z - 1) + detector_center_y;
                pos(id_x, id_y, id_z, 3) = pitch_x * (id_x - nx/2 - 1/2);
            end
        end
    end
    pos = reshape(pos, [], 3);

    n = size(pos, 1);
    det = zeros(n, 12);
    det(:, 1:3) = pos;
    det(:, 4) = cfg.conv.unit_size(1);   % wDet = X 方向
    det(:, 5) = cfg.conv.unit_size(2);   % tDet = Y 方向（朝 FOV）
    det(:, 6) = cfg.conv.unit_size(3);   % hDet = Z 方向
    det(:, 7) = det_coeff(1);
    det(:, 8) = det_coeff(2);
    det(:, 9) = det_coeff(3);
    det(:, 10) = energy_res;
    det(:, 11) = 0;
    det(:, 12) = 1;    % flag = 闪烁体（传统 SPECT 全是 NaI，无屏蔽体）

    det_params = single([n, reshape(det.', 1, [])]);
end
