function cfg = config_geometry()
% CONFIG_GEOMETRY 返回几何 + 物理 + 能量的统一配置结构体。
%
% 用户通过修改本函数内的字段来切换几何类型、能量、材料、散射开关等。
% generate_all.m 会读取这个配置，为 energy_list_keV 里的每个能量生成一套
% Params_*.dat 文件到 output/<geometry_type>_<E>keV/ 目录。
%
% 支持两种几何：
%   'JSCC'               - 多层 depth-of-interaction 平面探测器 + 平面随机孔准直器
%                          （32×64×4 布局，靠 CrystalMatrix 掩码选层）
%   'ConventionalSPECT'  - 传统 Siemens Symbia EHE 平行孔 SPECT（34×68×1 NaI）

    %% ---- 几何类型（主开关）----
    cfg.geometry_type = 'JSCC';        % 'JSCC' 或 'ConventionalSPECT'

    %% ---- 能量与材料 ----
    % energy_list_keV 里的每个能量都会生成一套独立的 Params_*.dat
    cfg.energy_list_keV = [218, 440];           % 批量生成的能量列表
    cfg.detector_material = 'GAGG';              % 'NaI' 或 'GAGG'（仅 ConventionalSPECT 用 NaI；
                                                %   JSCC 按 CrystalMatrix 标签 1 选 NaI/GAGG）
    cfg.collimator_material = 'Vacuum';             % 'Pb' 或 'W' 'Vacuum'
    cfg.shield_material = 'W';                      % 屏蔽体材料（CrystalMatrix 标签>1），独立于准直器
                                                %   一般是 W 或 Pb；不要用 Vacuum（否则屏蔽体无衰减）
    cfg.energy_resolution_ref = 0.13;           % 探测器能量分辨率基准值（FWHM 分数）
    cfg.energy_resolution_ref_keV = 511;        % 基准能量；其他能量按 R ∝ 1/√E 自动标度
                                                %   generate_all 会为每个能量算 res(E) = ref × √(ref_E / E)
                                                %   并填入 Detector[id*12+10]（散射引擎必需）

    %% ---- 散射 / 物理开关（控制 Params_Physics.dat）----
    % 与新版 GPU-Based-System-Matrix-Calculation 引擎的 Params_Physics 一一对应
    cfg.enable_compton = true;                  % Physics[0]  Compton 散射开关
    cfg.save_compton_only = true;               % Physics[2]  保存散射系统矩阵
    cfg.save_combined_sysmat = true;            % Physics[3]  保存 PE+Compton 合并矩阵
    cfg.use_same_energy_window = false;         % Physics[4]  false=按探测器分辨率自动推能窗（围绕光电峰 ±res/2）
    cfg.energy_window_lower_keV = 0;            % Physics[5]  能量窗下限（仅 use_same_energy_window=true 时生效）
    cfg.energy_window_upper_keV = 0;            % Physics[6]  能量窗上限
    cfg.compute_geo_relationship = true;        % Physics[8][9] 首次跑散射时让引擎计算几何关系位图
    cfg.enable_detector_recoil_escape = true;   % Physics[10] A中Compton后光子逃逸，记录A的反冲沉积
    cfg.enable_self_scatter_photopeak = true;   % Physics[11] A中Compton后在A内光电吸收，记录全能峰

    %% ---- FOV / 体素网格（两套几何共用）----
    cfg.fov.x_axis = -150:6:150;                % 51 体素，6mm
    cfg.fov.y_axis = -150:6:150;                % 51 体素，6mm
    cfg.fov.z_axis = -28.5:3:28.5;              % 20 层，3mm
    cfg.fov.num_rotation = 1;                   % 引擎只算 1 视角，靠极坐标后处理 RotMat 合成 60 视角
    cfg.fov.angle_per_rotation = 0;             % 弧度（num_rotation=1 时无意义）
    cfg.fov.shift_x = 0;                        % mm
    cfg.fov.shift_y = 0;                        % mm
    cfg.fov.shift_z = 0;                        % mm
    % CUDA adds this value to detector/collimator local Y coordinates.  For
    % JSCC it places the first detector layer at the established position.
    cfg.fov.fov2collimator0 = 170;              % mm, JSCC local-Y origin in the global FOV frame

    %% ---- JSCC 多层几何参数（geometry_type='JSCC' 时用）----
    cfg.jscc.crystal_matrix_file = 'CrystalMatrix_20250307_JSCCGC_32x64x4.mat';
    cfg.jscc.unit_size = [3, 3, 3];             % 前 3 层晶体尺寸 [X, Y_朝FOV, Z] mm（3×3×3 立方体）
    cfg.jscc.pitch = [4.2, 4.2, 3];             % 晶体间距 [px_X, py_Y层间, pz_Z] mm
    cfg.jscc.layer_y_base = 30;                 % 第 1 层 Y 深度，逐层 +pitch_z
    cfg.jscc.back_layer_y = 120;                % 第 4 层 Y 深度（细分层）
    cfg.jscc.back_layer_half_pitch = 2.1;       % 第 4 层细分半间距 mm（X/Z 平面内）
    cfg.jscc.back_layer_size = [2, 6, 2];       % 第 4 层晶体尺寸 [X, Y_朝FOV, Z] mm（2×6×2，6mm 垂直于层面）
    % Common physical reference requested for geometry comparisons: the EHE
    % collimator front face must coincide with this JSCC detector front face.
    cfg.fov.common_front_face_y = cfg.fov.fov2collimator0 ...
        + cfg.jscc.layer_y_base - cfg.jscc.unit_size(2) / 2;
    % JSCC 准直器：真空板无孔（等效无准直器，光子自由穿过）
    cfg.jscc.collimator.num_points = 0;         % 0=无孔（完整板）
    cfg.jscc.collimator.radius = 3;             % mm（num_points=0 时无效）
    cfg.jscc.collimator.min_distance = 6;       % mm（num_points=0 时无效）
    cfg.jscc.collimator.plane_size = 200;       % mm（num_points=0 时无效）
    cfg.jscc.collimator.boundary_distance = 3;  % mm（num_points=0 时无效）
    cfg.jscc.collimator.y_center = 1.5;         % mm（准直器板中心 Y）
    cfg.jscc.collimator.thickness = 3;          % mm
    cfg.jscc.collimator.width = 500;            % mm
    cfg.jscc.collimator.height = 500;           % mm

    %% ---- 传统平行孔 SPECT 几何参数（geometry_type='ConventionalSPECT' 时用）----
    cfg.conv.unit_size = [4, 10, 4];            % NaI 晶体尺寸 [X, Y_朝FOV, Z] mm（截面 4×4，厚 10mm 朝 FOV）
    cfg.conv.pitch = [4, 4, 3];                 % 晶体间距 mm
    cfg.conv.unit_num = [34, 68, 1];            % [nx, ny, nz] 晶体数
    cfg.conv.collimator_thickness = 50.5;       % EHE 准直器厚度 mm
    cfg.conv.detector_gap_y = 0;                % 探测器与准直器背面的间隙 mm
    % 传统准直器：Siemens Symbia EHE 三角晶格平行孔
    cfg.conv.collimator.hole_rows = 25;         % numPoints_x（z 方向孔行数）
    cfg.conv.collimator.hole_cols = 50;         % numPoints_y（x 方向孔列数）
    cfg.conv.collimator.hole_diameter = 2.5;    % mm
    cfg.conv.collimator.septal_thickness = 3.4; % mm
    cfg.conv.collimator.width = 330;            % mm（270+60）
    cfg.conv.collimator.height = 165;           % mm（135+30）
    % EHE local Y=0 is the collimator center, so its CUDA translation must
    % include half the plate thickness to put the front at common_front_face_y.
    cfg.conv.fov2collimator0 = cfg.fov.common_front_face_y ...
        + cfg.conv.collimator_thickness / 2;

    %% ---- 输出 ----
    cfg.output_root = 'output';                 % 相对 generate_all.m 所在目录
end
