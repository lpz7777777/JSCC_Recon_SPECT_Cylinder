function write_dat_files(outdir, cfg, energy_keV, det_params, col_params)
% WRITE_DAT_FILES 把 4 个 Params_*.dat 写入指定目录。
%
% 输入：
%   outdir       - 输出目录（已存在）
%   cfg          - config_geometry() 返回的配置结构体
%   energy_keV   - 当前能量（用于 Physics 的 target_pe_energy）
%   det_params   - build_detector 返回的 float32 行向量
%   col_params   - build_collimator 返回的 float32 行向量
%
% 同时构建 Params_Image.dat（来自 cfg.fov）和 Params_Physics.dat（来自 cfg 散射开关）。

    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    %% Params_Detector.dat
    write_bin(fullfile(outdir, 'Params_Detector.dat'), det_params);

    %% Params_Collimator.dat
    write_bin(fullfile(outdir, 'Params_Collimator.dat'), col_params);

    %% Params_Image.dat（12 字段头）
    f = cfg.fov;
    img = zeros(1, 12);
    img(1)  = length(f.x_axis);
    img(2)  = length(f.y_axis);
    img(3)  = length(f.z_axis);
    img(4)  = f.x_axis(2) - f.x_axis(1);
    img(5)  = f.y_axis(2) - f.y_axis(1);
    img(6)  = f.z_axis(2) - f.z_axis(1);
    img(7)  = f.num_rotation;
    img(8)  = f.angle_per_rotation;
    img(9)  = f.shift_x;
    img(10) = f.shift_y;
    img(11) = f.shift_z;
    img(12) = f.fov2collimator0;
    write_bin(fullfile(outdir, 'Params_Image.dat'), single(img));

    %% Params_Physics.dat（12 字段，控制散射/能量窗/探测器局部散射）
    % MATLAB 1-based 物理含义如下（写入后 C++ 读为 0-based）：
    %   physics(1) -> C++ Physics[0] flagUsingCompton
    %   physics(2) -> C++ Physics[1] flagSavingPESysmat        (始终 1)
    %   physics(3) -> C++ Physics[2] flagSavingComptonSysmat
    %   physics(4) -> C++ Physics[3] flagSavingPEComptonSysmat
    %   physics(5) -> C++ Physics[4] flagUsingSameEnergyWindow
    %   physics(6) -> C++ Physics[5] lowerThresholdEnergyWindow (keV)
    %   physics(7) -> C++ Physics[6] upperThresholdEnergyWindow (keV)
    %   physics(8) -> C++ Physics[7] targetPEEnergy             (keV)
    %   physics(9) -> C++ Physics[8] flagCalculateCrystalGeometryRelationship
    %   physics(10)-> C++ Physics[9] flagCalculateCollimatorGeometryRelationship
    %   physics(11)-> C++ Physics[10] flagDetectorRecoilEscapeResponse
    %   physics(12)-> C++ Physics[11] flagSelfComptonPhotoelectricResponse
    phy = zeros(1, 12);
    phy(1)  = bool_to_float(cfg.enable_compton);
    phy(2)  = 1;                                              % PE 系统矩阵始终保存
    phy(3)  = bool_to_float(cfg.save_compton_only);
    phy(4)  = bool_to_float(cfg.save_combined_sysmat);
    phy(5)  = bool_to_float(cfg.use_same_energy_window);
    phy(6)  = cfg.energy_window_lower_keV;
    phy(7)  = cfg.energy_window_upper_keV;
    phy(8)  = energy_keV;                                     % 目标 PE 能量
    phy(9)  = bool_to_float(cfg.compute_geo_relationship);
    phy(10) = bool_to_float(cfg.compute_geo_relationship);
    phy(11) = bool_to_float(cfg.enable_detector_recoil_escape);
    phy(12) = bool_to_float(cfg.enable_self_scatter_photopeak);
    write_bin(fullfile(outdir, 'Params_Physics.dat'), single(phy));
end


% --------------------------------------------------------------------------- %
%  辅助：写 float32 二进制文件                                                  %
% --------------------------------------------------------------------------- %
function write_bin(filepath, data)
    fid = fopen(filepath, 'wb');
    if fid < 0
        error('write_dat_files: 无法写入 %s', filepath);
    end
    fwrite(fid, data, 'float32');
    fclose(fid);
end


function v = bool_to_float(b)
    if b
        v = 1;
    else
        v = 0;
    end
end
