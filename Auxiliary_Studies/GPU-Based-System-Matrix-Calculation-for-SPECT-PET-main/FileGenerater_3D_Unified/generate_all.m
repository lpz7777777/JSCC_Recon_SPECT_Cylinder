function generate_all()
% GENERATE_ALL 主入口：按配置为每个能量生成一套 Params_*.dat。
%
% 用法：
%   1. 编辑 config_geometry.m 设置几何类型、能量列表、材料、散射开关等
%   2. 在本目录下运行：generate_all
%   3. 输出到 output/<geometry_type>_<E>keV/，每个子目录含完整 4 个 Params_*.dat
%   4. 将子目录内容拷到 CUDA 引擎（PEGen/ScatterGen）的工作目录即可运行
%
% 示例（在 MATLAB 命令行）：
%   >> cd FileGenerater_3D_Unified
%   >> generate_all

    cfg = config_geometry();
    if strcmp(cfg.geometry_type, 'ConventionalSPECT')
        cfg.fov.fov2collimator0 = cfg.conv.fov2collimator0;
    end
    this_dir = fileparts(mfilename('fullpath'));
    output_root = fullfile(this_dir, cfg.output_root);
    if ~exist(output_root, 'dir')
        mkdir(output_root);
    end

    fprintf('=====================================\n');
    fprintf('FileGenerater_3D_Unified\n');
    fprintf('几何类型: %s\n', cfg.geometry_type);
    fprintf('能量列表: %s keV\n', mat2str(cfg.energy_list_keV));
    fprintf('探测器材料: %s\n', cfg.detector_material);
    fprintf('准直器材料: %s\n', cfg.collimator_material);
    fprintf('散射开关: %d\n', cfg.enable_compton);
    fprintf('输出根目录: %s\n', output_root);
    fprintf('=====================================\n\n');

    n_energy = length(cfg.energy_list_keV);

    for i = 1:n_energy
        e = cfg.energy_list_keV(i);
        fprintf('--- 能量 %d/%d: %d keV ---\n', i, n_energy, e);

        % 按基准能量分辨率 + 1/√E 律计算当前能量的分辨率
        energy_res = cfg.energy_resolution_ref * sqrt(cfg.energy_resolution_ref_keV / e);
        fprintf('  能量分辨率 @%dkeV = %.4f (FWHM，由 %.0fkeV@%.2f 按 1/√E 标度)\n', ...
                e, energy_res, cfg.energy_resolution_ref_keV, cfg.energy_resolution_ref);

        % 查材料衰减系数
        det_coeff = material_db(cfg.detector_material, e);
        % 高 Z 屏蔽体材料（CrystalMatrix 标签>1）：独立于准直器，避免 Vacuum 准直器时屏蔽体也被置零
        if isfield(cfg, 'shield_material')
            highz_coeff = material_db(cfg.shield_material, e);
        else
            highz_coeff = material_db(cfg.collimator_material, e);  % 向后兼容
        end
        col_coeff = material_db(cfg.collimator_material, e);
        coeffs = struct('scintillator', det_coeff, 'highz', highz_coeff);

        fprintf('  探测器(闪烁体) [mu_t, mu_pe, mu_co] = [%.4f, %.4f, %.4f]\n', det_coeff);
        fprintf('  屏蔽体(高Z)    [mu_t, mu_pe, mu_co] = [%.4f, %.4f, %.4f] (%s)\n', ...
                highz_coeff, cfg.shield_material);
        fprintf('  准直器         [mu_t, mu_pe, mu_co] = [%.4f, %.4f, %.4f] (%s)\n', ...
                col_coeff, cfg.collimator_material);

        % 构建探测器与准直器（传入按能量算好的分辨率）
        det_params = build_detector(cfg, coeffs, e, energy_res);
        col_params = build_collimator(cfg, col_coeff, e);
        fprintf('  探测器晶体数: %d\n', det_params(1));
        fprintf('  准直器孔数: %d\n', col_params(11));

        % 写入分目录
        outdir = fullfile(output_root, sprintf('%s_%dkeV', cfg.geometry_type, e));
        write_dat_files(outdir, cfg, e, det_params, col_params);
        fprintf('  已写入: %s\n', outdir);

        % 绘制 3D 几何示意图
        plot_geometry_3d(cfg, det_params, col_params, e, ...
                         fullfile(outdir, sprintf('geometry_3d_%s_%dkeV', cfg.geometry_type, e)));
        fprintf('\n');
    end

    fprintf('全部完成。输出目录: %s\n', output_root);
end
