function run_gen_factors()
% RUN_GEN_FACTORS 为 218/440keV 的合并矩阵生成 Factors 目录。
%
% 输入：runs/<E>keV/SysMat_withScatter_*.sysmat + 对应 Params_Detector.dat
% 输出：Factors/<E>keV_RotateNum20/

    % 从当前工作目录推断引擎根目录（本脚本位于 GenFactors/ 下）
    this_dir = pwd;
    if endsWith(this_dir, 'GenFactors') || endsWith(this_dir, 'GenFactors\') || endsWith(this_dir, 'GenFactors/')
        repo_root = fileparts(this_dir);
    else
        % 兜底：从 mfilename 推断
        mf = fileparts(mfilename('fullpath'));
        if ~isempty(mf)
            repo_root = fileparts(mf);
        else
            repo_root = fileparts(this_dir);
        end
    end
    fg_dir = fullfile(repo_root, 'FileGenerater_3D_Unified');
    runs_root = fullfile(repo_root, 'runs');
    factors_root = fullfile(repo_root, 'Factors');

    energies = [440];
    sysmat_name = 'SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat';

    for i = 1:length(energies)
        e = energies(i);
        e_tag = sprintf('%dkeV', e);

        sysmat_file = fullfile(runs_root, sprintf('JSCC_%s', e_tag), sysmat_name);
        params_det = fullfile(fg_dir, 'output', sprintf('JSCC_%s', e_tag), 'Params_Detector.dat');
        outdir = fullfile(factors_root, sprintf('%s_RotateNum20', e_tag));

        if ~exist(sysmat_file, 'file')
            error('合并矩阵未找到: %s', sysmat_file);
        end
        if ~exist(params_det, 'file')
            error('Params_Detector.dat 未找到: %s', params_det);
        end

        gen_factors(e, sysmat_file, params_det, outdir, '');
        fprintf('\n');
    end
    fprintf('全部完成。Factors 输出到: %s\n', factors_root);
end
