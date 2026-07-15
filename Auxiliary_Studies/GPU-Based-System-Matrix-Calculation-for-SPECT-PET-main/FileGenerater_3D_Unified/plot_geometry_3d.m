function plot_geometry_3d(cfg, det_params, col_params, energy_keV, save_path)
% PLOT_GEOMETRY_3D 通过 Python+Plotly 绘制交互式 3D 几何示意图。
%
% 工作流程：
%   1. 解析 det_params / col_params / cfg 得到几何数据
%   2. 保存为临时 .mat 文件
%   3. system() 调用 plot_geometry_3d.py（直接读 .mat），输出交互式 HTML
%
% 输入：
%   cfg          - config_geometry() 配置结构体
%   det_params   - build_detector 返回的 float32 行向量
%   col_params   - build_collimator 返回的 float32 行向量
%   energy_keV   - 当前能量
%   save_path    - 输出路径（不含扩展名）；生成 <save_path>.html

    % ----------------------- 解析探测器 -----------------------
    % det 字段：[x,y,z, wDet(X), tDet(Y朝FOV), hDet(Z), mu_t,mu_pe,mu_co, eres, roty, flag]
    n_det = det_params(1);
    det = reshape(det_params(2:end), 12, n_det).';
    det_x  = double(det(:, 1));  det_y = double(det(:, 2));  det_z = double(det(:, 3));
    det_dx = double(det(:, 4));  det_dy = double(det(:, 5)); det_dz = double(det(:, 6));
    det_flag = double(det(:, 12));    % CrystalMatrix 标签（1=闪烁体, >1=屏蔽体）

    % ----------------------- 解析准直器 -----------------------
    num_holes = col_params(11);
    col_w  = double(col_params(12));
    col_t  = double(col_params(13));
    col_h  = double(col_params(14));
    % 准直器板 Y 范围：从头信息推导（中心 y_center，厚度 col_t）
    % 头 [15]=0 留空，板 Y 范围 = y_center ± t/2
    % 无孔时从头算；有孔时从第1孔的 y1/y2 取（等价）
    if num_holes > 0 && numel(col_params) >= 103
        hole_y1 = double(col_params(102));
        hole_y2 = double(col_params(103));
        hole = reshape(col_params(101:end), 9, num_holes).';
        hole_cx = double(hole(:, 1));
        hole_cz = double(hole(:, 4));
        hole_r  = double(hole(:, 5));
    else
        % 无孔：从 config 的 y_center 和 thickness 推导
        col_y_center = 1.5;  % cfg.jscc.collimator.y_center
        hole_y1 = col_y_center - col_t/2;
        hole_y2 = col_y_center + col_t/2;
        hole_cx = [];  hole_cz = [];  hole_r = [];
    end

    % ----------------------- 解析 FOV -----------------------
    f = cfg.fov;
    nx = length(f.x_axis);  ny = length(f.y_axis);  nz = length(f.z_axis);
    wx = f.x_axis(2)-f.x_axis(1);  wy = f.y_axis(2)-f.y_axis(1);  wz = f.z_axis(2)-f.z_axis(1);
    fov_xr = [f.shift_x - nx*wx/2, f.shift_x + nx*wx/2];
    fov_yr = [f.shift_y - ny*wy/2, f.shift_y + ny*wy/2];
    fov_zr = [f.shift_z - nz*wz/2, f.shift_z + nz*wz/2];

    % ----------------------- 统一到 FOV 原点的绝对坐标系 -----------------------
    % CUDA 引擎约定：FOV 中心在原点 (0,0,0)；探测器/准直器的 Y 在 .dat 里是
    % 相对准直器第 0 层的局部坐标，引擎运行时对它们做 +fov2collimator0 平移。
    % 可视化要显示真实相对位置，故同样平移探测器与准直器的 Y。
    fov2col = f.fov2collimator0;
    det_y = det_y + fov2col;
    hole_y1 = hole_y1 + fov2col;
    hole_y2 = hole_y2 + fov2col;

    % ----------------------- 保存临时 .mat -----------------------
    this_dir = fileparts(mfilename('fullpath'));
    tmp_mat = fullfile(this_dir, '_geometry_data.mat');

    geometry_data = struct( ...
        'det_x', det_x, 'det_y', det_y, 'det_z', det_z, ...
        'det_dx', det_dx, 'det_dy', det_dy, 'det_dz', det_dz, ...
        'det_flag', det_flag, ...
        'hole_cx', hole_cx, 'hole_cy1', hole_y1, 'hole_cy2', hole_y2, ...
        'hole_cz', hole_cz, 'hole_r', hole_r, ...
        'col_w', col_w, 'col_h', col_h, ...
        'fov_xr', fov_xr, 'fov_yr', fov_yr, 'fov_zr', fov_zr);
    save(tmp_mat, '-struct', 'geometry_data');

    % ----------------------- 调用 Python -----------------------
    title_str = sprintf('%s @ %dkeV (探测器%d晶体, 准直器%d孔)', ...
                        cfg.geometry_type, energy_keV, n_det, num_holes);
    html_path = [save_path '.html'];
    py_cmd = locate_python();
    py_script = fullfile(this_dir, 'plot_geometry_3d.py');

    % 用空格分隔参数；title 含空格/中文，用双引号包裹
    cmd = sprintf('"%s" "%s" "%s" "%s" "%s"', ...
                  py_cmd, py_script, tmp_mat, html_path, title_str);
    status = system(cmd);
    if status ~= 0
        warning('plot_geometry_3d: Python 调用失败 (status=%d)。请确认已安装 Python+Plotly+SciPy。', status);
        fprintf('  命令: %s\n', cmd);
    end

    % 清理临时 .mat
    if exist(tmp_mat, 'file')
        delete(tmp_mat);
    end
end


% ----------------------------------------------------------------------- %
%  定位 Python 可执行文件                                                    %
% ----------------------------------------------------------------------- %
function py = locate_python()
    persistent cached
    if ~isempty(cached)
        py = cached;
        return;
    end
    candidates = {'python', 'python3', 'py'};
    for i = 1:length(candidates)
        [status, ~] = system(['"' candidates{i} '" --version 2>&1']);
        if status == 0
            cached = candidates{i};
            py = cached;
            return;
        end
    end
    cached = 'python';
    py = cached;
end
