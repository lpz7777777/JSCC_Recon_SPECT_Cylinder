%% ContrastPhantom_DualEnergy_Rotate_3D
% 生成 225Ac 双能量(218+440 keV) Contrast Phantom 的 Geant4 GPS macro。
%
% 物理设定：
%   - 背景圆柱同时发射 218keV(221Fr) 和 440keV(213Bi)，按 225Ac 衰变链
%     长期平衡下的 gamma 产额比 11.4% : 26.1% 加权
%   - 6 个热圆柱交替为 218(Fr分布) 和 440(Bi分布)，隔一个相同能量，
%     体现 alpha 衰变后子体脱离螯合药物造成的空间分布偏移
%   - x_Fr 和 x_Bi 分别按空间积分归一化，再乘 Y218 和 Y440。因此整个 run
%     的期望初级光子数严格按 11.4% : 26.1% 分配，分支比只施加一次。
%
% 6 圆柱布局（60°间隔，从 0°起）：
%   rod 1 (  0°): 218 keV (Fr)
%   rod 2 ( 60°): 440 keV (Bi)
%   rod 3 (120°): 218 keV (Fr)
%   rod 4 (180°): 440 keV (Bi)
%   rod 5 (240°): 218 keV (Fr)
%   rod 6 (300°): 440 keV (Bi)

clear;

%% ---- 参数 ----
back_rod_d = 240;              % 背景圆柱直径 mm
rod_d = 10:4:30;              % 6 个热圆柱直径 mm
height = 30;                  % 圆柱高度 mm

x_center = 0;
y_center = -245;             % 背景圆柱中心 Y（探测器在 +Y 方向）
z = 0;

act = 6;                     % 热圆柱相对活度基准
LengthUnit = 'mm';
Theta_Min = 0;
Theta_Max = 180;
AngleUnit = 'deg';

% 双能量参数（225Ac 衰变链长期平衡）
ene_218 = 218;               % keV (221Fr)
ene_440 = 440;               % keV (213Bi)
yield_218 = 0.114;           % 218keV gamma 产额 (221Fr, 11.4%)
yield_440 = 0.261;           % 440keV gamma 产额 (213Bi, 26.1%)
yield_ratio_440_to_218 = yield_440 / yield_218;

rotate_num = 20;
total_events = 1e9;          % 全部旋转视角合计的初级 218+440 光子数
events_per_view = floor(total_events / rotate_num);
event_remainder = mod(total_events, rotate_num);

% 6 个热圆柱的能量分配（交替 218/440）
rod_ene = [ene_218, ene_440, ene_218, ene_440, ene_218, ene_440];
rod_label = {'Fr-218', 'Bi-440', 'Fr-218', 'Bi-440', 'Fr-218', 'Bi-440'};

% GPS source intensity 是每个体积 source 的总权重。先计算两张空间分布
% 各自的积分，再缩放 440 的全部 source，使整个 run 满足 Y440/Y218。
rod_base_weight = (act - 1) * rod_d.^2 / back_rod_d^2;
activity_weight_218 = 1 + sum(rod_base_weight(rod_ene == ene_218));
activity_weight_440 = 1 + sum(rod_base_weight(rod_ene == ene_440));
source_scale_218 = 1;
source_scale_440 = yield_ratio_440_to_218 * ...
    activity_weight_218 / activity_weight_440;

script_path = fileparts(mfilename("fullpath"));
save_path = fullfile(script_path, "Macro", ...
    "ContrastPhantom_DualEnergy_10_30_240_30_225Ac");
if ~exist(save_path, "dir")
    mkdir(save_path);
end

% 避免 rotate_num 从 60 改为 20 后，旧的 21.mac--60.mac 被误当成有效视角。
old_mac_files = dir(fullfile(save_path, "*.mac"));
for file_idx = 1:numel(old_mac_files)
    [~, stem] = fileparts(old_mac_files(file_idx).name);
    if ~isempty(regexp(stem, "^\d+$", "once"))
        delete(fullfile(old_mac_files(file_idx).folder, old_mac_files(file_idx).name));
    end
end

%% ---- 生成 macro ----
for id_rotate = 1 : rotate_num
    phi = (id_rotate - 1) * 2 * pi / rotate_num;
    macro_path = fullfile(save_path, sprintf("%d.mac", id_rotate));
    fid = fopen(macro_path, "w");
    if fid < 0
        error("Cannot create Geant4 macro: %s", macro_path);
    end
    events_this_view = events_per_view + double(id_rotate <= event_remainder);

    fprintf(fid, '# Contrast Phantom: 225Ac dual-energy (218+440 keV)\n');
    fprintf(fid, '# Rotate %d/%d, phi=%.1f deg\n', id_rotate, rotate_num, phi*180/pi);
    fprintf(fid, '# Whole-run expected yield ratio: 218:440 = 0.114:0.261\n');
    fprintf(fid, '# Rods: alternating Fr-218 / Bi-440 to model daughter redistribution\n');
    fprintf(fid, '# Fr/Bi spatial maps are normalized separately before yield weighting\n');
    fprintf(fid, '# beamOn counts selected primary photons, not 225Ac decays\n\n');

    % ===== 背景圆柱 source 0: 218 keV (Fr) =====
    % 权重 = yield_218（相对产额）
    fprintf(fid, '/gps/particle gamma\n');
    fprintf(fid, '/gps/energy %d keV\n', ene_218);
    fprintf(fid, '/gps/pos/type Volume\n');
    fprintf(fid, '/gps/pos/shape Cylinder\n');
    fprintf(fid, '/gps/pos/radius %.4f %s\n', back_rod_d/2, LengthUnit);
    fprintf(fid, '/gps/pos/halfz %.4f %s\n', height/2, LengthUnit);
    fprintf(fid, '/gps/pos/centre %.4f %.4f %.4f %s\n', x_center, y_center, z, LengthUnit);
    fprintf(fid, '/gps/ang/type iso\n');
    fprintf(fid, '/gps/ang/mintheta %.4f %s\n', Theta_Min, AngleUnit);
    fprintf(fid, '/gps/ang/maxtheta %.4f %s\n', Theta_Max, AngleUnit);
    fprintf(fid, '\n#\n');

    % ===== 背景圆柱 source 1: 440 keV (Bi)，权重 = yield_440 =====
    fprintf(fid, '/gps/source/add %.12f\n', source_scale_440);
    fprintf(fid, '/gps/particle gamma\n');
    fprintf(fid, '/gps/energy %d keV\n', ene_440);
    fprintf(fid, '/gps/pos/type Volume\n');
    fprintf(fid, '/gps/pos/shape Cylinder\n');
    fprintf(fid, '/gps/pos/radius %.4f %s\n', back_rod_d/2, LengthUnit);
    fprintf(fid, '/gps/pos/halfz %.4f %s\n', height/2, LengthUnit);
    fprintf(fid, '/gps/pos/centre %.4f %.4f %.4f %s\n', x_center, y_center, z, LengthUnit);
    fprintf(fid, '/gps/ang/type iso\n');
    fprintf(fid, '/gps/ang/mintheta %.4f %s\n', Theta_Min, AngleUnit);
    fprintf(fid, '/gps/ang/maxtheta %.4f %s\n', Theta_Max, AngleUnit);
    fprintf(fid, '\n#\n');

    % ===== 6 个热圆柱 =====
    for i = 1 : 6
        theta_tmp = (i-1) * pi/3 - phi;
        x_tmp = back_rod_d/4 * cos(theta_tmp) + x_center;
        y_tmp = back_rod_d/4 * sin(theta_tmp) + y_center;
        rod_d_tmp = rod_d(i);
        act_tmp = (act-1) * rod_d_tmp^2 / back_rod_d^2;
        if rod_ene(i) == ene_440
            source_weight = act_tmp * source_scale_440;
        else
            source_weight = act_tmp * source_scale_218;
        end

        fprintf(fid, '# rod %d: %s at (%.1f, %.1f), d=%.0fmm, relative weight=%.12f\n', ...
                i, rod_label{i}, x_tmp, y_tmp, rod_d_tmp, source_weight);
        fprintf(fid, '/gps/source/add %.12f\n', source_weight);
        fprintf(fid, '/gps/particle gamma\n');
        fprintf(fid, '/gps/energy %d keV\n', rod_ene(i));
        fprintf(fid, '/gps/pos/type Volume\n');
        fprintf(fid, '/gps/pos/shape Cylinder\n');
        fprintf(fid, '/gps/ang/type iso\n');
        fprintf(fid, '/gps/ang/mintheta %.4f %s\n', Theta_Min, AngleUnit);
        fprintf(fid, '/gps/ang/maxtheta %.4f %s\n', Theta_Max, AngleUnit);
        fprintf(fid, '/gps/pos/centre %.4f %.4f %.4f %s\n', x_tmp, y_tmp, z, LengthUnit);
        fprintf(fid, '/gps/pos/radius %.4f %s\n', rod_d_tmp/2, LengthUnit);
        fprintf(fid, '/gps/pos/halfz %.4f %s\n', height/2, LengthUnit);
        fprintf(fid, '\n#\n');
    end

    fprintf(fid, '/run/beamOn %d\n', events_this_view);
    fclose(fid);
end

weight_218 = source_scale_218 * activity_weight_218;
weight_440 = source_scale_440 * activity_weight_440;
if abs(weight_440 / weight_218 - yield_ratio_440_to_218) > 1e-12
    error("Whole-run GPS energy ratio does not match Y440/Y218.");
end

fprintf("已生成 %d 个 macro 文件到 %s\n", rotate_num, save_path);
fprintf("总初级光子数: %.0f，每个视角: %d（余数分配到前 %d 个视角）\n", ...
    total_events, events_per_view, event_remainder);
fprintf("全 run 期望比例: Y218=%.3f, Y440=%.3f, Y440/Y218=%.6f\n", ...
    yield_218, yield_440, yield_ratio_440_to_218);
fprintf("热圆柱: rod1/3/5=218keV(Fr), rod2/4/6=440keV(Bi)，各自相对本能量背景为 %.1f 倍\n", act);
fprintf("GPS 期望事件份额: 218=%.4f%%, 440=%.4f%%\n", ...
    100 * weight_218 / (weight_218 + weight_440), ...
    100 * weight_440 / (weight_218 + weight_440));

%% ---- 可视化（调用 Python+Plotly 生成 HTML）----
first_mac = fullfile(save_path, "1.mac");
html_out = fullfile(save_path, "phantom_3d.html");
visualizer = fullfile(script_path, "visualize_phantom.py");
cmd = sprintf('python "%s" "%s" "%s"', visualizer, first_mac, html_out);
status = system(cmd);
if status ~= 0
    fprintf('可视化生成失败（需要 Python+Plotly）。命令: %s\n', cmd);
else
    fprintf('可视化已保存: %s\n', html_out);
end
