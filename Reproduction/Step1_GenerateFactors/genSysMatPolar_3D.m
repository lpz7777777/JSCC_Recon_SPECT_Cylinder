%% genSysMatPolar_3D.m
% =========================================================================
% 功能：从 SysMat.sysmat 直接生成极坐标系统矩阵及旋转映射矩阵
%       整合了原来 generateDet_3D.m 中读取和筛选系统矩阵的步骤
% =========================================================================
%
% 输入文件（需在 Data/ 目录中）：
%   - SysMat.sysmat   Monte Carlo 模拟生成的原始系统矩阵（二进制float32）
%                     形状为 [51, 51, 20, total_bins]（所有CrystalMatrix中!=0的bin）
%   - CrystalMatrix_20250307_JSCCGC_32x64x4.mat
%                     晶体矩阵定义，用于筛选 CrystalMatrix==1 的 bin
%
% 输出文件（在当前目录下）：
%   - SysMat_polar     极坐标系统矩阵（二进制float32）
%   - coor_polar.csv   单层极坐标像素坐标
%   - coor_polar_full.csv  完整极坐标像素坐标（含Z轴）
%   - RotMat.csv / RotMat_full.csv       旋转映射矩阵
%   - RotMatInv.csv / RotMatInv_full.csv  旋转逆映射矩阵
%
% 极坐标参数说明：
%   - 径向范围：3mm 到 150mm，步长 3mm（50个环）
%   - 角度分辨率：每环 40~160 个角度采样点，随半径增大而增多
%   - 旋转角度数：20
%   - Z轴：20层，-28.5mm 到 28.5mm，步长 3mm

clear; clc; close all;

%% ==================== 参数设置 ====================

% 直角坐标图像网格参数（与CUDA系统矩阵计算时一致）
s_x_axis = -150 : 6 : 150;      % X方向：-150mm 到 150mm，步长6mm（51个点）
s_y_axis = -150 : 6 : 150;      % Y方向：同上（51个点）
s_z_axis = - 28.5 : 3 : 28.5;   % Z方向：-28.5mm 到 28.5mm，步长3mm（20层）

% 极坐标参数
r_value = 3 : 3 : 150;           % 径向：3mm 到 150mm，步长3mm（50个环）
theta_num_value = 40 : 40 : 160; % 每环角度数：内环40，外环160，线性递增

% 备选参数（更粗的极坐标网格）
% r_value = 6 : 6 : 150;
% theta_num_value = 20 : 20 : 80;

rotate_num = 20;  % 旋转角度数（SPECT采集的投影角度数）

%% ==================== 第1步：读取原始系统矩阵 ====================
% 从 SysMat.sysmat 读取 Monte Carlo 模拟生成的直角坐标系统矩阵
% 然后筛选 CrystalMatrix==1 的晶体单元对应的列

fprintf('正在读取 SysMat.sysmat ...\n');

% 加载晶体矩阵定义，用于筛选有效bin
load('Data/CrystalMatrix_20250307_JSCCGC_32x64x4.mat');

% 读取原始系统矩阵文件（二进制float32格式）
fid = fopen("Data/SysMat.sysmat", "r");
SysMat_raw = fread(fid, "float32");
fclose(fid);

% reshape为4D矩阵：[X方向51, Y方向51, Z方向20, 所有非零晶体bin数]
% 注意：SysMat.sysmat 包含 CrystalMatrix 中所有 !=0 的 bin
%       （即值=1的前3层NaI + 值=2的后1层CdZnTe）
SysMat_raw = reshape(SysMat_raw, 51, 51, 20, []);

% 仅保留 CrystalMatrix==1 的 bin（前3层NaI晶体，用于标准SPECT重建）
% 展平CrystalMatrix以便索引
CrystalMatrix_flat = reshape(CrystalMatrix, [], 1);
% 只取前51*51*20-1对应的层（与generateDet_3D.m中的逻辑一致）
CrystalMatrix_flat = CrystalMatrix_flat(1:size(SysMat_raw, 4));
SysMat_cartesian = SysMat_raw(:, :, :, CrystalMatrix_flat == 1);

fprintf('原始系统矩阵bin数: %d\n', size(SysMat_raw, 4));
fprintf('筛选后(CrystalMatrix==1)bin数: %d\n', size(SysMat_cartesian, 4));
fprintf('直角坐标系统矩阵尺寸: [%d, %d, %d, %d]\n', size(SysMat_cartesian));

% 释放原始矩阵内存
clear SysMat_raw CrystalMatrix_flat;

%% 获取直角坐标
[coor_cartesian_x, coor_cartesian_y] = meshgrid(s_x_axis, s_y_axis);

%% 将直角坐标转换为极坐标
coor_polar = [];

for id_r = 1 : length(r_value)
    r = r_value(id_r);
    theta_num = theta_num_value(ceil(id_r / (length(r_value)/length(theta_num_value))));

    for id_theta = 1 : theta_num
        theta = (id_theta - 1) * 360 / theta_num;
        x = r * cosd(theta);
        y = r * sind(theta);

        coor_polar = cat(1, coor_polar, [x, y]);
    end
end

save("coor_polar.mat", "coor_polar");
writematrix(coor_polar, "coor_polar.csv");

coor_polar_full = [];
for id_z = 1 : length(s_z_axis)
    z = s_z_axis(id_z);
    coor_polar_full_tmp = cat(2, coor_polar, z * ones(size(coor_polar, 1), 1));
    coor_polar_full = cat(1, coor_polar_full, coor_polar_full_tmp);
end
save("coor_polar_full.mat", "coor_polar_full");
writematrix(coor_polar_full, "coor_polar_full.csv");

%% 生成旋转矩阵
RotMat = [];

for id_rotate = 1 : rotate_num
    RotMat_tmp = [];

    for id_r = 1 : length(r_value)
        r = r_value(id_r);
        theta_num = theta_num_value(ceil(id_r / (length(r_value)/length(theta_num_value))));
        interval = theta_num / rotate_num;
        RotMat_tmp_r = zeros(1, theta_num);
    
        for id_theta = 1 : theta_num
            RotMat_tmp_r(id_theta) = mod((id_rotate - 1) * interval + id_theta - 1, theta_num) + 1;
        end
        
        RotMat_tmp_r = RotMat_tmp_r + length(RotMat_tmp);
        RotMat_tmp = cat(2, RotMat_tmp, RotMat_tmp_r);
    end

    RotMat = cat(1, RotMat, RotMat_tmp);
end

RotMat = RotMat.';

inverse_index = [];
for i = 1 : rotate_num
    [x, inverse_index_tmp] = sort(RotMat(:, i));
    inverse_index = cat(2, inverse_index, inverse_index_tmp);
end
RotMatInv = inverse_index;

save("RotMat.mat", "RotMat");
save("RotMatInv.mat", "RotMatInv");
writematrix(RotMat, "RotMat.csv");
writematrix(RotMatInv, "RotMatInv.csv");

RotMat_full = [];
RotMatInv_full = [];
for id_z = 1 : length(s_z_axis)
    RotMat_tmp = RotMat + size(RotMat_full, 1);
    RotMatInv_tmp = RotMatInv + size(RotMatInv_full, 1);
    RotMat_full = cat(1, RotMat_full, RotMat_tmp);
    RotMatInv_full = cat(1, RotMatInv_full, RotMatInv_tmp);
end
save("RotMat_full.mat", "RotMat_full");
save("RotMatInv_full.mat", "RotMatInv_full");
writematrix(RotMat_full, "RotMat_full.csv");
writematrix(RotMatInv_full, "RotMatInv_full.csv");

%% 系统矩阵生成
SysMat_polar = zeros(length(coor_polar), size(SysMat_cartesian, 3), size(SysMat_cartesian, 4));

for id_z = 1 : size(SysMat_cartesian, 3)
    for id_crystal = 1 : size(SysMat_cartesian, 4)
        SysMat_cartesian_tmp = SysMat_cartesian(:, :, id_z, id_crystal);
        SysMat_polar_tmp = interp2(coor_cartesian_x, coor_cartesian_y, SysMat_cartesian_tmp.', coor_polar(:, 1), coor_polar(:, 2), 'linear');
        SysMat_polar(:, id_z, id_crystal) = SysMat_polar_tmp;
    end
end

SysMat_polar = permute(SysMat_polar, [3, 1, 2]);
fid = fopen("SysMat_polar", "w");
fwrite(fid, SysMat_polar, "float32");
fclose(fid);