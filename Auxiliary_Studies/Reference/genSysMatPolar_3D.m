s_x_axis = -150 : 6 : 150;
s_y_axis = -150 : 6 : 150;
s_z_axis = - 28.5 : 3 : 28.5;

r_value = 6 : 6 : 150;
theta_num_value = 40 : 40 : 80;

rotate_num = 40;

%% 读取系统矩阵
fid = fopen("SysMat_tmp", "r");
SysMat_cartesian = fread(fid, "float32");
fclose(fid); 
SysMat_cartesian = reshape(SysMat_cartesian, length(s_x_axis), length(s_y_axis), length(s_z_axis), []);

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