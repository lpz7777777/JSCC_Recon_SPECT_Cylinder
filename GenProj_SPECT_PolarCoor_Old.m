%% Factors
pixel_num_cartesian_z = 20;

pixel_num_x = 800;
pixel_num_y = 800;
pixel_num_z = 120;
pixel_l_x = 0.5;
pixel_l_y = 0.5;
pixel_l_z = 0.5;
fov_l_x = pixel_num_x * pixel_l_x;
fov_l_y = pixel_num_y * pixel_l_y;
fov_l_z = pixel_num_z * pixel_l_z;

total_count = 1e11; % Total simulated photons

%% load Mat
% file_path = "./Factors/511keV_RotateNum20/";
file_path = "./Factors/140keV_RotateNum20/";
load(sprintf("%sRotMat.mat", file_path));
load(sprintf("%scoor_polar_full.mat", file_path));

fid = fopen(sprintf("%sSysMat_polar", file_path), "r");
SysMat = fread(fid, "float32");
fclose(fid); 
SysMat = reshape(SysMat, [], size(coor_polar_full, 1));
total_count_singleview = total_count / size(RotMat, 2);

%% GenProj
coor_cartesian = zeros(pixel_num_x, pixel_num_y, pixel_num_z, 3);
for id_z = 1 : pixel_num_z
    coor_cartesian(:, :, id_z, 1) = ((-fov_l_x/2 + pixel_l_x/2) : pixel_l_x : (fov_l_x/2 - pixel_l_x/2)).' .* ones(1, pixel_num_y);
    coor_cartesian(:, :, id_z, 2) = ((-fov_l_y/2 + pixel_l_y/2) : pixel_l_y : (fov_l_y/2 - pixel_l_y/2)) .* ones(pixel_num_x, 1);
    coor_cartesian(:, :, id_z, 3) = (id_z - pixel_num_z/2 - 1/2) * pixel_l_z;
end

coor_cartesian = reshape(coor_cartesian, [], 3);

% back_rod_r = 120;
% rod_r = 8 : 2 : 18;
% rod_h = 30;
% act = 6;
% img_cartesian = ContrastPhantom(coor_cartesian, back_rod_r, rod_r, rod_h, act);

rod_num = [10, 6, 6, 3, 3, 3];
height = 30;
back_rod_r = 200;
rod_r = 5:2:15;
rod_h = 30;
act = 1;
center_factor = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4];

img_cartesian = HotRodPhantom(coor_cartesian, back_rod_r, rod_r, rod_h, act, rod_num, center_factor);


% X_value = -70 : 20 : 70;
% Y_value = -70 : 20 : 70;
% Z_value = -20 : 20 : 20;
% act = 1;
% rod_r = 5;
% img_cartesian = PointArrayPhantom(coor_cartesian, X_value, Y_value, Z_value, rod_r, act);

% back_rod_r = 140;
% rod_r = 15:3:30;
% rod_h = 30;
% act = 0;
% img_cartesian = ContrastPhantom(coor_cartesian, back_rod_r, rod_r, rod_h, act);


% back_rod_r = 150;
% rod_r = 6:2:16;
% rod_num = [6, 6, 6, 3, 3, 3];
% center_factor = 0.6;
% act = 8;
% img_seg = HotRodPhantom(coor_seg, back_rod_r, rod_r, act, rod_num, center_factor);

%% Interp
img_cartesian = reshape(img_cartesian, pixel_num_x, pixel_num_y, pixel_num_z);
[coor_cartesian_x, coor_cartesian_y, coor_cartesian_z] = meshgrid(((-fov_l_x/2 + pixel_l_x/2) : pixel_l_x : (fov_l_x/2 - pixel_l_x/2)), ((-fov_l_y/2 + pixel_l_y/2) : pixel_l_y : (fov_l_y/2 - pixel_l_y/2)),  ((-fov_l_z/2 + pixel_l_z/2) : pixel_l_z : (fov_l_z/2 - pixel_l_z/2)));
img_polar = interp3(coor_cartesian_x, coor_cartesian_y, coor_cartesian_z, img_cartesian, coor_polar_full(:, 1), coor_polar_full(:, 2), coor_polar_full(:, 3), 'linear');

% img_polar_tmp = zeros(1160, 20);
% img_polar_tmp(:, 6:15) = 1;
% img_polar_tmp(:, 6:15) = img_polar .* img_polar_tmp(:, 6:15);
% % img_polar = img_polar .* ones(1, pixel_num_cartesian_z);
% img_polar = img_polar_tmp;
img_polar = reshape(img_polar, [], pixel_num_cartesian_z);
img_polar = img_polar * total_count_singleview / sum(img_polar, "all");

%% Rotate
CntStat = [];
rotate_num = size(RotMat, 2);
for id_rotate = 1 : rotate_num
    RotMat_tmp = RotMat(:, id_rotate);
    img_tmp = reshape(img_polar(RotMat_tmp, :), [], 1);
    CntStat_tmp = (SysMat * img_tmp).';

    for i = 1 : length(CntStat_tmp)
        sigma = sqrt(CntStat_tmp(i));
        tmp_count = normrnd(CntStat_tmp(i), sigma);
        CntStat_tmp(i) = max(0, round(tmp_count));
    end

    CntStat = cat(1, CntStat, CntStat_tmp);
end

writematrix(CntStat, "./CntStat/CntStat.csv");

%% Plot

fid = fopen("img_cartesian", "w");
fwrite(fid, img_cartesian, "float32");
fclose(fid);

% figure;
% color = 1 - gray(256);
% % img_cartesian = flip(img_cartesian, 1);
% imagesc(img_cartesian);
% axis square
% colormap(color);
% title("img");

%%
function img_seg = ContrastPhantom(coor_seg, back_rod_r, rod_r, rod_h, act)
    img_seg = zeros(size(coor_seg, 1), 1);
    
    flag_rod = (coor_seg(:, 1).^2 + coor_seg(:, 2).^2) < back_rod_r^2;
    flag_rod_h = abs(coor_seg(:, 3)) < rod_h/2;
    flag_rod = logical(flag_rod .* flag_rod_h);

    img_seg(flag_rod) = 1;

    for i = 1 : 6
        theta_tmp = -(i-1) * pi/3 + pi/2;
        x_tmp = back_rod_r/2 * cos(theta_tmp);
        y_tmp = back_rod_r/2 * sin(theta_tmp);
        rod_r_tmp = rod_r(i);

        flag_rod = ((coor_seg(:, 1) - x_tmp).^2 + (coor_seg(:, 2) - y_tmp).^2) <= rod_r_tmp^2;
        flag_rod_h = abs(coor_seg(:, 3)) < rod_h/2;
        flag_rod = logical(flag_rod .* flag_rod_h);
        img_seg(flag_rod) = act;
    end
end

function img_seg = HotRodPhantom(coor_seg, back_rod_r, rod_r, rod_h, act, rod_num, center_factor)
    img_seg = zeros(size(coor_seg, 1), 1);

    % flag_rod = (coor_seg(:, 1).^2 + coor_seg(:, 2).^2) < back_rod_r^2;
    % img_seg(flag_rod) = 1;

    for i = 1 : 6
        theta_tmp = (i-1) * pi/3 + pi / 6;
        x_tmp = back_rod_r * center_factor(i) * cos(theta_tmp);
        y_tmp = back_rod_r * center_factor(i) * sin(theta_tmp);
        rod_r_tmp = rod_r(i);
        rod_num_tmp = rod_num(i);
        d_tmp = rod_r_tmp * 4;
    
        if mod(i, 2) == 1
            % ----- 奇数簇 -----
            if rod_num_tmp == 6
                x_rod = [0, -d_tmp/2, d_tmp/2, -d_tmp, 0, d_tmp];
                y_rod = [2*d_tmp/sqrt(3), sqrt(3)/6*d_tmp, sqrt(3)/6*d_tmp, -sqrt(3)/3*d_tmp, -sqrt(3)/3*d_tmp, -sqrt(3)/3*d_tmp];
    
            elseif rod_num_tmp == 3
                x_rod = [0, -d_tmp/2, d_tmp/2];
                y_rod = [d_tmp/sqrt(3), -d_tmp/sqrt(3)/2, -d_tmp/sqrt(3)/2];
    
            elseif rod_num_tmp == 10
                % ----- 10根棒：正三角形排布（1+2+3+4） -----
                R = 4;
                vs = sqrt(3)/2 * d_tmp;
                x_rod = []; y_rod = [];
                for row = 1:R
                    n = row;
                    y_row = ((R+1)/2 - row) * vs;
                    x_positions = (-(n-1)/2 : 1 : (n-1)/2) * d_tmp;
                    x_rod = [x_rod, x_positions];
                    y_rod = [y_rod, y_row * ones(1,n)];
                end
    
            elseif rod_num_tmp == 15
                % ----- 15根棒：正三角形排布（1+2+3+4+5） -----
                R = 5;
                vs = sqrt(3)/2 * d_tmp;
                x_rod = []; y_rod = [];
                for row = 1:R
                    n = row;
                    y_row = ((R+1)/2 - row) * vs;
                    x_positions = (-(n-1)/2 : 1 : (n-1)/2) * d_tmp;
                    x_rod = [x_rod, x_positions];
                    y_rod = [y_rod, y_row * ones(1,n)];
                end
            else
                error('Unsupported rod_num value: %d', rod_num_tmp);
            end
    
        else
            % ----- 偶数簇（Y 反向镜像） -----
            if rod_num_tmp == 6
                x_rod = [0, -d_tmp/2, d_tmp/2, -d_tmp, 0, d_tmp];
                y_rod = -[2*d_tmp/sqrt(3), sqrt(3)/6*d_tmp, sqrt(3)/6*d_tmp, -sqrt(3)/3*d_tmp, -sqrt(3)/3*d_tmp, -sqrt(3)/3*d_tmp];
    
            elseif rod_num_tmp == 3
                x_rod = [0, -d_tmp/2, d_tmp/2];
                y_rod = -[d_tmp/sqrt(3), -d_tmp/sqrt(3)/2, -d_tmp/sqrt(3)/2];
    
            elseif rod_num_tmp == 10
                R = 4;
                vs = sqrt(3)/2 * d_tmp;
                x_rod = []; y_rod = [];
                for row = 1:R
                    n = row;
                    y_row = -((R+1)/2 - row) * vs; % 取反
                    x_positions = (-(n-1)/2 : 1 : (n-1)/2) * d_tmp;
                    x_rod = [x_rod, x_positions];
                    y_rod = [y_rod, y_row * ones(1,n)];
                end
    
            elseif rod_num_tmp == 15
                R = 5;
                vs = sqrt(3)/2 * d_tmp;
                x_rod = []; y_rod = [];
                for row = 1:R
                    n = row;
                    y_row = -((R+1)/2 - row) * vs; % 取反
                    x_positions = (-(n-1)/2 : 1 : (n-1)/2) * d_tmp;
                    x_rod = [x_rod, x_positions];
                    y_rod = [y_rod, y_row * ones(1,n)];
                end
            else
                error('Unsupported rod_num value: %d', rod_num_tmp);
            end
        end
    
        for j = 1 : length(x_rod)
            x_rod_tmp = x_rod(j) + x_tmp;
            y_rod_tmp = y_rod(j) + y_tmp;
            r_tmp = sqrt(x_rod_tmp^2 + y_rod_tmp^2);

            theta_tmp_1 = -atan2(y_rod_tmp, x_rod_tmp) + pi/2;
            x_rod_tmp = r_tmp * cos(theta_tmp_1);
            y_rod_tmp = r_tmp * sin(theta_tmp_1);
            
            flag_rod = ((coor_seg(:, 1) - x_rod_tmp).^2 + (coor_seg(:, 2) - y_rod_tmp).^2) <= rod_r_tmp^2;
            flag_rod_h = abs(coor_seg(:, 3)) < rod_h/2;
            flag_rod = logical(flag_rod .* flag_rod_h);
            img_seg(flag_rod) = act;
        end
    end

end

function img_seg = PointArrayPhantom(coor_seg, X_value, Y_value, Z_value, rod_r, act)
    img_seg = zeros(size(coor_seg, 1), 1);    

    NumX = length(X_value);
    NumY = length(Y_value);
    NumZ = length(Z_value);

    
    for IdX = 1 : NumX
        X = X_value(IdX);
        for IdY = 1 : NumY
            Y = Y_value(IdY);
            for IdZ = 1 : NumZ
                Z = Z_value(IdZ);
                flag_sphere = ((coor_seg(:, 1) - X).^2 + (coor_seg(:, 2) - Y).^2 + (coor_seg(:, 3) - Z).^2) <= rod_r^2;
                img_seg(flag_sphere) = act;
            end
        end
    end
end