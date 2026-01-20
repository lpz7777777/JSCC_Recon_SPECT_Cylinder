folderPath = uigetdir("./Figure/");
file_path = "./Factors/511keV/";
load(sprintf("%sRotMat.mat", file_path));
load(sprintf("%sRotMatInv.mat", file_path));
load(sprintf("%scoor_polar.mat", file_path));

show_center = [-30, 0, 0];
generate = 0;

if folderPath ~= 0
    slashPositions = strfind(folderPath, '\');
    if ~isempty(slashPositions)
        Name = folderPath(slashPositions(end) + 1 : end);
    else
        Name = folderPath;
    end
end
% Name = "ContrastPhantom_70_5e9_1_ER0.08_OSEM4_SDU5726210_DDU433766";
Path = sprintf("./Figure/%s/", Name);
Sigma_Gaussfit = 0.01;
Subset_Num = 1;
iter_max = 4000;
iter_interval = 100;
% iter_show_tmp = 50:50:200;
% iter_show_tmp = 25:25:100;
% iter_show_tmp = 10:10:40;
% % iter_show_tmp = 5:5:20;
% iter_show_tmp = 100:100:400;
% iter_show_tmp = 200:200:800;
% iter_show_tmp = 250:250:1000;
% iter_show_tmp = 450:450:1800;

% iter_show_tmp = 1250:1250:5000;

% iter_show_tmp = 2000:2000:8000;
% iter_show_tmp = 1000:1000:4000;
iter_show_tmp = 500:500:2000;
% iter_show_tmp = 2500:2500:10000;
% iter_show_tmp = 5000:5000:20000;

pixel_num_x = 100;
pixel_num_y = 100;
pixel_l_x = 3;
pixel_l_y = 3;
pixel_l_z = 3;
fov_l_x = pixel_num_x * pixel_l_x;
fov_l_y = pixel_num_y * pixel_l_y;
pixel_num_cartesian_z = 20;
rotate_num = size(RotMat, 2);
show_center_pixcel = round(show_center ./ [pixel_l_x, pixel_l_y, pixel_l_z] + [pixel_num_x, pixel_num_y, pixel_num_cartesian_z]/2);

Cut_Range = 5;
range_U = (1 + Cut_Range) : (pixel_num_x - Cut_Range);
range_V = (1 + Cut_Range) : (pixel_num_y - Cut_Range);

Min_X = -1/2 * pixel_num_x * pixel_l_x;
Max_X = -Min_X;
Min_Y = Min_X;
Max_Y = Max_X;
Min_Z = -1/2 * pixel_num_cartesian_z * pixel_l_z;
Max_Z = -Min_Z;

color = 1 - gray(256);
% color = hot;

Path_Polar = sprintf("%sPolar/", Path);
Path_Cartesian = sprintf("%sCartesian/", Path);
mkdir(Path_Cartesian);


%%
% f = figure;
% f.Position = [100, 100, 1600, 800];
% t = tiledlayout(1, 2);
% 
% nexttile;
% plot(coor_polar(:, 1), coor_polar(:, 2), ".", "MarkerSize", 6);axis equal;axis square;
% xlim([-150, 150]);
% ylim([-150, 150]);
% nexttile;
% plot(reshape(coor_cartesian_x, [], 1), reshape(coor_cartesian_y, [], 1), ".", "MarkerSize", 3);axis equal;axis square;
% xlim([-150, 150]);
% ylim([-150, 150]);
%%
if generate == 1
[coor_cartesian_x, coor_cartesian_y] = meshgrid(((-fov_l_x/2 + pixel_l_x/2) : pixel_l_x : (fov_l_x/2 - pixel_l_x/2)), ((-fov_l_y/2 + pixel_l_y/2) : pixel_l_y : (fov_l_y/2 - pixel_l_y/2)));

%%
fid = fopen(sprintf("%sImage_SC_Iter_%d_%d", Path_Polar, iter_max, floor(iter_max/iter_interval)), "r");
img_sc_iter = reshape(fread(fid, "float32"), size(coor_polar, 1), pixel_num_cartesian_z, []);
fclose(fid);

fid = fopen(sprintf("%sImage_SCD_Iter_%d_%d", Path_Polar, round(iter_max/2), floor(iter_max/2/iter_interval)), "r");
img_scd_iter = reshape(fread(fid, "float32"), size(coor_polar, 1), pixel_num_cartesian_z, []);
fclose(fid);

fid = fopen(sprintf("%sImage_JSCCD_Iter_%d_%d", Path_Polar, round(iter_max/2), floor(iter_max/2/iter_interval)), "r");
img_jsccd_iter = reshape(fread(fid, "float32"), size(coor_polar, 1), pixel_num_cartesian_z, []);
fclose(fid);

fid = fopen(sprintf("%sImage_JSCCSD_Iter_%d_%d", Path_Polar, iter_max, floor(iter_max/iter_interval)), "r");
img_jsccsd_iter = reshape(fread(fid, "float32"), size(coor_polar, 1), pixel_num_cartesian_z, []);
fclose(fid);

%% To Cartesian
img_sc_iter_cartesian = zeros(pixel_num_x*pixel_num_x, pixel_num_cartesian_z, size(img_sc_iter, 3));
for iter_id = 1 : size(img_sc_iter, 3)
    img_tmp = img_sc_iter(:, :, iter_id);
    for id_z = 1 : pixel_num_cartesian_z
        Img_Polar_tmp = img_tmp(:, id_z);
        Img_cartesian_tmp = griddata(coor_polar(:, 1), coor_polar(:, 2), Img_Polar_tmp, coor_cartesian_x, coor_cartesian_y, "linear").';
        img_sc_iter_cartesian(:, id_z, iter_id) = reshape(Img_cartesian_tmp, 1, []);
    end
end
img_sc_iter_cartesian(isnan(img_sc_iter_cartesian)) = 0;

img_scd_iter_cartesian = zeros(pixel_num_x*pixel_num_x, pixel_num_cartesian_z, size(img_scd_iter, 3));
for iter_id = 1 : size(img_scd_iter, 3)
    img_tmp = img_scd_iter(:, :, iter_id);
    for id_z = 1 : pixel_num_cartesian_z
        Img_Polar_tmp = img_tmp(:, id_z);
        Img_cartesian_tmp = griddata(coor_polar(:, 1), coor_polar(:, 2), Img_Polar_tmp, coor_cartesian_x, coor_cartesian_y, "linear").';
        img_scd_iter_cartesian(:, id_z, iter_id) = reshape(Img_cartesian_tmp, 1, []);
    end
end
img_scd_iter_cartesian(isnan(img_scd_iter_cartesian)) = 0;

img_jsccd_iter_cartesian = zeros(pixel_num_x*pixel_num_x, pixel_num_cartesian_z, size(img_jsccd_iter, 3));
for iter_id = 1 : size(img_jsccd_iter, 3)
    img_tmp = img_jsccd_iter(:, :, iter_id);
    for id_z = 1 : pixel_num_cartesian_z
        Img_Polar_tmp = img_tmp(:, id_z);
        Img_cartesian_tmp = griddata(coor_polar(:, 1), coor_polar(:, 2), Img_Polar_tmp, coor_cartesian_x, coor_cartesian_y, "linear").';
        img_jsccd_iter_cartesian(:, id_z, iter_id) = reshape(Img_cartesian_tmp, 1, []);
    end
end
img_jsccd_iter_cartesian(isnan(img_jsccd_iter_cartesian)) = 0;

img_jsccsd_iter_cartesian = zeros(pixel_num_x*pixel_num_x, pixel_num_cartesian_z, size(img_jsccsd_iter, 3));
for iter_id = 1 : size(img_jsccsd_iter, 3)
    img_tmp = img_jsccsd_iter(:, :, iter_id);
    for id_z = 1 : pixel_num_cartesian_z
        Img_Polar_tmp = img_tmp(:, id_z);
        Img_cartesian_tmp = griddata(coor_polar(:, 1), coor_polar(:, 2), Img_Polar_tmp, coor_cartesian_x, coor_cartesian_y, "linear").';
        img_jsccsd_iter_cartesian(:, id_z, iter_id) = reshape(Img_cartesian_tmp, 1, []);
    end
end
img_jsccsd_iter_cartesian(isnan(img_jsccsd_iter_cartesian)) = 0;

fid = fopen(sprintf("%sImage_SC_Iter_%d_%d", Path_Cartesian, iter_max, floor(iter_max/iter_interval)), "w");
fwrite(fid, img_sc_iter_cartesian, "float32");
fclose(fid);

fid = fopen(sprintf("%sImage_SCD_Iter_%d_%d", Path_Cartesian, round(iter_max/2), floor(iter_max/2/iter_interval)), "w");
fwrite(fid, img_scd_iter_cartesian, "float32");
fclose(fid);

fid = fopen(sprintf("%sImage_JSCCD_Iter_%d_%d", Path_Cartesian, round(iter_max/2), floor(iter_max/2/iter_interval)), "w");
fwrite(fid, img_jsccd_iter_cartesian, "float32");
fclose(fid);

fid = fopen(sprintf("%sImage_JSCCSD_Iter_%d_%d", Path_Cartesian, iter_max, floor(iter_max/iter_interval)), "w");
fwrite(fid, img_jsccsd_iter_cartesian, "float32");
fclose(fid);


else
%%
fid = fopen(sprintf("%sImage_SC_Iter_%d_%d", Path_Cartesian, iter_max, floor(iter_max/iter_interval)), "r");
img_sc_iter_cartesian = reshape(fread(fid, "float32"), pixel_num_x, pixel_num_y, pixel_num_cartesian_z, []);
fclose(fid);

fid = fopen(sprintf("%sImage_SCD_Iter_%d_%d", Path_Cartesian, round(iter_max/2), floor(iter_max/2/iter_interval)), "r");
img_scd_iter_cartesian = reshape(fread(fid, "float32"), pixel_num_x, pixel_num_y, pixel_num_cartesian_z, []);
fclose(fid);

fid = fopen(sprintf("%sImage_JSCCD_Iter_%d_%d", Path_Cartesian, round(iter_max/2), floor(iter_max/2/iter_interval)), "r");
img_jsccd_iter_cartesian = reshape(fread(fid, "float32"), pixel_num_x, pixel_num_y, pixel_num_cartesian_z, []);
fclose(fid);

fid = fopen(sprintf("%sImage_JSCCSD_Iter_%d_%d", Path_Cartesian, iter_max, floor(iter_max/iter_interval)), "r");
img_jsccsd_iter_cartesian = reshape(fread(fid, "float32"), pixel_num_x, pixel_num_y, pixel_num_cartesian_z, []);
fclose(fid);

%%
f = figure;
f.Position = [100 100 1100 800];

t_outer = tiledlayout(length(iter_show_tmp), 15);
t_outer.TileSpacing = 'none';
t_outer.Padding = "tight";

% t_inner_sc = cell(length(iter_show_tmp), 1);
% t_inner_jsccsd = cell(length(iter_show_tmp), 1);

id_iter_show = 0;
for iter_show = iter_show_tmp
    id_iter_show = id_iter_show + 1;

    img_sc = img_sc_iter_cartesian(:, :, :, round(iter_show/iter_interval));
    img_jsccsd = img_jsccsd_iter_cartesian(:, :, :, round(iter_show/iter_interval));

    % SC
    % --------Transverse--------
    nexttile(t_outer, [1, 3]);
    img_tmp = img_sc(:, :, show_center_pixcel(3));
    img_tmp = imgaussfilt(img_tmp, Sigma_Gaussfit);
    max_colorbar = 1 * max(img_tmp(range_U,range_V), [], "all");
    imagesc([Min_X Max_X], [Min_Y Max_Y], img_tmp, [0, max_colorbar]);
    cb = colorbar("westoutside");
    cb.Label.String = sprintf("Iteration=%d", iter_show);
    cb.Label.FontSize = 11;
    axis equal                                                                                                                                             
    colormap(color);
    ylabel("x (mm)");
    ylim([Min_X Max_X]);
    xlim([Min_Y Max_Y]);

    hold on
    line([Min_Y Max_Y]*3/4, [show_center(1), show_center(1)], 'Color','red','LineStyle','--', "LineWidth", 0.5);
    line([show_center(2), show_center(2)], [Min_X Max_X]*3/4, 'Color','blue','LineStyle','--', "LineWidth", 0.5);

    if id_iter_show == 1
        title(sprintf("SC\nTransverse"), "FontSize", 11);
    elseif id_iter_show == length(iter_show_tmp)
        xlabel("y (mm)");
    end

    % --------Coronal--------
    nexttile(t_outer, [1, 2]);
    img_tmp = squeeze(img_sc(:, show_center_pixcel(2), :));
    img_tmp = imgaussfilt(img_tmp, Sigma_Gaussfit);
    imagesc([Min_Z Max_Z], [Min_X Max_X], img_tmp, [0, max_colorbar]);
    axis equal                                                                                                                                             
    colormap(color);
    ylabel("x (mm)");
    ylim([Min_X Max_X]);
    xlim([Min_Z Max_Z]);

    hold on
    line([show_center(3), show_center(3)], [Min_X Max_X]*3/4, 'Color','black','LineStyle','--', "LineWidth", 0.5);

    if id_iter_show == 1
        title("Coronal", "FontSize", 11, "Color", "blue");
    elseif id_iter_show == length(iter_show_tmp)
        xlabel("z (mm)");
    end

    % --------Sagittal--------
    nexttile(t_outer, [1, 2]);
    img_tmp = squeeze(img_sc(show_center_pixcel(1), :, :));
    img_tmp = imgaussfilt(img_tmp, Sigma_Gaussfit);
    % max_colorbar = 1 * max(img_tmp(range_V, :), [], "all");
    imagesc([Min_Z Max_Z], [Min_Y Max_Y], img_tmp, [0, max_colorbar]);
    axis equal                                                                                                                                             
    colormap(color);
    ylabel("y (mm)");
    ylim([Min_Y Max_Y]);
    xlim([Min_Z Max_Z]);

    if id_iter_show == 1
        title("Sagittal", "FontSize", 11, "Color", "red");
    elseif id_iter_show == length(iter_show_tmp)
        xlabel("z (mm)");
    end

    ax = nexttile(t_outer);
    % ylim(ax, [Min_X Max_X]);
    % xlim(ax, [Min_Z Max_Z]);
    % line([0, 0], [Min_X Max_X], 'Color','black', "LineWidth", 2);
    ax.Visible = "off";

    % JSCC
    % --------Transverse--------
    nexttile(t_outer, [1, 3]);
    img_tmp = img_jsccsd(:, :, show_center_pixcel(3));
    img_tmp = imgaussfilt(img_tmp, Sigma_Gaussfit);
    max_colorbar = 1 * max(img_tmp(range_U,range_V), [], "all");
    imagesc([Min_X Max_X], [Min_Y Max_Y], img_tmp, [0, max_colorbar]);
    cb = colorbar("westoutside");
    % cb.Label.String = sprintf("Iteration=%d", iter_show);
    axis equal                                                                                                                                             
    colormap(color);
    ylabel("x (mm)");  
    ylim([Min_X Max_X]);
    xlim([Min_Y Max_Y]);

    hold on
    line([Min_Y Max_Y]*3/4, [show_center(1), show_center(1)], 'Color','red','LineStyle','--', "LineWidth", 0.5);
    line([show_center(2), show_center(2)], [Min_X Max_X]*3/4, 'Color','blue','LineStyle','--', "LineWidth", 0.5);

    if id_iter_show == 1
        title(sprintf("JSCC\nTransverse"), "FontSize", 11);
    elseif id_iter_show == length(iter_show_tmp)
        xlabel("y (mm)");
    end

    % --------Coronal--------
    nexttile(t_outer, [1, 2]);
    img_tmp = squeeze(img_jsccsd(:, show_center_pixcel(2), :));
    img_tmp = imgaussfilt(img_tmp, Sigma_Gaussfit);
    % max_colorbar = 1 * max(img_tmp(range_U, :), [], "all");
    imagesc([Min_Z Max_Z], [Min_X Max_X], img_tmp, [0, max_colorbar]);
    axis equal                                                                                                                                             
    colormap(color);
    ylabel("x (mm)");
    ylim([Min_X Max_X]);
    xlim([Min_Z Max_Z]);

    hold on
    line([show_center(3), show_center(3)], [Min_X Max_X]*3/4, 'Color','black','LineStyle','--', "LineWidth", 0.5);

    if id_iter_show == 1
        title("Coronal", "FontSize", 11, "Color", "blue");
    elseif id_iter_show == length(iter_show_tmp)
        xlabel("z (mm)");
    end

    % --------Sagittal--------
    nexttile(t_outer, [1, 2]);
    img_tmp = squeeze(img_jsccsd(show_center_pixcel(1), :, :));
    img_tmp = imgaussfilt(img_tmp, Sigma_Gaussfit);
    % max_colorbar = 1 * max(img_tmp(range_V, :), [], "all");
    imagesc([Min_Z Max_Z], [Min_Y Max_Y], img_tmp, [0, max_colorbar]);
    axis equal                                                                                                                                             
    colormap(color);
    ylabel("y (mm)");
    ylim([Min_Y Max_Y]);
    xlim([Min_Z Max_Z]);

    if id_iter_show == 1
        title("Sagittal", "FontSize", 11, "Color", "red");
    elseif id_iter_show == length(iter_show_tmp)
        xlabel("z (mm)");
    end
end

title(t_outer, sprintf("Data Name: %s    OSEM Subset Number: %d", Name, Subset_Num), "Interpreter", "none");
saveas(f, sprintf('%simg_show.png', Path));
saveas(f, "img_show.png");
end

%%



%%
% f = figure;
% f.Position = [100 100 1000 300];
% 
% t = tiledlayout(1, 3);
% t.TileSpacing = 'tight';
% t.Padding = "tight";
% 
% % for iter_show = iter_show_tmp
% iter_sc = 400;
% iter_jsccsd = 200;
% 
% img_sc = img_sc_iter(:, iter_sc/iter_interval);
% img_jsccsd = img_jsccsd_iter(:, iter_jsccsd/iter_interval);
% 
% % x_tmp = -222.5 : 5 : 222.5;
% % x_tmp = -267 : 6 : 267;
% 
% lineprofile_x = [-150, 150];
% lineprofile_y = [0, 0];
% 
% nexttile;
% a = reshape(img_sc, pixel_num_x, pixel_num_y);
% a = imgaussfilt(a, Sigma_Gaussfit);
% imagesc([Min_X Max_X], [Min_Y Max_Y], a(range_U,range_V), [0, 1 * max(a(range_U,range_V), [], "all")]);
% colorbar
% axis square
% colormap(color);
% 
% hold on
% line(lineprofile_y, lineprofile_x, 'Color','red','LineStyle','--', "LineWidth", 2);
% 
% title(sprintf("SC, Iter=%d", iter_sc));
% 
% nexttile;
% a = reshape(img_jsccsd, pixel_num_x, pixel_num_y);
% a = imgaussfilt(a, Sigma_Gaussfit);
% imagesc([Min_X Max_X], [Min_Y Max_Y], a(range_U,range_V), [0, 1 * max(a(range_U,range_V), [], "all")]);
% colorbar
% axis square
% % axis off
% colormap(color);
% 
% hold on
% line(lineprofile_y, lineprofile_x, 'Color','red','LineStyle','--', "LineWidth", 2);
% 
% title(sprintf("JSCC, Iter=%d", iter_jsccsd));
% 
% 
% if lineprofile_x(1) ~= lineprofile_x(2)
%     x_index_start = ceil(lineprofile_x(1) / pixel_l_x + pixel_num_x / 2);
%     x_index_end = ceil(lineprofile_x(2) / pixel_l_x + pixel_num_x / 2);
%     y_index = ceil(lineprofile_y(1) / pixel_l_x + pixel_num_y / 2);
% 
%     x_start = (x_index_start - pixel_num_x/2 - 1/2) * pixel_l_x;
%     x_end = (x_index_end - pixel_num_x/2 - 1/2) * pixel_l_x;
%     x_tmp = x_start : pixel_l_x : x_end;
% 
% 
%     nexttile;
%     hold on
%     a = reshape(img_sc, pixel_num_x, pixel_num_y);
%     a = imgaussfilt(a, Sigma_Gaussfit);
%     plot(x_tmp, a(x_index_start:x_index_end, y_index), "LineWidth", 2);
% 
%     a = reshape(img_jsccsd, pixel_num_x, pixel_num_y);
%     a = imgaussfilt(a, Sigma_Gaussfit);
%     plot(x_tmp, a(x_index_start:x_index_end, y_index), "LineWidth", 2);
%     title("Line Profile");
%     legend("SC", "JSCC");
%     xlabel("x(mm)");
%     ylabel("Pixel Value");
% 
%     xlim([x_start, x_end]);
% 
% end
% 
% saveas(f, 'LineProfile.png');
% 
