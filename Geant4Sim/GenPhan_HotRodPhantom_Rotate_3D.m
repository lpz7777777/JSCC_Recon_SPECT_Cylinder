back_rod_r = 200;
rod_r = 5:2:15;
rod_num = [10, 6, 6, 3, 3, 3];
height = 30;

rotate_num = 20;

x_center = 0;
y_center = -245;
z = 0;
act = 10;
LengthUnit = "mm";
Theta_Min = 0;
Theta_Max = 180;
AngleUnit = "deg";
center_factor = 0.5 : (-0.01) : 0.45;
center_factor = [0.5, 0.49, 0.48, 0.47, 0.46, 0.45];
center_factor = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
center_factor = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4];
% center_factor = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35];
ene = 511; % keV

flag_back_rod = 0;

phi_plot = linspace(0, 2*pi, 100);

save_path = "./Macro/HotRodPhantom_10_30_30_511keV_5e10/";
mkdir(save_path);

f = figure;
for id_rotate = 1 : rotate_num
    phi = (id_rotate - 1) * 2 * pi / rotate_num;
    fid = fopen(sprintf("%s%d.mac", save_path, id_rotate), "w");

    if flag_back_rod == 1
        fprintf(fid, "/gps/particle gamma\n");
        fprintf(fid, "/gps/energy %d keV\n", ene);
        fprintf(fid, "/gps/pos/type Plane\n");
        fprintf(fid, "/gps/pos/shape Annulus\n");
        fprintf(fid, "/gps/ang/type iso\n");
        fprintf(fid, "/gps/ang/mintheta %.4f %s\n", Theta_Min, AngleUnit);
        fprintf(fid, "/gps/ang/maxtheta %.4f %s\n", Theta_Max, AngleUnit);
        fprintf(fid, "/gps/pos/centre %.4f %.4f %.4f %s\n", x_center, y_center, z, LengthUnit);
        fprintf(fid, "/gps/pos/radius %.4f %s\n", back_rod_r, LengthUnit);
        fprintf(fid, "/gps/pos/inner_radius 0 mm\n");
        fprintf(fid, "\n#\n");
        x = back_rod_r .* cos(phi_plot);
        y = back_rod_r .* sin(phi_plot);
        fill(x, y, "k", "FaceAlpha", 1 / act, "EdgeColor", "none");
    end

    hold on

    for i = 1 : 6
        theta_tmp = (i - 1) * pi / 3 + pi / 6;
        x_tmp = back_rod_r * center_factor(i) * cos(theta_tmp);
        y_tmp = back_rod_r * center_factor(i) * sin(theta_tmp);
        rod_r_tmp = rod_r(i);
        rod_num_tmp = rod_num(i);

        if flag_back_rod == 1
            act_tmp = (act - 1) * rod_r_tmp^2 / back_rod_r^2;
        else
            act_tmp = rod_r_tmp^2 / rod_r(1)^2;
        end

        d_tmp = rod_r_tmp * 4;
        [x_rod, y_rod] = build_isosceles_triangle_rods(rod_num_tmp, d_tmp, mod(i, 2) == 0);

        if flag_back_rod == 1
            for j = 1 : length(x_rod)
                x_rod_tmp = x_rod(j) + x_tmp;
                y_rod_tmp = y_rod(j) + y_tmp;
                r = sqrt(x_rod_tmp^2 + y_rod_tmp^2);
                theta_rot = atan2(y_rod_tmp, x_rod_tmp) - phi;

                x_rod_tmp_rotate = r * cos(theta_rot) + x_center;
                y_rod_tmp_rotate = r * sin(theta_rot) + y_center;

                fprintf(fid, "/gps/source/add %.6f\n", act_tmp);
                fprintf(fid, "/gps/particle gamma\n");
                fprintf(fid, "/gps/energy %d keV\n", ene);
                fprintf(fid, "/gps/pos/type Volume\n");
                fprintf(fid, "/gps/pos/shape Cylinder\n");
                fprintf(fid, "/gps/ang/type iso\n");
                fprintf(fid, "/gps/ang/mintheta %.4f %s\n", Theta_Min, AngleUnit);
                fprintf(fid, "/gps/ang/maxtheta %.4f %s\n", Theta_Max, AngleUnit);
                fprintf(fid, "/gps/pos/centre %.4f %.4f %.4f %s\n", x_rod_tmp_rotate, y_rod_tmp_rotate, z, LengthUnit);
                fprintf(fid, "/gps/pos/radius %.4f %s\n", rod_r_tmp, LengthUnit);
                fprintf(fid, "/gps/pos/halfz %.4f %s\n", height / 2, LengthUnit);
                fprintf(fid, "\n#\n");

                if id_rotate == 1
                    x = x_rod_tmp + rod_r_tmp .* cos(phi_plot);
                    y = y_rod_tmp + rod_r_tmp .* sin(phi_plot);
                    fill(x, y, "k", "FaceAlpha", 1, "EdgeColor", "none");
                    hold on
                end
            end
        else
            for j = 1 : length(x_rod)
                x_rod_tmp = x_rod(j) + x_tmp;
                y_rod_tmp = y_rod(j) + y_tmp;
                r = sqrt(x_rod_tmp^2 + y_rod_tmp^2);
                theta_rot = atan2(y_rod_tmp, x_rod_tmp) - phi;

                x_rod_tmp_rotate = r * cos(theta_rot) + x_center;
                y_rod_tmp_rotate = r * sin(theta_rot) + y_center;

                if i * j ~= 1
                    fprintf(fid, "/gps/source/add %.6f\n", act_tmp);
                end

                fprintf(fid, "/gps/particle gamma\n");
                fprintf(fid, "/gps/energy %d keV\n", ene);
                fprintf(fid, "/gps/pos/type Volume\n");
                fprintf(fid, "/gps/pos/shape Cylinder\n");
                fprintf(fid, "/gps/ang/type iso\n");
                fprintf(fid, "/gps/ang/mintheta %.4f %s\n", Theta_Min, AngleUnit);
                fprintf(fid, "/gps/ang/maxtheta %.4f %s\n", Theta_Max, AngleUnit);
                fprintf(fid, "/gps/pos/centre %.4f %.4f %.4f %s\n", x_rod_tmp_rotate, y_rod_tmp_rotate, z, LengthUnit);
                fprintf(fid, "/gps/pos/radius %.4f %s\n", rod_r_tmp, LengthUnit);
                fprintf(fid, "/gps/pos/halfz %.4f %s\n", height / 2, LengthUnit);
                fprintf(fid, "\n#\n");

                if id_rotate == 1
                    x = x_rod_tmp + rod_r_tmp .* cos(phi_plot);
                    y = y_rod_tmp + rod_r_tmp .* sin(phi_plot);
                    fill(x, y, "k", "FaceAlpha", 1, "EdgeColor", "none");
                    hold on
                end
            end
        end
    end

    fprintf(fid, "/run/beamOn 25000000\n");
    fclose(fid);
end

Min_X = -150;
Max_X = -Min_X;
Min_Y = Min_X;
Max_Y = Max_X;
axis equal;
axis square
% axis off
xlim([Min_X Max_X]);
ylim([Min_Y Max_Y]);
f.Position = [400, 400, 300, 300];

function [x_rod, y_rod] = build_isosceles_triangle_rods(rod_num_tmp, d_tmp, flip_y)
    row_num = (sqrt(8 * rod_num_tmp + 1) - 1) / 2;
    if abs(row_num - round(row_num)) > 1e-8
        error("rod_num=%d cannot form a 1+2+...+R isosceles triangle array.", rod_num_tmp);
    end

    row_num = round(row_num);
    vertical_step = sqrt(3) / 2 * d_tmp;
    x_rod = [];
    y_rod = [];

    for row = 1 : row_num
        rod_count_in_row = row;
        y_row = (2 * (row_num - 1) / 3 - (row - 1)) * vertical_step;
        x_positions = (-(rod_count_in_row - 1) / 2 : 1 : (rod_count_in_row - 1) / 2) * d_tmp;

        x_rod = [x_rod, x_positions];
        y_rod = [y_rod, y_row * ones(1, rod_count_in_row)];
    end

    if flip_y
        y_rod = -y_rod;
    end
end
