%% change anger - 3D ver
format bank;

rod_r = 5;

X_value = -70 : 20 : 70;
Y_value = -70 : 20 : 70;
Z_value = -20 : 20 : 20;

% X_value = -40 : 40 : 40;
% Y_value = -40 : 40 : 40;
% Z_value = -40 : 40 : 40;
% 
% X_value = -20 : 40 : 20;
% Y_value = -20 : 40 : 20;
% Z_value = -20 : 40 : 20;

NumX = length(X_value);
NumY = length(Y_value);
NumZ = length(Z_value);
LengthUnit = 'mm';

x_center = 0;
y_center = -245;
rotate_num = 20;

Theta_Min = 0;
Theta_Max = 180;
AngleUnit = 'deg';

save_path = "./Macro/PointArray_8_8_3_20/";
mkdir(save_path);
for id_rotate = 1 : rotate_num
    fid=fopen(sprintf("%s%d.mac", save_path, id_rotate),'w');

    phi = (id_rotate - 1) * 2 * pi / rotate_num;

    for IdX = 1 : NumX
        X = X_value(IdX);
        for IdY = 1 : NumY
            Y = Y_value(IdY);
            for IdZ = 1 : NumZ
                theta_tmp = atan2(Y, X);

                if isnan(theta_tmp)
                    theta_tmp = 0;
                end

                r_tmp = sqrt(X^2 + Y^2);
                X_tmp = r_tmp * cos(theta_tmp - phi) + x_center;
                Y_tmp = r_tmp * sin(theta_tmp - phi) + y_center;

                Z = Z_value(IdZ);

                if IdX*IdY*IdZ ~= 1
                    fprintf(fid,'/gps/source/add 1\n');
                end
                fprintf(fid, "/gps/particle gamma\n");
                fprintf(fid, "/gps/energy 511 keV\n");
                fprintf(fid, "/gps/pos/type Volume\n");
                fprintf(fid, "/gps/pos/shape Sphere\n");
                fprintf(fid, "/gps/pos/radius %.4f %s\n", rod_r, LengthUnit);
                fprintf(fid, "/gps/ang/type iso\n");
                fprintf(fid,'/gps/ang/mintheta %.4f %s\n', Theta_Min, AngleUnit);
                fprintf(fid,'/gps/ang/maxtheta %.4f %s\n', Theta_Max, AngleUnit);
                fprintf(fid,'/gps/pos/centre %.4f %.4f %.4f %s\n', X_tmp, Y_tmp, Z, LengthUnit);
                fprintf(fid,'#\n');
            end
        end
    end

    fprintf(fid,'/run/beamOn 25000000\n');
    fclose(fid);
end

