back_rod_d = 240;
rod_d = 16:4:36;
height = 30;

x_center = 0;
y_center = -245;
z = 0;
act = 6;
LengthUnit = 'mm';
Theta_Min = 0;
Theta_Max = 180;
AngleUnit = 'deg';
ene = 511; % keV
rotate_num = 20;
save_path = "./Macro/ContrastPhantom_240_30_511keV_RotateNum20_Test/";
mkdir(save_path);

for id_rotate = 1 : rotate_num
    phi = (id_rotate - 1) * 2 * pi / rotate_num;
    fid=fopen(sprintf("%s%d.mac", save_path, id_rotate),'w');
    
    fprintf(fid, "/gps/particle gamma\n");
    fprintf(fid, "/gps/energy %d keV\n", ene);
    fprintf(fid, "/gps/pos/type Volume\n");
    fprintf(fid, "/gps/pos/shape Cylinder\n");
    fprintf(fid, "/gps/pos/radius %.4f %s\n", back_rod_d/2, LengthUnit);
    fprintf(fid, "/gps/pos/halfz %.4f %s\n", height/2, LengthUnit);
    fprintf(fid, "/gps/pos/centre %.4f %.4f %.4f %s\n", x_center, y_center, z, LengthUnit);
    fprintf(fid, "/gps/ang/type iso\n");
    fprintf(fid,'/gps/ang/mintheta %.4f %s\n', Theta_Min, AngleUnit);
    fprintf(fid,'/gps/ang/maxtheta %.4f %s\n', Theta_Max, AngleUnit);
    fprintf(fid,'\n#\n');
    
    for i = 1 : 6
        theta_tmp = (i-1) * pi/3 - phi;
        x_tmp = back_rod_d/4 * cos(theta_tmp) + x_center;
        y_tmp = back_rod_d/4 * sin(theta_tmp) + y_center;
        rod_d_tmp = rod_d(i);
        act_tmp = (act-1) * rod_d_tmp^2 / back_rod_d^2;
    
        fprintf(fid,'/gps/source/add %.6f\n', act_tmp);
        fprintf(fid, "/gps/particle gamma\n");
        fprintf(fid, "/gps/energy %d keV\n", ene);
        fprintf(fid, "/gps/pos/type Volume\n");
        fprintf(fid, "/gps/pos/shape Cylinder\n");
        fprintf(fid, "/gps/ang/type iso\n");
        fprintf(fid,'/gps/ang/mintheta %.4f %s\n', Theta_Min, AngleUnit);
        fprintf(fid,'/gps/ang/maxtheta %.4f %s\n', Theta_Max, AngleUnit);
        fprintf(fid,'/gps/pos/centre %.4f %.4f %.4f %s\n', x_tmp, y_tmp, z, LengthUnit);
        fprintf(fid,'/gps/pos/radius %.4f %s\n', rod_d_tmp/2, LengthUnit);
        fprintf(fid, "/gps/pos/halfz %.4f %s\n", height/2, LengthUnit);
        fprintf(fid,'\n#\n');
        
    end
    
    fprintf(fid,'/run/beamOn 500000\n');
    fclose(fid);
end