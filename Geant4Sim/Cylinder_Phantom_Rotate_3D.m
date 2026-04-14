rod_d = 300;
height = 120;
x_circle = 0;
y_circle = 0;

x_center = 0;
y_center = -245;
z = 0;
act = 6;
LengthUnit = 'mm';
Theta_Min = 0;
Theta_Max = 180;
AngleUnit = 'deg';
ene = 662; % keV
rotate_num = 10;
save_path = "./Macro/Cylinder_300_120_0_0/";
mkdir(save_path);

theta_circle = atan(y_circle/x_circle);
r_circle = sqrt(x_circle^2 + y_circle^2);
for id_rotate = 1 : rotate_num
    phi = (id_rotate - 1) * 2 * pi / rotate_num;
    x_circle_tmp = r_circle * cos(theta_circle - phi) + x_center;
    y_circle_tmp = r_circle * sin(theta_circle - phi) + y_center;
    
    fid=fopen(sprintf("%s%d.mac", save_path, id_rotate),'w');
    
    fprintf(fid, "/gps/particle gamma\n");
    fprintf(fid, "/gps/energy %d keV\n", ene);
    fprintf(fid, "/gps/pos/type Volume\n");
    fprintf(fid, "/gps/pos/shape Cylinder\n");
    fprintf(fid, "/gps/pos/radius %.4f %s\n", rod_d/2, LengthUnit);
    fprintf(fid, "/gps/pos/halfz %.4f %s\n", height/2, LengthUnit);
    fprintf(fid, "/gps/pos/centre %.4f %.4f %.4f %s\n", x_circle_tmp, y_circle_tmp, z, LengthUnit);
    fprintf(fid, "/gps/ang/type iso\n");
    fprintf(fid,'/gps/ang/mintheta %.4f %s\n', Theta_Min, AngleUnit);
    fprintf(fid,'/gps/ang/maxtheta %.4f %s\n', Theta_Max, AngleUnit);
    fprintf(fid,'\n#\n');
    
    fprintf(fid,'/run/beamOn 50000000\n');
    fclose(fid);
end