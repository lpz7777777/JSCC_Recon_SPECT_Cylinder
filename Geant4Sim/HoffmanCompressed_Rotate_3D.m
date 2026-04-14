%% HoffmanCompressed_Rotate_3D
% Generate Geant4 GPS macros from the compressed Hoffman phantom volume.
% Default: use the project-aligned target volume (100 x 100 x 20).
%
% If your Geant4 GPS build does not support "Para" for box-like voxel
% sources, switch cfg.source_shape to "Sphere".

clear;
clc;
format bank;

cfg.volume_type = "target"; % "target" or "native"
cfg.activity_threshold = 1e-4;
cfg.activity_power = 1.0;

cfg.ene_keV = 511;
cfg.rotate_num = 20;
cfg.beam_on = 50000000;

cfg.x_center_mm = 0;
cfg.y_center_mm = -245;
cfg.z_center_mm = 0;

cfg.theta_min_deg = 0;
cfg.theta_max_deg = 180;
cfg.angle_unit = "deg";
cfg.length_unit = "mm";
cfg.source_shape = "Sphere"; % "Para" or "Sphere"

script_dir = fileparts(mfilename("fullpath"));
% phantom_dir = fullfile(script_dir, "Preview", "HoffmanRawCompressed_300x300x60");
phantom_dir = fullfile(script_dir, "Preview", "HoffmanRawCompressed_300x300x60_XYx1p40");

[volume, pixel_size_mm, volume_name_tag] = load_compressed_volume(phantom_dir, cfg.volume_type);

voxel_table = build_voxel_table(volume, pixel_size_mm, cfg.activity_threshold, cfg.activity_power);
save_path = fullfile(script_dir, "Macro", sprintf("Hoffman_%s_%dsrc_RotateNum%d_5e10", volume_name_tag, size(voxel_table, 1), cfg.rotate_num));
if ~exist(save_path, "dir")
    mkdir(save_path);
end

write_summary(fullfile(save_path, "source_summary.txt"), cfg, pixel_size_mm, size(volume), voxel_table, save_path);

for id_rotate = 1:cfg.rotate_num
    phi = (id_rotate - 1) * 2 * pi / cfg.rotate_num;
    fid = fopen(fullfile(save_path, sprintf("%d.mac", id_rotate)), "w");
    if fid < 0
        error("Failed to open macro file for writing.");
    end

    for k = 1:size(voxel_table, 1)
        weight = voxel_table(k, 4);
        x0 = voxel_table(k, 1);
        y0 = voxel_table(k, 2);
        z0 = voxel_table(k, 3) + cfg.z_center_mm;

        theta_tmp = atan2(y0, x0);
        if isnan(theta_tmp)
            theta_tmp = 0;
        end
        r_tmp = sqrt(x0 ^ 2 + y0 ^ 2);
        x_rot = r_tmp * cos(theta_tmp - phi) + cfg.x_center_mm;
        y_rot = r_tmp * sin(theta_tmp - phi) + cfg.y_center_mm;

        if k ~= 1
            fprintf(fid, "/gps/source/add %.8f\n", weight);
        end
        write_single_source(fid, cfg, x_rot, y_rot, z0, pixel_size_mm);
    end

    fprintf(fid, "/run/beamOn %d\n", cfg.beam_on);
    fclose(fid);
end

fprintf("Macro output: %s\n", save_path);
fprintf("Volume type: %s\n", cfg.volume_type);
fprintf("Voxel sources: %d\n", size(voxel_table, 1));
fprintf("Pixel size: %.5f x %.5f x %.5f mm\n", pixel_size_mm(1), pixel_size_mm(2), pixel_size_mm(3));


function [volume, pixel_size_mm, name_tag] = load_compressed_volume(phantom_dir, volume_type)
switch string(volume_type)
    case "target"
        raw_path = fullfile(phantom_dir, "hoffman_compressed_target_float32.raw");
        dims = [100, 100, 20];
        pixel_size_mm = [3, 3, 3];
        name_tag = "Target";
    case "native"
        raw_path = fullfile(phantom_dir, "hoffman_compressed_native_float32.raw");
        dims = [256, 256, 19];
        pixel_size_mm = [0.97656, 0.97656, 6.39];
        name_tag = "Native";
    otherwise
        error("Unsupported volume_type: %s", volume_type);
end

fid = fopen(raw_path, "r", "ieee-le");
if fid < 0
    error("Failed to open %s", raw_path);
end
volume = fread(fid, prod(dims), "single=>single");
fclose(fid);
if numel(volume) ~= prod(dims)
    error("Unexpected element count in %s", raw_path);
end
volume = reshape(volume, dims);
end


function voxel_table = build_voxel_table(volume, pixel_size_mm, activity_threshold, activity_power)
mask = volume > activity_threshold;
[ix, iy, iz] = ind2sub(size(volume), find(mask));
weights = double(volume(mask)) .^ activity_power;

x = voxel_centers(size(volume, 1), pixel_size_mm(1));
y = voxel_centers(size(volume, 2), pixel_size_mm(2));
z = voxel_centers(size(volume, 3), pixel_size_mm(3));

voxel_table = [x(ix)', y(iy)', z(iz)', weights];
voxel_table = sortrows(voxel_table, -4);
end


function write_single_source(fid, cfg, x_mm, y_mm, z_mm, pixel_size_mm)
fprintf(fid, "/gps/particle gamma\n");
fprintf(fid, "/gps/energy %.6f keV\n", cfg.ene_keV);
fprintf(fid, "/gps/pos/type Volume\n");
fprintf(fid, "/gps/ang/type iso\n");
fprintf(fid, "/gps/ang/mintheta %.6f %s\n", cfg.theta_min_deg, cfg.angle_unit);
fprintf(fid, "/gps/ang/maxtheta %.6f %s\n", cfg.theta_max_deg, cfg.angle_unit);
fprintf(fid, "/gps/pos/centre %.6f %.6f %.6f %s\n", x_mm, y_mm, z_mm, cfg.length_unit);

switch string(cfg.source_shape)
    case "Para"
        fprintf(fid, "/gps/pos/shape Para\n");
        fprintf(fid, "/gps/pos/halfx %.6f %s\n", pixel_size_mm(1) / 2, cfg.length_unit);
        fprintf(fid, "/gps/pos/halfy %.6f %s\n", pixel_size_mm(2) / 2, cfg.length_unit);
        fprintf(fid, "/gps/pos/halfz %.6f %s\n", pixel_size_mm(3) / 2, cfg.length_unit);
        fprintf(fid, "/gps/pos/paralp 0.0 deg\n");
        fprintf(fid, "/gps/pos/parthe 0.0 deg\n");
        fprintf(fid, "/gps/pos/parphi 0.0 deg\n");
    case "Sphere"
        fprintf(fid, "/gps/pos/shape Sphere\n");
        fprintf(fid, "/gps/pos/radius %.6f %s\n", min(pixel_size_mm) / 2, cfg.length_unit);
    otherwise
        error("Unsupported source_shape: %s", cfg.source_shape);
end

fprintf(fid, "#\n");
end


function write_summary(file_path, cfg, pixel_size_mm, volume_size, voxel_table, save_path)
fid = fopen(file_path, "w");
if fid < 0
    error("Failed to open %s for writing.", file_path);
end
fprintf(fid, "Hoffman compressed Geant4 macro generation\n");
fprintf(fid, "save_path = %s\n", save_path);
fprintf(fid, "volume_type = %s\n", cfg.volume_type);
fprintf(fid, "volume_size = %d x %d x %d\n", volume_size(1), volume_size(2), volume_size(3));
fprintf(fid, "pixel_size_mm = %.6f x %.6f x %.6f\n", pixel_size_mm(1), pixel_size_mm(2), pixel_size_mm(3));
fprintf(fid, "source_shape = %s\n", cfg.source_shape);
fprintf(fid, "rotate_num = %d\n", cfg.rotate_num);
fprintf(fid, "beam_on = %d\n", cfg.beam_on);
fprintf(fid, "activity_threshold = %.6g\n", cfg.activity_threshold);
fprintf(fid, "activity_power = %.6f\n", cfg.activity_power);
fprintf(fid, "voxel_sources = %d\n", size(voxel_table, 1));
fprintf(fid, "weight_sum = %.8f\n", sum(voxel_table(:, 4)));
fprintf(fid, "weight_max = %.8f\n", max(voxel_table(:, 4)));
fprintf(fid, "weight_min = %.8f\n", min(voxel_table(:, 4)));
fclose(fid);
end


function centers = voxel_centers(num_voxel, voxel_size)
centers = (-num_voxel / 2 + 0.5) * voxel_size : voxel_size : (num_voxel / 2 - 0.5) * voxel_size;
end
