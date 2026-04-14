%% BrainPhantom_Truncated_3D
% Generate a PET-like truncated brain phantom for the current reconstruction FOV.
% The phantom is built on a finer internal grid, then averaged back to the
% system grid (100 x 100 x 20) for later use in this project.

clear;
clc;

cfg.target_pixel_num = [100, 100, 20];
cfg.target_pixel_size_mm = [3, 3, 3];
cfg.fov_mm = cfg.target_pixel_num .* cfg.target_pixel_size_mm;

cfg.super_factor = [4, 4, 2];
cfg.native_pixel_num = cfg.target_pixel_num .* cfg.super_factor;
cfg.native_pixel_size_mm = cfg.fov_mm ./ cfg.native_pixel_num;

cfg.output_name = "TruncatedBrainPETLike_300x300x60";

script_dir = fileparts(mfilename("fullpath"));
output_dir = fullfile(script_dir, "Preview", cfg.output_name);
if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

xn = voxel_centers(cfg.native_pixel_num(1), cfg.native_pixel_size_mm(1));
yn = voxel_centers(cfg.native_pixel_num(2), cfg.native_pixel_size_mm(2));
zn = voxel_centers(cfg.native_pixel_num(3), cfg.native_pixel_size_mm(3));
[X, Y, Z] = ndgrid(xn, yn, zn);

activity_native = build_pet_like_brain(X, Y, Z);
display_native = build_display_volume(activity_native, X, Y, Z);

activity = block_average_downsample(activity_native, cfg.super_factor);
display_volume = block_average_downsample(display_native, cfg.super_factor);

xt = voxel_centers(cfg.target_pixel_num(1), cfg.target_pixel_size_mm(1));
yt = voxel_centers(cfg.target_pixel_num(2), cfg.target_pixel_size_mm(2));
zt = voxel_centers(cfg.target_pixel_num(3), cfg.target_pixel_size_mm(3));

brain_mask = activity > 0;
active_idx = find(brain_mask);
[ix, iy, iz] = ind2sub(size(activity), active_idx);
voxel_list = [xt(ix)', yt(iy)', zt(iz)', double(activity(active_idx))];

phantom = struct();
phantom.activity = activity;
phantom.display_volume = display_volume;
phantom.mask = brain_mask;
phantom.x_mm = xt;
phantom.y_mm = yt;
phantom.z_mm = zt;
phantom.voxel_size_mm = cfg.target_pixel_size_mm;
phantom.pixel_num = cfg.target_pixel_num;
phantom.activity_native = activity_native;
phantom.display_native = display_native;
phantom.x_native_mm = xn;
phantom.y_native_mm = yn;
phantom.z_native_mm = zn;
phantom.voxel_native_size_mm = cfg.native_pixel_size_mm;
phantom.pixel_native_num = cfg.native_pixel_num;
phantom.fov_mm = cfg.fov_mm;
phantom.super_factor = cfg.super_factor;
phantom.activity_labels = struct( ...
    "cortex", 1.00, ...
    "deep_gray", 0.90, ...
    "white_matter", 0.62, ...
    "cerebellum", 0.92, ...
    "brainstem", 0.55, ...
    "ventricles", 0.04, ...
    "sulci", 0.12);

save(fullfile(output_dir, "brain_phantom_truncated.mat"), "phantom", "voxel_list", "-v7.3");
writematrix(voxel_list, fullfile(output_dir, "brain_voxel_list.csv"));

axial_mip = max(display_native, [], 3);
coronal_mip = squeeze(max(display_native, [], 2))';
sagittal_mip = squeeze(max(display_native, [], 1))';
axial_center = display_native(:, :, ceil(cfg.native_pixel_num(3) / 2));

fig = figure("Color", "w", "Position", [100, 100, 1380, 380]);
tiledlayout(1, 4, "TileSpacing", "compact", "Padding", "compact");

nexttile;
imagesc(yn, xn, axial_mip);
axis image;
set(gca, "YDir", "normal");
title("Axial MIP");
xlabel("Y / mm");
ylabel("X / mm");
colorbar;

nexttile;
imagesc(yn, zn, coronal_mip);
axis image;
set(gca, "YDir", "normal");
title("Coronal MIP");
xlabel("Y / mm");
ylabel("Z / mm");
colorbar;

nexttile;
imagesc(xn, zn, sagittal_mip);
axis image;
set(gca, "YDir", "normal");
title("Sagittal MIP");
xlabel("X / mm");
ylabel("Z / mm");
colorbar;

nexttile;
imagesc(yn, xn, axial_center);
axis image;
set(gca, "YDir", "normal");
title("Center Axial Slice");
xlabel("Y / mm");
ylabel("X / mm");
colorbar;

colormap(flipud(gray(256)));
set(findall(fig, "Type", "axes"), "CLim", [0, 1]);
saveas(fig, fullfile(output_dir, "brain_phantom_preview.png"));
savefig(fig, fullfile(output_dir, "brain_phantom_preview.fig"));

fprintf("Output directory: %s\n", output_dir);
fprintf("FOV: %.0f x %.0f x %.0f mm\n", cfg.fov_mm(1), cfg.fov_mm(2), cfg.fov_mm(3));
fprintf("Target grid: %d x %d x %d\n", cfg.target_pixel_num(1), cfg.target_pixel_num(2), cfg.target_pixel_num(3));
fprintf("Native grid: %d x %d x %d\n", cfg.native_pixel_num(1), cfg.native_pixel_num(2), cfg.native_pixel_num(3));
fprintf("Active target voxels: %d / %d (%.2f%%)\n", numel(active_idx), numel(activity), 100 * numel(active_idx) / numel(activity));
fprintf("Activity range: %.2f to %.2f\n", min(activity(active_idx)), max(activity(active_idx)));


function activity = build_pet_like_brain(X, Y, Z)
activity = zeros(size(X), "single");

Yc = Y - 6;
z_scale = sqrt(max(0.20, 1 - (Z ./ 28) .^ 2));
Xs = X ./ z_scale;
Ys = Yc ./ z_scale;

theta = atan2(Ys, Xs);
r_xy = sqrt(Xs .^ 2 + Ys .^ 2);

outer_r = 1 ./ sqrt((cos(theta) ./ 71) .^ 2 + (sin(theta) ./ 89) .^ 2);
outer_r = outer_r .* (1 + 0.050 * cos(theta) .^ 2);
outer_r = outer_r .* (1 + 0.018 * sin(2 * theta) + 0.012 * sin(5 * theta + 0.05 * Z));
outer_r = outer_r .* (1 - 0.11 * exp(-((theta - pi / 2) / 0.55) .^ 2));
outer_r = outer_r .* (1 - 0.08 * exp(-((theta + pi / 2) / 0.60) .^ 2));
outer_r = outer_r .* (1 - 0.05 * exp(-((abs(theta) - pi) / 0.42) .^ 2));

outer_brain = r_xy <= outer_r;
outer_brain((Ys < -71) & (abs(Xs) > 18)) = false;
outer_brain((abs(Xs) > 59) & (Ys < -8)) = false;
outer_brain((abs(Xs) > 44) & (Ys > 67)) = false;
outer_brain((abs(Xs) > 36) & (Ys < -64)) = false;

inner_ratio = 0.665 ...
    + 0.035 * sin(4 * theta + 0.08 * Z) ...
    + 0.022 * sin(9 * theta - 0.10 * Y) ...
    + 0.018 * cos(13 * theta + 0.06 * X);
inner_ratio = min(max(inner_ratio, 0.60), 0.76);
inner_r = outer_r .* inner_ratio;

white_matter = outer_brain & (r_xy <= inner_r);
gray_matter = outer_brain & ~white_matter;

% Richer cortical folding pattern for the central axial slice.
gyri_phase = theta ...
    + 0.16 * sin(2 * theta) ...
    + 0.11 * sin(3 * theta + 0.05 * Z) ...
    + 0.03 * X / 30;
outer_band = outer_r - inner_r;
radial_frac = (r_xy - inner_r) ./ max(outer_band, 1e-6);

sulci_a = sin(10 * gyri_phase + 0.25 * Z + 0.04 * r_xy);
sulci_b = sin(17 * gyri_phase - 0.18 * Z + 0.06 * Ys);
sulci_c = cos(24 * gyri_phase + 0.03 * X - 0.06 * Y);
sulci_mix = 0.55 * sulci_a + 0.30 * sulci_b + 0.15 * sulci_c;

sulci_outer = gray_matter & radial_frac > 0.64 & sulci_mix > 0.48;
sulci_inner = gray_matter & radial_frac > 0.50 & radial_frac <= 0.64 & sulci_mix > 0.86;
sulci = sulci_outer | sulci_inner;

% Sylvian fissures and insular region.
left_sylvian = in_rotated_ellipsoid(X, Y, Z, -34, -4, -1, 16, 5.8, 7, deg2rad(-18));
right_sylvian = in_rotated_ellipsoid(X, Y, Z, 34, -4, -1, 16, 5.8, 7, deg2rad(18));
sylvian = left_sylvian | right_sylvian;
insular_white_left = in_rotated_ellipsoid(X, Y, Z, -31, -1, -1, 13, 10, 8, deg2rad(-18));
insular_white_right = in_rotated_ellipsoid(X, Y, Z, 31, -1, -1, 13, 10, 8, deg2rad(18));

% Butterfly-like ventricles.
left_vent_body = in_rotated_ellipsoid(X, Y, Z, -9, -2, 0, 4.0, 13.0, 6.5, deg2rad(-8));
right_vent_body = in_rotated_ellipsoid(X, Y, Z, 9, -2, 0, 4.0, 13.0, 6.5, deg2rad(8));
left_ant_horn = in_rotated_ellipsoid(X, Y, Z, -11.5, 14, 0, 2.6, 8.0, 4.2, deg2rad(-30));
right_ant_horn = in_rotated_ellipsoid(X, Y, Z, 11.5, 14, 0, 2.6, 8.0, 4.2, deg2rad(30));
left_inf_horn = in_rotated_ellipsoid(X, Y, Z, -20, -16, -2, 3.0, 8.5, 4.5, deg2rad(-46));
right_inf_horn = in_rotated_ellipsoid(X, Y, Z, 20, -16, -2, 3.0, 8.5, 4.5, deg2rad(46));
third_vent = in_ellipsoid(X, Y, Z, 0, -7, 0, 2.0, 7.0, 5.0);
ventricles = left_vent_body | right_vent_body | left_ant_horn | right_ant_horn | left_inf_horn | right_inf_horn | third_vent;

% Deep gray nuclei.
left_thalamus = in_rotated_ellipsoid(X, Y, Z, -9, -24, 0, 7.0, 11.0, 7, deg2rad(-18));
right_thalamus = in_rotated_ellipsoid(X, Y, Z, 9, -24, 0, 7.0, 11.0, 7, deg2rad(18));
left_caudate = in_rotated_ellipsoid(X, Y, Z, -13, 3, 0, 4.0, 9.0, 5.5, deg2rad(-17));
right_caudate = in_rotated_ellipsoid(X, Y, Z, 13, 3, 0, 4.0, 9.0, 5.5, deg2rad(17));
left_putamen = in_rotated_ellipsoid(X, Y, Z, -22, -7, 0, 5.0, 10.0, 5.5, deg2rad(-23));
right_putamen = in_rotated_ellipsoid(X, Y, Z, 22, -7, 0, 5.0, 10.0, 5.5, deg2rad(23));
deep_gray = left_thalamus | right_thalamus | left_caudate | right_caudate | left_putamen | right_putamen;

% Interhemispheric fissure.
midline = (abs(X) <= 1.2) & (Y > 10) & (abs(Z) <= 16);

gray_matter(sulci | sylvian | midline) = false;
white_matter(ventricles) = false;
white_matter(insular_white_left | insular_white_right) = true;
white_matter(deep_gray) = false;

% Truncated lower structures.
cerebellum_left = in_ellipsoid(X, Y, Z, -23, -61, -17, 19, 16, 7.5);
cerebellum_right = in_ellipsoid(X, Y, Z, 23, -61, -17, 19, 16, 7.5);
cerebellum_mid = in_ellipsoid(X, Y, Z, 0, -63, -17, 9, 12, 6.5);
cerebellum = cerebellum_left | cerebellum_right | cerebellum_mid;
cere_rib = sin(0.28 * X + 0.52 * atan2(Z + 17, X + eps) * 8) > 0.80;
cere_sulci = cerebellum & cere_rib;
cerebellum(cere_sulci) = false;
brainstem = in_ellipsoid(X, Y, Z, 0, -38, -19, 9.5, 12, 8);

gray_matter = gray_matter & ~ventricles & ~deep_gray & ~cerebellum & ~brainstem;
white_matter = white_matter & ~ventricles & ~deep_gray;
deep_gray = deep_gray & outer_brain & ~ventricles;

activity(gray_matter) = 1.00;
activity(white_matter) = 0.62;
activity(deep_gray) = 0.90;
activity(cerebellum) = 0.92;
activity(brainstem) = 0.55;
activity(ventricles) = 0.04;
activity(sulci) = 0.12;
activity(sylvian) = 0.08;
activity(midline) = 0.05;
end


function display_volume = build_display_volume(activity, X, Y, Z)
display_volume = activity;

Yc = Y - 6;
z_scale = sqrt(max(0.20, 1 - (Z ./ 28) .^ 2));
Xs = X ./ z_scale;
Ys = Yc ./ z_scale;
theta = atan2(Ys, Xs);
r_xy = sqrt(Xs .^ 2 + Ys .^ 2);

outer_r = 1 ./ sqrt((cos(theta) ./ 71) .^ 2 + (sin(theta) ./ 89) .^ 2);
outer_r = outer_r .* (1 + 0.050 * cos(theta) .^ 2);
outer_r = outer_r .* (1 + 0.018 * sin(2 * theta) + 0.012 * sin(5 * theta + 0.05 * Z));
outer_r = outer_r .* (1 - 0.11 * exp(-((theta - pi / 2) / 0.55) .^ 2));
outer_r = outer_r .* (1 - 0.08 * exp(-((theta + pi / 2) / 0.60) .^ 2));
outer_r = outer_r .* (1 - 0.05 * exp(-((abs(theta) - pi) / 0.42) .^ 2));
outer_brain = r_xy <= outer_r;

head_outline = (r_xy <= outer_r * 1.12) & ~outer_brain & (abs(Z) <= 24);
display_volume(head_outline) = max(display_volume(head_outline), 0.18);
end


function volume_ds = block_average_downsample(volume, factor)
sz = size(volume);
sx = factor(1);
sy = factor(2);
szf = factor(3);

volume_rs = reshape(volume, sx, sz(1) / sx, sy, sz(2) / sy, szf, sz(3) / szf);
volume_ds = squeeze(mean(mean(mean(volume_rs, 1), 3), 5));
end


function centers = voxel_centers(num_voxel, voxel_size)
centers = (-num_voxel / 2 + 0.5) * voxel_size : voxel_size : (num_voxel / 2 - 0.5) * voxel_size;
end


function mask = in_ellipsoid(X, Y, Z, xc, yc, zc, ax, ay, az)
mask = ((X - xc) ./ ax) .^ 2 + ((Y - yc) ./ ay) .^ 2 + ((Z - zc) ./ az) .^ 2 <= 1;
end


function mask = in_rotated_ellipsoid(X, Y, Z, xc, yc, zc, ax, ay, az, phi)
xp = (X - xc) * cos(phi) + (Y - yc) * sin(phi);
yp = -(X - xc) * sin(phi) + (Y - yc) * cos(phi);
zp = Z - zc;
mask = (xp ./ ax) .^ 2 + (yp ./ ay) .^ 2 + (zp ./ az) .^ 2 <= 1;
end
