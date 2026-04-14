%% BrainPhantom_SliceStack_3D
% Build a truncated brain phantom from an image-driven stack of fine axial slices.
% A reference FDG-like axial slice is processed into a high-resolution central
% slice, then smoothly deformed along z to form a 3D volume. Finally, the
% volume is block-averaged to the system grid (100 x 100 x 20).

clear;
clc;

cfg.target_pixel_num = [100, 100, 20];
cfg.target_pixel_size_mm = [3, 3, 3];
cfg.fov_mm = cfg.target_pixel_num .* cfg.target_pixel_size_mm;

cfg.super_factor = [4, 4, 2];
cfg.native_pixel_num = cfg.target_pixel_num .* cfg.super_factor;
cfg.native_pixel_size_mm = cfg.fov_mm ./ cfg.native_pixel_num;

cfg.output_name = "TruncatedBrainSliceStack_300x300x60";

script_dir = fileparts(mfilename("fullpath"));
repo_dir = fileparts(script_dir);
output_dir = fullfile(script_dir, "Preview", cfg.output_name);
if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

ref_path = fullfile(repo_dir, "figure.png");
if ~exist(ref_path, "file")
    error("Reference image not found: %s", ref_path);
end

native_xy = cfg.native_pixel_num(1:2);
template = prepare_reference_slice(ref_path, native_xy);

activity_native = build_slice_stack(template, cfg.native_pixel_num(3));
display_native = build_display_volume(activity_native);

activity = block_average_downsample(activity_native, cfg.super_factor);
display_volume = block_average_downsample(display_native, cfg.super_factor);

xt = voxel_centers(cfg.target_pixel_num(1), cfg.target_pixel_size_mm(1));
yt = voxel_centers(cfg.target_pixel_num(2), cfg.target_pixel_size_mm(2));
zt = voxel_centers(cfg.target_pixel_num(3), cfg.target_pixel_size_mm(3));

brain_mask = activity > 1e-3;
active_idx = find(brain_mask);
[ix, iy, iz] = ind2sub(size(activity), active_idx);
voxel_list = [xt(ix)', yt(iy)', zt(iz)', double(activity(active_idx))];

phantom = struct();
phantom.activity = activity;
phantom.display_volume = display_volume;
phantom.mask = brain_mask;
phantom.reference_template = template;
phantom.activity_native = activity_native;
phantom.display_native = display_native;
phantom.voxel_list = voxel_list;
phantom.target_pixel_num = cfg.target_pixel_num;
phantom.target_pixel_size_mm = cfg.target_pixel_size_mm;
phantom.native_pixel_num = cfg.native_pixel_num;
phantom.native_pixel_size_mm = cfg.native_pixel_size_mm;
phantom.fov_mm = cfg.fov_mm;
phantom.super_factor = cfg.super_factor;
phantom.x_mm = xt;
phantom.y_mm = yt;
phantom.z_mm = zt;

save(fullfile(output_dir, "brain_phantom_slicestack.mat"), "phantom", "-v7.3");
writematrix(voxel_list, fullfile(output_dir, "brain_voxel_list.csv"));

write_raw_float32(fullfile(output_dir, "brain_phantom_target_float32.raw"), activity);
write_raw_float32(fullfile(output_dir, "brain_phantom_native_float32.raw"), activity_native);
write_mhd( ...
    fullfile(output_dir, "brain_phantom_target_float32.mhd"), ...
    "brain_phantom_target_float32.raw", ...
    cfg.target_pixel_num, ...
    cfg.target_pixel_size_mm);
write_mhd( ...
    fullfile(output_dir, "brain_phantom_native_float32.mhd"), ...
    "brain_phantom_native_float32.raw", ...
    cfg.native_pixel_num, ...
    cfg.native_pixel_size_mm);
write_info_txt( ...
    fullfile(output_dir, "brain_phantom_raw_info.txt"), ...
    cfg.target_pixel_num, ...
    cfg.target_pixel_size_mm, ...
    cfg.native_pixel_num, ...
    cfg.native_pixel_size_mm);

native_x = voxel_centers(cfg.native_pixel_num(1), cfg.native_pixel_size_mm(1));
native_y = voxel_centers(cfg.native_pixel_num(2), cfg.native_pixel_size_mm(2));
native_z = voxel_centers(cfg.native_pixel_num(3), cfg.native_pixel_size_mm(3));

axial_mip = max(display_native, [], 3);
coronal_mip = squeeze(max(display_native, [], 2))';
sagittal_mip = squeeze(max(display_native, [], 1))';
axial_center = display_native(:, :, ceil(cfg.native_pixel_num(3) / 2));

fig = figure("Color", "w", "Position", [100, 100, 1380, 380]);
tiledlayout(1, 4, "TileSpacing", "compact", "Padding", "compact");

nexttile;
imagesc(native_y, native_x, axial_mip);
axis image;
set(gca, "YDir", "normal");
title("Axial MIP");
xlabel("Y / mm");
ylabel("X / mm");
colorbar;

nexttile;
imagesc(native_y, native_z, coronal_mip);
axis image;
set(gca, "YDir", "normal");
title("Coronal MIP");
xlabel("Y / mm");
ylabel("Z / mm");
colorbar;

nexttile;
imagesc(native_x, native_z, sagittal_mip);
axis image;
set(gca, "YDir", "normal");
title("Sagittal MIP");
xlabel("X / mm");
ylabel("Z / mm");
colorbar;

nexttile;
imagesc(native_y, native_x, axial_center);
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
fprintf("Reference image: %s\n", ref_path);
fprintf("FOV: %.0f x %.0f x %.0f mm\n", cfg.fov_mm(1), cfg.fov_mm(2), cfg.fov_mm(3));
fprintf("Target grid: %d x %d x %d\n", cfg.target_pixel_num(1), cfg.target_pixel_num(2), cfg.target_pixel_num(3));
fprintf("Native grid: %d x %d x %d\n", cfg.native_pixel_num(1), cfg.native_pixel_num(2), cfg.native_pixel_num(3));
fprintf("Active target voxels: %d / %d (%.2f%%)\n", numel(active_idx), numel(activity), 100 * numel(active_idx) / numel(activity));
fprintf("Activity range: %.2f to %.2f\n", min(activity(active_idx)), max(activity(active_idx)));


function template = prepare_reference_slice(ref_path, out_size)
img = imread(ref_path);
if ndims(img) == 3
    gray = rgb2gray(img);
else
    gray = img;
end
gray = im2double(gray);

mask = gray < 0.97;
mask(1:round(0.15 * size(mask, 1)), :) = false;
mask(:, 1:round(0.03 * size(mask, 2))) = false;

cc = bwconncomp(mask);
if cc.NumObjects == 0
    error("Failed to detect a brain region from the reference image.");
end

stats = regionprops(cc, "Area", "BoundingBox");
[~, idx] = max([stats.Area]);
bbox = stats(idx).BoundingBox;

pad = 18;
r1 = max(1, floor(bbox(2)) - pad);
r2 = min(size(gray, 1), ceil(bbox(2) + bbox(4)) + pad);
c1 = max(1, floor(bbox(1)) - pad);
c2 = min(size(gray, 2), ceil(bbox(1) + bbox(3)) + pad);
crop = gray(r1:r2, c1:c2);

side = max(size(crop, 1), size(crop, 2));
canvas = ones(side, side);
row0 = floor((side - size(crop, 1)) / 2) + 1;
col0 = floor((side - size(crop, 2)) / 2) + 1;
canvas(row0:row0 + size(crop, 1) - 1, col0:col0 + size(crop, 2) - 1) = crop;

resized = imresize(canvas, out_size, "bicubic");
inv_img = 1 - resized;
inv_img = inv_img / max(inv_img(:));
inv_img = imgaussfilt(inv_img, 0.7);

head_seed = inv_img > 0.025;
head_mask = largest_filled_component(head_seed);
dist_to_bg = bwdist(~head_mask);
ring_mask = head_mask & (dist_to_bg <= 7) & (inv_img < 0.22);
brain_mask = head_mask & ~ring_mask;
brain_mask = imfill(brain_mask, "holes");
brain_mask = largest_filled_component(brain_mask);

activity = inv_img .* brain_mask;
activity = imadjust(activity, [0.03 0.92], [0 1], 0.90);

template = struct();
template.activity = min(max(activity, 0), 1);
template.brain_mask = brain_mask;
template.head_mask = head_mask;
template.ref_crop = canvas;
end


function volume = build_slice_stack(template, nz)
base = template.activity;
brain_mask = template.brain_mask;

[ny, nx] = size(base);
x = linspace(-1, 1, nx);
y = linspace(-1, 1, ny);
[X, Y] = meshgrid(x, y);

base_soft = imgaussfilt(base, 1.6);
volume = zeros(ny, nx, nz, "single");

for k = 1:nz
    zeta = (k - (nz + 1) / 2) / ((nz - 1) / 2);
    az = abs(zeta);

    scale_x = 0.78 + 0.22 * (1 - az ^ 1.55);
    scale_y = 0.72 + 0.28 * (1 - az ^ 1.75);
    shift_y = 0.02 * zeta - 0.05 * zeta ^ 2;

    warped = warp_image(base, scale_x, scale_y, 0, shift_y);
    warped_mask = warp_image(single(brain_mask), scale_x, scale_y, 0, shift_y) > 0.15;

    yy = Y - shift_y;
    superior_taper = 1 - 0.78 * max(zeta, 0) .* sigmoid((yy - 0.18) / 0.07);
    inferior_taper = 1 - 0.55 * max(-zeta, 0) .* sigmoid((-yy - 0.34) / 0.08);
    lateral_taper = 1 - 0.18 * az .* sigmoid((abs(X) - 0.76) / 0.05);
    taper = superior_taper .* inferior_taper .* lateral_taper;

    detail_mix = max(0.35, 1 - 0.90 * az);
    slice = detail_mix .* warped + (1 - detail_mix) .* imgaussfilt(warped, 2.2);
    slice = slice .* taper .* warped_mask;

    % Gradually contract low-activity cavities away from the center slice.
    vent_fill = 0.22 * az * exp(-((X / 0.12) .^ 2 + ((Y + 0.02) / 0.20) .^ 2));
    slice = max(slice, vent_fill .* warped_mask);

    % Smoothly fade the stack at the axial ends.
    axial_envelope = max(0, 1 - az ^ 2.2);
    slice = slice .* (0.38 + 0.62 * axial_envelope);

    % Keep the cortex sharper than the white matter.
    cortex_edge = max(0, warped - imgaussfilt(warped, 3.0));
    slice = slice + 0.28 * (1 - az) * cortex_edge;

    slice = min(max(slice, 0), 1);
    slice(~warped_mask) = 0;
    volume(:, :, k) = single(slice);
end

% Mix in a broad smooth component for 3D continuity.
for k = 1:nz
    volume(:, :, k) = 0.82 * volume(:, :, k) + 0.18 * base_soft .* max(volume(:, :, k) > 0.02, 0);
end
volume = min(max(volume, 0), 1);
end


function display_volume = build_display_volume(activity_volume)
display_volume = activity_volume;
for k = 1:size(activity_volume, 3)
    slice = activity_volume(:, :, k);
    head_mask = imfill(slice > 0.015, "holes");
    if any(head_mask(:))
        dist_to_bg = bwdist(~head_mask);
        outline = head_mask & dist_to_bg <= 6 & slice < 0.18;
        display_volume(:, :, k) = max(display_volume(:, :, k), 0.18 * outline);
    end
end
end


function warped = warp_image(img, sx, sy, tx, ty)
[ny, nx] = size(img);
x = linspace(-1, 1, nx);
y = linspace(-1, 1, ny);
[X, Y] = meshgrid(x, y);

Xs = (X - tx) / sx;
Ys = (Y - ty) / sy;
warped = interp2(x, y, img, Xs, Ys, "linear", 0);
end


function mask = largest_filled_component(seed)
seed = logical(seed);
cc = bwconncomp(seed);
if cc.NumObjects == 0
    mask = false(size(seed));
    return;
end
areas = cellfun(@numel, cc.PixelIdxList);
[~, idx] = max(areas);
mask = false(size(seed));
mask(cc.PixelIdxList{idx}) = true;
mask = imfill(mask, "holes");
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


function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end


function write_raw_float32(file_path, volume)
fid = fopen(file_path, "w", "ieee-le");
if fid < 0
    error("Failed to open %s for writing.", file_path);
end
count = fwrite(fid, single(volume), "single");
fclose(fid);
if count ~= numel(volume)
    error("Failed to write the full volume to %s.", file_path);
end
end


function write_mhd(file_path, raw_name, dims_xyz, spacing_xyz)
fid = fopen(file_path, "w");
if fid < 0
    error("Failed to open %s for writing.", file_path);
end
fprintf(fid, "ObjectType = Image\n");
fprintf(fid, "NDims = 3\n");
fprintf(fid, "BinaryData = True\n");
fprintf(fid, "BinaryDataByteOrderMSB = False\n");
fprintf(fid, "CompressedData = False\n");
fprintf(fid, "TransformMatrix = 1 0 0 0 1 0 0 0 1\n");
fprintf(fid, "Offset = 0 0 0\n");
fprintf(fid, "CenterOfRotation = 0 0 0\n");
fprintf(fid, "AnatomicalOrientation = RAI\n");
fprintf(fid, "ElementSpacing = %.6f %.6f %.6f\n", spacing_xyz(1), spacing_xyz(2), spacing_xyz(3));
fprintf(fid, "DimSize = %d %d %d\n", dims_xyz(1), dims_xyz(2), dims_xyz(3));
fprintf(fid, "ElementType = MET_FLOAT\n");
fprintf(fid, "ElementDataFile = %s\n", raw_name);
fclose(fid);
end


function write_info_txt(file_path, target_dims, target_spacing, native_dims, native_spacing)
fid = fopen(file_path, "w");
if fid < 0
    error("Failed to open %s for writing.", file_path);
end
fprintf(fid, "Target float32 raw\n");
fprintf(fid, "  file: brain_phantom_target_float32.raw\n");
fprintf(fid, "  dims: %d x %d x %d\n", target_dims(1), target_dims(2), target_dims(3));
fprintf(fid, "  spacing_mm: %.6f x %.6f x %.6f\n", target_spacing(1), target_spacing(2), target_spacing(3));
fprintf(fid, "  byte_order: little-endian\n");
fprintf(fid, "  data_type: float32\n");
fprintf(fid, "  memory_order: MATLAB linear order for array size [X, Y, Z]\n");
fprintf(fid, "\n");
fprintf(fid, "Native float32 raw\n");
fprintf(fid, "  file: brain_phantom_native_float32.raw\n");
fprintf(fid, "  dims: %d x %d x %d\n", native_dims(1), native_dims(2), native_dims(3));
fprintf(fid, "  spacing_mm: %.6f x %.6f x %.6f\n", native_spacing(1), native_spacing(2), native_spacing(3));
fprintf(fid, "  byte_order: little-endian\n");
fprintf(fid, "  data_type: float32\n");
fprintf(fid, "  memory_order: MATLAB linear order for array size [X, Y, Z]\n");
fprintf(fid, "\n");
fprintf(fid, "MetaImage helper files are also provided for direct loading when supported.\n");
fclose(fid);
end
