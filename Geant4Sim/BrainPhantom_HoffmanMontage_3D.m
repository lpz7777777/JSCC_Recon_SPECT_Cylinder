%% BrainPhantom_HoffmanMontage_3D
% Reconstruct a Hoffman-like truncated brain phantom from a montage image
% containing axial slices arranged left-to-right, top-to-bottom.

clear;
clc;

cfg.target_pixel_num = [100, 100, 20];
cfg.target_pixel_size_mm = [3, 3, 3];
cfg.fov_mm = cfg.target_pixel_num .* cfg.target_pixel_size_mm;

cfg.super_factor = [4, 4, 2];
cfg.native_pixel_num = cfg.target_pixel_num .* cfg.super_factor;
cfg.native_pixel_size_mm = cfg.fov_mm ./ cfg.native_pixel_num;

cfg.output_name = "HoffmanMontage_300x300x60";

script_dir = fileparts(mfilename("fullpath"));
repo_dir = fileparts(script_dir);
output_dir = fullfile(script_dir, "Preview", cfg.output_name);
if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

ref_path = fullfile(repo_dir, "figure2.png");
if ~exist(ref_path, "file")
    error("Reference montage image not found: %s", ref_path);
end

montage_gray = read_gray_image(ref_path);
[tiles, tile_info] = extract_montage_tiles(montage_gray);
slice_bank = clean_hoffman_tiles(tiles, cfg.native_pixel_num(1:2));

activity_native = interpolate_slice_stack(slice_bank, cfg.native_pixel_num(3));
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
phantom.activity_native = activity_native;
phantom.display_native = display_native;
phantom.slice_bank = slice_bank;
phantom.tile_info = tile_info;
phantom.target_pixel_num = cfg.target_pixel_num;
phantom.target_pixel_size_mm = cfg.target_pixel_size_mm;
phantom.native_pixel_num = cfg.native_pixel_num;
phantom.native_pixel_size_mm = cfg.native_pixel_size_mm;
phantom.fov_mm = cfg.fov_mm;
phantom.super_factor = cfg.super_factor;
phantom.x_mm = xt;
phantom.y_mm = yt;
phantom.z_mm = zt;

save(fullfile(output_dir, "brain_phantom_hoffman.mat"), "phantom", "voxel_list", "-v7.3");
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

save_clean_slice_montage(fullfile(output_dir, "hoffman_clean_slices.png"), slice_bank);
save_volume_preview(fullfile(output_dir, "brain_phantom_preview.png"), display_native, cfg.native_pixel_num, cfg.native_pixel_size_mm);

fprintf("Output directory: %s\n", output_dir);
fprintf("Reference montage: %s\n", ref_path);
fprintf("Detected source slices: %d\n", size(slice_bank, 3));
fprintf("Target grid: %d x %d x %d\n", cfg.target_pixel_num(1), cfg.target_pixel_num(2), cfg.target_pixel_num(3));
fprintf("Native grid: %d x %d x %d\n", cfg.native_pixel_num(1), cfg.native_pixel_num(2), cfg.native_pixel_num(3));
fprintf("Active target voxels: %d / %d (%.2f%%)\n", numel(active_idx), numel(activity), 100 * numel(active_idx) / numel(activity));
fprintf("Activity range: %.2f to %.2f\n", min(activity(active_idx)), max(activity(active_idx)));


function img = read_gray_image(path_in)
img = imread(path_in);
if ndims(img) == 3
    img = rgb2gray(img);
end
img = im2double(img);
end


function [tiles, info] = extract_montage_tiles(img)
row_mean = mean(img, 2);
col_mean = mean(img, 1);

row_groups = contiguous_groups(find(row_mean > 0.90));
col_groups = contiguous_groups(find(col_mean > 0.90));

row_edges = group_centers(row_groups);
col_edges = group_centers(col_groups);

row_bounds = intervals_from_edges(row_edges, size(img, 1));
col_bounds = intervals_from_edges(col_edges, size(img, 2));

tiles = {};
index = 0;
for r = 1:size(row_bounds, 1)
    for c = 1:size(col_bounds, 1)
        rr = row_bounds(r, 1):row_bounds(r, 2);
        cc = col_bounds(c, 1):col_bounds(c, 2);
        tile = img(rr, cc);
        r_core = round(size(tile, 1) * 0.20):round(size(tile, 1) * 0.80);
        c_core = round(size(tile, 2) * 0.20):round(size(tile, 2) * 0.80);
        core = tile(r_core, c_core);
        if max(core(:)) > 0.22 && mean(core(:)) > 0.01
            index = index + 1;
            tiles{index} = tile; %#ok<AGROW>
            info(index).row = r; %#ok<AGROW>
            info(index).col = c; %#ok<AGROW>
            info(index).rows = [rr(1), rr(end)]; %#ok<AGROW>
            info(index).cols = [cc(1), cc(end)]; %#ok<AGROW>
        end
    end
end
end


function slices = clean_hoffman_tiles(tiles, out_xy)
num_tiles = numel(tiles);
slices = zeros(out_xy(1), out_xy(2), num_tiles, "single");

for k = 1:num_tiles
    tile = tiles{k};

    % Remove frame-like dark border inside each cell.
    tile = tile(8:end-8, 8:end-8);
    tile = imgaussfilt(tile, 0.7);

    bg = median(tile(:));
    tile = max(tile - bg, 0);
    if max(tile(:)) > 0
        tile = tile / max(tile(:));
    end

    seed = tile > max(0.08, graythresh(tile) * 0.80);
    seed = imopen(seed, strel("disk", 2, 0));
    seed = imclose(seed, strel("disk", 5, 0));
    seed = select_center_component(seed);

    if ~any(seed(:))
        continue;
    end

    stats = regionprops(seed, "BoundingBox");
    bbox = stats(1).BoundingBox;
    pad = 8;
    r1 = max(1, floor(bbox(2)) - pad);
    r2 = min(size(tile, 1), ceil(bbox(2) + bbox(4)) + pad);
    c1 = max(1, floor(bbox(1)) - pad);
    c2 = min(size(tile, 2), ceil(bbox(1) + bbox(3)) + pad);
    crop = tile(r1:r2, c1:c2);

    side = max(size(crop, 1), size(crop, 2));
    canvas = zeros(side, side);
    row0 = floor((side - size(crop, 1)) / 2) + 1;
    col0 = floor((side - size(crop, 2)) / 2) + 1;
    canvas(row0:row0 + size(crop, 1) - 1, col0:col0 + size(crop, 2) - 1) = crop;

    resized = imresize(canvas, out_xy, "bicubic");
    resized = imgaussfilt(resized, 0.8);

    resized = resized - quantile(resized(:), 0.08);
    resized = max(resized, 0);
    if max(resized(:)) > 0
        resized = resized / max(resized(:));
    end

    mask = imgaussfilt(resized, 1.0) > 0.06;
    mask = imopen(mask, strel("disk", 2, 0));
    mask = imclose(mask, strel("disk", 5, 0));
    mask = select_center_component(mask);
    mask = imfill(mask, "holes");

    resized = imadjust(resized, [0.03 0.95], [0 1], 0.85);
    resized = 0.5 * (resized + fliplr(resized));
    resized = imsharpen(resized, "Radius", 2.0, "Amount", 1.2);
    resized = imgaussfilt(resized, 0.5);
    resized = resized .* imgaussfilt(double(mask), 1.2);
    resized(~mask) = 0;
    resized(resized < 0.03) = 0;
    resized = min(max(resized, 0), 1);

    slices(:, :, k) = single(resized);
end
end


function volume = interpolate_slice_stack(slice_bank, nz)
[nx, ny, ns] = size(slice_bank);
z_src = linspace(-1, 1, ns);
z_dst = linspace(-1, 1, nz);

volume = zeros(nx, ny, nz, "single");
for ix = 1:nx
    line = squeeze(slice_bank(ix, :, :));
    interp_line = interp1(z_src, line', z_dst, "pchip", "extrap")';
    volume(ix, :, :) = single(interp_line);
end

% Smooth along z and apply gentle axial taper using montage endpoints.
for ix = 1:nx
    for iy = 1:ny
        profile = squeeze(volume(ix, iy, :));
        profile = smoothdata(profile, "gaussian", 5);
        volume(ix, iy, :) = profile;
    end
end

[X, Y, Z] = ndgrid(linspace(-1, 1, nx), linspace(-1, 1, ny), z_dst);
axial_taper = 0.42 + 0.58 * (1 - abs(Z) .^ 1.8);
volume = volume .* axial_taper;

% Keep head extent truncated to the current FOV.
xy_mask = max(slice_bank, [], 3) > 0.06;
xy_mask = imgaussfilt(double(xy_mask), 1.5) > 0.12;
xy_mask = imfill(xy_mask, "holes");
xy_mask = largest_filled_component(xy_mask);
for k = 1:nz
    volume(:, :, k) = volume(:, :, k) .* single(xy_mask);
end

volume = min(max(volume, 0), 1);
end


function display_volume = build_display_volume(activity_volume)
display_volume = activity_volume;
for k = 1:size(activity_volume, 3)
    slice = activity_volume(:, :, k);
    head_mask = imfill(slice > 0.02, "holes");
    head_mask = largest_filled_component(head_mask);
    if any(head_mask(:))
        dist_to_bg = bwdist(~head_mask);
        outline = head_mask & dist_to_bg <= 5 & slice < 0.20;
        display_volume(:, :, k) = max(display_volume(:, :, k), 0.18 * outline);
    end
end
end


function save_clean_slice_montage(path_out, slice_bank)
num_tiles = size(slice_bank, 3);
cols = 5;
rows = ceil(num_tiles / cols);
cell_h = size(slice_bank, 1);
cell_w = size(slice_bank, 2);
gap = 10;

canvas = ones(rows * cell_h + (rows + 1) * gap, cols * cell_w + (cols + 1) * gap, "single");
for k = 1:num_tiles
    r = floor((k - 1) / cols) + 1;
    c = mod(k - 1, cols) + 1;
    r0 = gap + (r - 1) * (cell_h + gap) + 1;
    c0 = gap + (c - 1) * (cell_w + gap) + 1;
    canvas(r0:r0 + cell_h - 1, c0:c0 + cell_w - 1) = 1 - slice_bank(:, :, k);
end
imwrite(canvas, path_out);
end


function save_volume_preview(path_out, volume, native_num, native_spacing)
x = voxel_centers(native_num(1), native_spacing(1));
y = voxel_centers(native_num(2), native_spacing(2));
z = voxel_centers(native_num(3), native_spacing(3));

axial_mip = max(volume, [], 3);
coronal_mip = squeeze(max(volume, [], 2))';
sagittal_mip = squeeze(max(volume, [], 1))';
axial_center = volume(:, :, ceil(native_num(3) / 2));

fig = figure("Color", "w", "Position", [100, 100, 1380, 380]);
tiledlayout(1, 4, "TileSpacing", "compact", "Padding", "compact");

nexttile;
imagesc(y, x, axial_mip);
axis image;
set(gca, "YDir", "normal");
title("Axial MIP");
xlabel("Y / mm");
ylabel("X / mm");
colorbar;

nexttile;
imagesc(y, z, coronal_mip);
axis image;
set(gca, "YDir", "normal");
title("Coronal MIP");
xlabel("Y / mm");
ylabel("Z / mm");
colorbar;

nexttile;
imagesc(x, z, sagittal_mip);
axis image;
set(gca, "YDir", "normal");
title("Sagittal MIP");
xlabel("X / mm");
ylabel("Z / mm");
colorbar;

nexttile;
imagesc(y, x, axial_center);
axis image;
set(gca, "YDir", "normal");
title("Center Axial Slice");
xlabel("Y / mm");
ylabel("X / mm");
colorbar;

colormap(flipud(gray(256)));
set(findall(fig, "Type", "axes"), "CLim", [0, 1]);
saveas(fig, path_out);
savefig(fig, replace(path_out, ".png", ".fig"));
close(fig);
end


function groups = contiguous_groups(arr)
groups = zeros(0, 2);
if isempty(arr)
    return;
end
s = arr(1);
p = arr(1);
for i = 2:numel(arr)
    x = arr(i);
    if x ~= p + 1
        groups(end + 1, :) = [s, p]; %#ok<AGROW>
        s = x;
    end
    p = x;
end
groups(end + 1, :) = [s, p];
end


function centers = group_centers(groups)
centers = round(mean(groups, 2));
end


function bounds = intervals_from_edges(edges, max_idx)
edges = edges(:)';
bounds = zeros(numel(edges) - 1, 2);
for i = 1:numel(edges) - 1
    bounds(i, 1) = edges(i) + 1;
    bounds(i, 2) = edges(i + 1) - 1;
end
if bounds(1, 1) < 1
    bounds(1, 1) = 1;
end
if bounds(end, 2) > max_idx
    bounds(end, 2) = max_idx;
end
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


function mask = select_center_component(seed)
seed = logical(seed);
cc = bwconncomp(seed);
if cc.NumObjects == 0
    mask = false(size(seed));
    return;
end

stats = regionprops(cc, "Area", "Centroid");
center = ([size(seed, 2), size(seed, 1)] + 1) / 2;
best_score = -inf;
best_idx = 1;
for k = 1:numel(stats)
    dist = norm(stats(k).Centroid - center);
    score = stats(k).Area / (1 + 0.04 * dist ^ 2);
    if score > best_score
        best_score = score;
        best_idx = k;
    end
end

mask = false(size(seed));
mask(cc.PixelIdxList{best_idx}) = true;
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
