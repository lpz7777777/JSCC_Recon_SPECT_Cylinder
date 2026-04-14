%% BrainPhantom_HoffmanRawCompressed_3D
% Build a z-compressed Hoffman phantom from the downloaded raw YAFF volume.
% Rule: for each 10-slice block, exclude the special dark slices at positions
% 3 and 8, then keep one normal slice and stack the selected slices.

clear;
clc;

cfg.source_size = [256, 256, 250];
cfg.source_spacing_mm = [0.97656, 0.97656, 0.639 * 10];
cfg.group_size = 10;
cfg.exclude_pos_in_group = [3, 8];
cfg.prefer_pos_in_group = 5;

cfg.target_pixel_num = [100, 100, 20];
cfg.target_pixel_size_mm = [3, 3, 3];
cfg.target_fov_mm = cfg.target_pixel_num .* cfg.target_pixel_size_mm;
cfg.transverse_scale = 1.4;

cfg.output_name = "HoffmanRawCompressed_300x300x60_" + format_scale_tag(cfg.transverse_scale);

script_dir = fileparts(mfilename("fullpath"));
raw_dir = fullfile(script_dir, "3D_DRO_Hoffman_v6_raw");
raw_path = fullfile(raw_dir, "hoffman_phantom_3_31_2016_inv.yaff");
hdr_path = fullfile(raw_dir, "hoffman_phantom_3_31_2016_inv.yhdr");
output_dir = fullfile(script_dir, "Preview", cfg.output_name);
if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

if ~exist(raw_path, "file")
    error("Raw file not found: %s", raw_path);
end

source_volume_int16 = read_source_volume(raw_path, cfg.source_size);
source_volume = single(source_volume_int16);
if max(source_volume(:)) > 0
    source_volume = source_volume ./ max(source_volume(:));
end

[selected_volume, selected_indices, block_info] = compress_by_groups( ...
    source_volume, ...
    cfg.group_size, ...
    cfg.exclude_pos_in_group, ...
    cfg.prefer_pos_in_group);

selected_spacing_mm = cfg.source_spacing_mm;
selected_size = [cfg.source_size(1), cfg.source_size(2), size(selected_volume, 3)];

target_volume = resample_to_target_fov( ...
    selected_volume, ...
    selected_spacing_mm, ...
    cfg.target_pixel_num, ...
    cfg.target_pixel_size_mm, ...
    cfg.transverse_scale);
target_volume = min(max(target_volume, 0), 1);

target_mask = target_volume > 1e-4;
[ix, iy, iz] = ind2sub(size(target_volume), find(target_mask));
xt = voxel_centers(cfg.target_pixel_num(1), cfg.target_pixel_size_mm(1));
yt = voxel_centers(cfg.target_pixel_num(2), cfg.target_pixel_size_mm(2));
zt = voxel_centers(cfg.target_pixel_num(3), cfg.target_pixel_size_mm(3));
voxel_list = [xt(ix)', yt(iy)', zt(iz)', double(target_volume(target_mask))];

phantom = struct();
phantom.source_size = cfg.source_size;
phantom.source_spacing_mm = [0.97656, 0.97656, 0.639];
phantom.selected_indices = selected_indices;
phantom.block_info = block_info;
phantom.selected_volume = selected_volume;
phantom.selected_spacing_mm = selected_spacing_mm;
phantom.target_volume = target_volume;
phantom.target_pixel_num = cfg.target_pixel_num;
phantom.target_pixel_size_mm = cfg.target_pixel_size_mm;
phantom.target_fov_mm = cfg.target_fov_mm;
phantom.transverse_scale = cfg.transverse_scale;
phantom.voxel_list = voxel_list;
phantom.header_path = hdr_path;

save(fullfile(output_dir, "brain_phantom_hoffman_compressed.mat"), "phantom", "-v7.3");
writematrix(voxel_list, fullfile(output_dir, "brain_voxel_list.csv"));

write_raw_float32(fullfile(output_dir, "hoffman_compressed_native_float32.raw"), selected_volume);
write_raw_float32(fullfile(output_dir, "hoffman_compressed_target_float32.raw"), target_volume);
write_mhd( ...
    fullfile(output_dir, "hoffman_compressed_native_float32.mhd"), ...
    "hoffman_compressed_native_float32.raw", ...
    selected_size, ...
    selected_spacing_mm);
write_mhd( ...
    fullfile(output_dir, "hoffman_compressed_target_float32.mhd"), ...
    "hoffman_compressed_target_float32.raw", ...
    cfg.target_pixel_num, ...
    cfg.target_pixel_size_mm);

save_selected_slice_montage(fullfile(output_dir, "hoffman_selected_slices.png"), selected_volume);
save_volume_preview(fullfile(output_dir, "hoffman_compressed_preview.png"), selected_volume, selected_spacing_mm);
save_volume_preview(fullfile(output_dir, "hoffman_target_preview.png"), target_volume, cfg.target_pixel_size_mm);
write_summary_txt(fullfile(output_dir, "hoffman_compressed_info.txt"), cfg, selected_indices, selected_size);

fprintf("Output directory: %s\n", output_dir);
fprintf("Source raw: %s\n", raw_path);
fprintf("Selected slices: %d\n", numel(selected_indices));
fprintf("Selected indices (1-based): %s\n", mat2str(selected_indices));
fprintf("Native compressed size: %d x %d x %d\n", selected_size(1), selected_size(2), selected_size(3));
fprintf("Target size: %d x %d x %d\n", cfg.target_pixel_num(1), cfg.target_pixel_num(2), cfg.target_pixel_num(3));


function volume = read_source_volume(raw_path, sz)
fid = fopen(raw_path, "r", "ieee-le");
if fid < 0
    error("Failed to open %s.", raw_path);
end
volume = fread(fid, prod(sz), "int16=>int16");
fclose(fid);
if numel(volume) ~= prod(sz)
    error("Unexpected element count while reading %s.", raw_path);
end
volume = reshape(volume, sz);
end


function [selected_volume, selected_indices, block_info] = compress_by_groups(volume, group_size, exclude_pos, prefer_pos)
nz = size(volume, 3);
num_groups = floor(nz / group_size);
selected = [];
selected_indices = [];
block_info = struct("group_id", {}, "source_range", {}, "selected_index", {}, "selected_pos_in_group", {}, "valid_indices", {});

for g = 1:num_groups
    z0 = (g - 1) * group_size + 1;
    group_idx = z0:(z0 + group_size - 1);
    pos_in_group = 1:group_size;
    valid_mask = ~ismember(pos_in_group, exclude_pos);
    valid_idx = group_idx(valid_mask);

    area_score = squeeze(sum(sum(volume(:, :, valid_idx) > 0, 1), 2));
    nonempty = area_score > 0;
    valid_idx_nonempty = valid_idx(nonempty);

    if isempty(valid_idx_nonempty)
        continue;
    end

    preferred_global = z0 + prefer_pos - 1;
    if any(valid_idx_nonempty == preferred_global)
        chosen = preferred_global;
    else
        chosen = valid_idx_nonempty(ceil(numel(valid_idx_nonempty) / 2));
    end

    selected = cat(3, selected, volume(:, :, chosen));
    selected_indices(end + 1) = chosen; %#ok<AGROW>
    block_info(end + 1).group_id = g; %#ok<AGROW>
    block_info(end).source_range = [group_idx(1), group_idx(end)];
    block_info(end).selected_index = chosen;
    block_info(end).selected_pos_in_group = chosen - z0 + 1;
    block_info(end).valid_indices = valid_idx_nonempty;
end

selected_volume = selected;
end


function target_volume = resample_to_target_fov(source_volume, source_spacing_mm, target_num, target_spacing_mm, transverse_scale)
source_size = size(source_volume);
source_extent_mm = double(source_size) .* source_spacing_mm;

xs = voxel_centers(source_size(1), source_spacing_mm(1));
ys = voxel_centers(source_size(2), source_spacing_mm(2));
zs = voxel_centers(source_size(3), source_spacing_mm(3));

xt = voxel_centers(target_num(1), target_spacing_mm(1));
yt = voxel_centers(target_num(2), target_spacing_mm(2));
zt = voxel_centers(target_num(3), target_spacing_mm(3));

z_centered = linspace(-source_extent_mm(3) / 2 + source_spacing_mm(3) / 2, ...
    source_extent_mm(3) / 2 - source_spacing_mm(3) / 2, target_num(3));

[Xt, Yt, Zt] = ndgrid(xt, yt, zt);
Zs = Zt;
if source_extent_mm(3) > 0
    Zs = interp1(zt, z_centered, Zt(:, 1, 1), "linear", "extrap");
    Zs = reshape(Zs, [], 1, 1);
    Zs = repmat(Zs, 1, target_num(2), target_num(3));
    Zs = permute(Zs, [1 2 3]); %#ok<NASGU>
end

% Keep the FOV fixed and enlarge/shrink only the phantom itself in XY.
[Xq, Yq, Zq] = ndgrid(xt ./ transverse_scale, yt ./ transverse_scale, linspace(zs(1), zs(end), target_num(3)));
target_volume = interpn(xs, ys, zs, source_volume, Xq, Yq, Zq, "linear", 0);
end


function tag = format_scale_tag(scale_value)
tag = sprintf("XYx%.2f", scale_value);
tag = strrep(tag, ".", "p");
end


function save_selected_slice_montage(path_out, volume)
num_tiles = size(volume, 3);
cols = 5;
rows = ceil(num_tiles / cols);
cell_h = size(volume, 1);
cell_w = size(volume, 2);
gap = 10;
canvas = ones(rows * cell_h + (rows + 1) * gap, cols * cell_w + (cols + 1) * gap, "single");
for k = 1:num_tiles
    r = floor((k - 1) / cols) + 1;
    c = mod(k - 1, cols) + 1;
    r0 = gap + (r - 1) * (cell_h + gap) + 1;
    c0 = gap + (c - 1) * (cell_w + gap) + 1;
    canvas(r0:r0 + cell_h - 1, c0:c0 + cell_w - 1) = 1 - volume(:, :, k);
end
imwrite(canvas, path_out);
end


function save_volume_preview(path_out, volume, spacing_mm)
num = size(volume);
x = voxel_centers(num(1), spacing_mm(1));
y = voxel_centers(num(2), spacing_mm(2));
z = voxel_centers(num(3), spacing_mm(3));

axial_mip = max(volume, [], 3);
coronal_mip = squeeze(max(volume, [], 2))';
sagittal_mip = squeeze(max(volume, [], 1))';
axial_center = volume(:, :, ceil(num(3) / 2));

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


function write_summary_txt(file_path, cfg, selected_indices, selected_size)
fid = fopen(file_path, "w");
if fid < 0
    error("Failed to open %s for writing.", file_path);
end
fprintf(fid, "Source YAFF compressed Hoffman phantom\n");
fprintf(fid, "group_size = %d\n", cfg.group_size);
fprintf(fid, "excluded_pos_in_group = %s\n", mat2str(cfg.exclude_pos_in_group));
fprintf(fid, "preferred_pos_in_group = %d\n", cfg.prefer_pos_in_group);
fprintf(fid, "selected_indices_1based = %s\n", mat2str(selected_indices));
fprintf(fid, "\n");
fprintf(fid, "Native compressed volume\n");
fprintf(fid, "  dims = %d x %d x %d\n", selected_size(1), selected_size(2), selected_size(3));
fprintf(fid, "  spacing_mm = %.5f x %.5f x %.5f\n", cfg.source_spacing_mm(1), cfg.source_spacing_mm(2), cfg.source_spacing_mm(3));
fprintf(fid, "\n");
fprintf(fid, "Target project volume\n");
fprintf(fid, "  dims = %d x %d x %d\n", cfg.target_pixel_num(1), cfg.target_pixel_num(2), cfg.target_pixel_num(3));
fprintf(fid, "  spacing_mm = %.5f x %.5f x %.5f\n", cfg.target_pixel_size_mm(1), cfg.target_pixel_size_mm(2), cfg.target_pixel_size_mm(3));
fprintf(fid, "  transverse_scale = %.4f\n", cfg.transverse_scale);
fclose(fid);
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


function centers = voxel_centers(num_voxel, voxel_size)
centers = (-num_voxel / 2 + 0.5) * voxel_size : voxel_size : (num_voxel / 2 - 0.5) * voxel_size;
end
