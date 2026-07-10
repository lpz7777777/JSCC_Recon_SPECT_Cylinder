%% GenProj_ContrastPhantom_SingleEnergy_PolarCoor
% Forward-project a single-energy contrast phantom with SysMat_polar and add
% counting noise to generate CntStat files.
%
% This script matches the single-energy source geometry in:
%   Geant4Sim/ContrastPhantom_Rotate_3D.m
%
% Default output:
%   CntStat/440keV_RotateNum20/CntStat_ContrastPhantom_10_30_240_30_1e9.csv
%   CntStat/440keV_RotateNum20/CntStat_ContrastPhantom_10_30_240_30_1e10.csv
%   CntStat/440keV_RotateNum20/CntStat_ContrastPhantom_10_30_240_30_1e11.csv

clear;
clc;

%% Configuration
cfg = struct();
cfg.repo_root = fileparts(fileparts(mfilename("fullpath")));
if strlength(string(cfg.repo_root)) == 0
    cfg.repo_root = pwd;
end

cfg.energy_keV = 440;
cfg.rotate_num = 20;
cfg.total_count_list = [1e9, 1e10, 1e11];
cfg.count_level_list = ["1e9", "1e10", "1e11"];
cfg.data_file_name = "ContrastPhantom_10_30_240_30";
cfg.output_root = fullfile(cfg.repo_root, "CntStat");
cfg.factor_root = fullfile(cfg.repo_root, "Factors");

% Noise model: "poisson", "gaussian", or "none".
cfg.noise_model = "poisson";
cfg.rng_seed = 20260709;

% Phantom geometry, matching Geant4Sim/ContrastPhantom_Rotate_3D.m.
cfg.back_rod_d = 240;         % mm
cfg.rod_d = 10:4:30;          % mm
cfg.height = 30;              % mm
cfg.act = 6;
cfg.center_xy = [0, 0];       % FOV coordinates, not Geant4 world coordinates

if numel(cfg.total_count_list) ~= numel(cfg.count_level_list)
    error("cfg.total_count_list and cfg.count_level_list must have the same length.");
end

rng(cfg.rng_seed);

%% Load factors
factor_dir = fullfile(cfg.factor_root, sprintf("%dkeV_RotateNum%d", cfg.energy_keV, cfg.rotate_num));
if ~exist(factor_dir, "dir")
    error("Cannot find factor directory: %s", factor_dir);
end

coor_polar_full = load_named_array( ...
    fullfile(factor_dir, "coor_polar_full.mat"), ...
    fullfile(factor_dir, "coor_polar_full.csv"), ...
    "coor_polar_full");
rot_mat_full = load_named_array( ...
    fullfile(factor_dir, "RotMat_full.mat"), ...
    fullfile(factor_dir, "RotMat_full.csv"), ...
    "RotMat_full");

pixel_num = size(coor_polar_full, 1);
if size(rot_mat_full, 1) ~= pixel_num
    error("RotMat_full rows (%d) do not match coor_polar_full rows (%d).", ...
        size(rot_mat_full, 1), pixel_num);
end
if size(rot_mat_full, 2) ~= cfg.rotate_num
    error("RotMat_full has %d rotations, expected cfg.rotate_num=%d.", ...
        size(rot_mat_full, 2), cfg.rotate_num);
end

sysmat_path = fullfile(factor_dir, "SysMat_polar");
sysmat = read_sysmat_polar(sysmat_path, pixel_num);
detector_num = size(sysmat, 1);

fprintf("[%d keV] Factor dir  = %s\n", cfg.energy_keV, factor_dir);
fprintf("[%d keV] SysMat size = %d detectors x %d pixels\n", ...
    cfg.energy_keV, detector_num, size(sysmat, 2));

%% Build the source image once and compute unit-count forward projection
img_raw = build_single_energy_contrast_source(coor_polar_full, cfg);
raw_sum = sum(img_raw, "all");
if raw_sum <= 0
    error("The generated source image is empty. Check phantom geometry and coor_polar_full.");
end

img_unit = img_raw * single(1 / cfg.rotate_num / raw_sum);
cntstat_mean_unit = zeros(cfg.rotate_num, detector_num, "single");

for id_rotate = 1:cfg.rotate_num
    rot_idx = rot_mat_full(:, id_rotate);
    img_rot = img_unit(rot_idx);
    cntstat_mean_unit(id_rotate, :) = (sysmat * img_rot).';
    fprintf("[%d keV] Unit forward projection %d/%d done. unit mean sum = %.6g\n", ...
        cfg.energy_keV, id_rotate, cfg.rotate_num, sum(cntstat_mean_unit(id_rotate, :), "all"));
end
cntstat_mean_unit = max(cntstat_mean_unit, 0);

%% Write all requested count levels
out_dir = fullfile(cfg.output_root, sprintf("%dkeV_RotateNum%d", cfg.energy_keV, cfg.rotate_num));
ensure_dir(out_dir);

phantom_out_dir = fullfile(cfg.output_root, "PhantomPreview");
ensure_dir(phantom_out_dir);
save(fullfile(phantom_out_dir, sprintf("%s_%dkeV_source_polar.mat", cfg.data_file_name, cfg.energy_keV)), ...
    "cfg", "coor_polar_full", "img_raw", "img_unit", "cntstat_mean_unit", "-v7.3");

for id_count = 1:numel(cfg.total_count_list)
    total_count = cfg.total_count_list(id_count);
    count_level = cfg.count_level_list(id_count);

    cntstat_mean = double(cntstat_mean_unit) * total_count;
    cntstat = apply_count_noise(cntstat_mean, cfg.noise_model);

    out_file = fullfile(out_dir, sprintf("CntStat_%s_%s.csv", cfg.data_file_name, count_level));
    mean_file = fullfile(out_dir, sprintf("CntStatMean_%s_%s.csv", cfg.data_file_name, count_level));

    writematrix(cntstat, out_file);
    writematrix(cntstat_mean, mean_file);

    fprintf("[%d keV] Wrote noisy CntStat: %s\n", cfg.energy_keV, out_file);
    fprintf("[%d keV] Wrote mean  CntStat: %s\n", cfg.energy_keV, mean_file);
    fprintf("[%d keV] %s noisy total = %.6g, mean total = %.6g\n", ...
        cfg.energy_keV, count_level, sum(cntstat, "all"), sum(cntstat_mean, "all"));
end

fprintf("\nDone.\n");

%% Local functions
function arr = load_named_array(mat_path, csv_path, preferred_name)
    preferred_name = char(preferred_name);
    if exist(mat_path, "file")
        data = load(mat_path);
        if isfield(data, preferred_name)
            arr = data.(preferred_name);
            return;
        end
        names = fieldnames(data);
        if numel(names) ~= 1
            error("MAT file %s does not contain %s and has multiple variables.", ...
                mat_path, preferred_name);
        end
        arr = data.(names{1});
        return;
    end
    if exist(csv_path, "file")
        arr = readmatrix(csv_path);
        return;
    end
    error("Cannot find %s or %s.", mat_path, csv_path);
end

function sysmat = read_sysmat_polar(sysmat_path, pixel_num)
    if ~exist(sysmat_path, "file")
        error("Cannot find SysMat_polar: %s", sysmat_path);
    end

    info = dir(sysmat_path);
    elem_num = info.bytes / 4;
    if abs(elem_num - round(elem_num)) > 0
        error("SysMat_polar byte size is not divisible by float32 size: %s", sysmat_path);
    end
    elem_num = round(elem_num);
    if mod(elem_num, pixel_num) ~= 0
        error("SysMat_polar element count %d is incompatible with pixel_num=%d.", ...
            elem_num, pixel_num);
    end

    detector_num = elem_num / pixel_num;
    fid = fopen(sysmat_path, "r");
    if fid < 0
        error("Failed to open SysMat_polar: %s", sysmat_path);
    end
    cleanup = onCleanup(@() fclose(fid));
    sysmat = fread(fid, [detector_num, pixel_num], "single=>single");
    if numel(sysmat) ~= elem_num
        error("Failed to read all SysMat_polar elements from %s.", sysmat_path);
    end
end

function img = build_single_energy_contrast_source(coor, cfg)
    x = coor(:, 1) - cfg.center_xy(1);
    y = coor(:, 2) - cfg.center_xy(2);
    z = coor(:, 3);

    back_r = cfg.back_rod_d / 2;
    back_mask = (x.^2 + y.^2) <= back_r^2 & abs(z) <= cfg.height / 2;

    img = zeros(size(coor, 1), 1, "single");
    img(back_mask) = single(1);

    for i = 1:numel(cfg.rod_d)
        theta = (i - 1) * pi / 3;
        cx = cfg.back_rod_d / 4 * cos(theta) + cfg.center_xy(1);
        cy = cfg.back_rod_d / 4 * sin(theta) + cfg.center_xy(2);
        rod_r = cfg.rod_d(i) / 2;

        rod_mask = ((coor(:, 1) - cx).^2 + (coor(:, 2) - cy).^2) <= rod_r^2 & ...
            abs(coor(:, 3)) <= cfg.height / 2;
        img(rod_mask) = single(cfg.act);
    end
end

function cntstat = apply_count_noise(cntstat_mean, noise_model)
    noise_model = string(noise_model);
    switch noise_model
        case "none"
            cntstat = round(cntstat_mean);
        case "poisson"
            if exist("poissrnd", "file") == 2
                cntstat = poissrnd(cntstat_mean);
            else
                warning("poissrnd is unavailable. Falling back to Gaussian approximation.");
                cntstat = round(cntstat_mean + sqrt(cntstat_mean) .* randn(size(cntstat_mean)));
            end
        case "gaussian"
            cntstat = round(cntstat_mean + sqrt(cntstat_mean) .* randn(size(cntstat_mean)));
        otherwise
            error("Unknown noise_model: %s", noise_model);
    end
    cntstat = max(cntstat, 0);
end

function ensure_dir(path_value)
    if ~exist(path_value, "dir")
        mkdir(path_value);
    end
end
