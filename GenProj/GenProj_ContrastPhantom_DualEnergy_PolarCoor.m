%% GenProj_ContrastPhantom_DualEnergy_PolarCoor
% Forward-project the dual-energy 225Ac contrast phantom with SysMat_polar and
% add counting noise to generate single-photon CntStat files.
%
% This script matches the source model in:
%   Geant4Sim/ContrastPhantom_DualEnergy_Rotate_3D.m
%
% Outputs:
%   CntStat/218keV_RotateNum20/CntStat_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_1e9.csv
%   CntStat/440keV_RotateNum20/CntStat_ContrastPhantom_DualEnergy_10_30_240_30_225Ac_1e9.csv
%
% Notes:
%   - The Geant4 macro places the source at y = -245 mm in world coordinates.
%     Here the phantom is built in the reconstruction FOV coordinate system,
%     centered at (0, 0, 0), because coor_polar_full is defined around the FOV.
%   - The default RotateNum is 20 because the currently generated 218/440
%     Factors are RotateNum20. Change cfg.rotate_num to 60 once RotateNum60
%     Factors are available.

clear;
clc;

%% Configuration
cfg = struct();
cfg.repo_root = fileparts(fileparts(mfilename("fullpath")));
if strlength(string(cfg.repo_root)) == 0
    cfg.repo_root = pwd;
end

cfg.rotate_num = 20;
cfg.total_count = 1e9;
cfg.count_level = "1e9";
cfg.data_file_name = "ContrastPhantom_DualEnergy_10_30_240_30_225Ac";
cfg.output_root = fullfile(cfg.repo_root, "CntStat");

% Prefer main-project Factors/. If the new 218/440 Factors have not been copied
% there yet, fall back to the GPU system-matrix auxiliary workspace.
cfg.factor_roots = [
    string(fullfile(cfg.repo_root, "Factors"))
    string(fullfile(cfg.repo_root, "Auxiliary_Studies", ...
        "GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main", "Factors"))
];

% Noise model: "poisson", "gaussian", or "none".
% "poisson" is the default counting model; "gaussian" reproduces the style
% used by GenProj_SPECT_PolarCoor.m.
cfg.noise_model = "poisson";
cfg.rng_seed = 20260709;

% Phantom geometry, matching ContrastPhantom_DualEnergy_Rotate_3D.m.
cfg.back_rod_d = 240;         % mm
cfg.rod_d = 10:4:30;          % mm
cfg.height = 30;              % mm
cfg.act = 6;
cfg.center_xy = [0, 0];       % FOV coordinates, not Geant4 world coordinates

% 225Ac gamma yields used in the Geant4 macro.
cfg.energy_keV = [218, 440];
cfg.yield = [0.114, 0.261];
cfg.rod_energy_keV = [218, 440, 218, 440, 218, 440];

% "macro" reproduces the exact GPS source weights in
% ContrastPhantom_DualEnergy_Rotate_3D.m:
%   background 218 weight = 1
%   background 440 weight = yield_440 / yield_218
%   rod additive weights = (act - 1) * rod_area / back_area
% Multiplying all source weights by yield_218 gives the density model below.
%
% If you later want each energy's rods to be act-times hotter than that same
% energy's own background, switch this to "per_energy_hot".
cfg.weight_mode = "macro";

rng(cfg.rng_seed);

%% Load one reference coordinate grid and build both source-density images
factor_dirs = containers.Map("KeyType", "double", "ValueType", "char");
for id_energy = 1:numel(cfg.energy_keV)
    energy_keV = cfg.energy_keV(id_energy);
    factor_dirs(energy_keV) = resolve_factor_dir(cfg.factor_roots, energy_keV, cfg.rotate_num);
end

ref_factor_dir = factor_dirs(cfg.energy_keV(1));
coor_polar_full = load_named_array( ...
    fullfile(ref_factor_dir, "coor_polar_full.mat"), ...
    fullfile(ref_factor_dir, "coor_polar_full.csv"), ...
    "coor_polar_full");

pixel_num = size(coor_polar_full, 1);
img_raw_all = zeros(pixel_num, numel(cfg.energy_keV), "single");
for id_energy = 1:numel(cfg.energy_keV)
    img_raw_all(:, id_energy) = build_dual_energy_contrast_source( ...
        coor_polar_full, cfg, cfg.energy_keV(id_energy));
end

raw_sum_all = sum(img_raw_all, "all");
if raw_sum_all <= 0
    error("The generated source image is empty. Check phantom geometry and coor_polar_full.");
end

total_count_singleview = cfg.total_count / cfg.rotate_num;
img_polar_all = img_raw_all * single(total_count_singleview / raw_sum_all);

fprintf("Dual-energy source model\n");
fprintf("  total_count            = %.6g\n", cfg.total_count);
fprintf("  rotate_num             = %d\n", cfg.rotate_num);
fprintf("  total_count_singleview = %.6g\n", total_count_singleview);
for id_energy = 1:numel(cfg.energy_keV)
    fprintf("  %d keV image sum/view   = %.6g (%.2f%%)\n", ...
        cfg.energy_keV(id_energy), ...
        sum(img_polar_all(:, id_energy), "all"), ...
        100 * sum(img_polar_all(:, id_energy), "all") / total_count_singleview);
end

phantom_out_dir = fullfile(cfg.output_root, "PhantomPreview");
ensure_dir(phantom_out_dir);
save(fullfile(phantom_out_dir, "ContrastPhantom_DualEnergy_225Ac_source_polar.mat"), ...
    "cfg", "coor_polar_full", "img_raw_all", "img_polar_all", "-v7.3");

%% Forward project each energy independently and write CntStat files
for id_energy = 1:numel(cfg.energy_keV)
    energy_keV = cfg.energy_keV(id_energy);
    factor_dir = factor_dirs(energy_keV);

    fprintf("\n[%d keV] Loading factors from %s\n", energy_keV, factor_dir);
    rot_mat_full = load_named_array( ...
        fullfile(factor_dir, "RotMat_full.mat"), ...
        fullfile(factor_dir, "RotMat_full.csv"), ...
        "RotMat_full");

    if size(rot_mat_full, 1) ~= pixel_num
        error("[%d keV] RotMat_full rows (%d) do not match coor_polar_full rows (%d).", ...
            energy_keV, size(rot_mat_full, 1), pixel_num);
    end
    if size(rot_mat_full, 2) ~= cfg.rotate_num
        error("[%d keV] RotMat_full has %d rotations, expected cfg.rotate_num=%d.", ...
            energy_keV, size(rot_mat_full, 2), cfg.rotate_num);
    end

    sysmat_path = fullfile(factor_dir, "SysMat_polar");
    sysmat = read_sysmat_polar(sysmat_path, pixel_num);
    detector_num = size(sysmat, 1);
    fprintf("[%d keV] SysMat size = %d detectors x %d pixels\n", ...
        energy_keV, detector_num, size(sysmat, 2));

    img_polar = img_polar_all(:, id_energy);
    cntstat_mean = zeros(cfg.rotate_num, detector_num, "single");

    for id_rotate = 1:cfg.rotate_num
        rot_idx = rot_mat_full(:, id_rotate);
        img_rot = img_polar(rot_idx);
        cntstat_mean(id_rotate, :) = (sysmat * img_rot).';
        fprintf("[%d keV] Forward projection %d/%d done. mean sum = %.6g\n", ...
            energy_keV, id_rotate, cfg.rotate_num, sum(cntstat_mean(id_rotate, :), "all"));
    end

    cntstat_mean = max(cntstat_mean, 0);
    cntstat = apply_count_noise(cntstat_mean, cfg.noise_model);

    out_dir = fullfile(cfg.output_root, sprintf("%dkeV_RotateNum%d", energy_keV, cfg.rotate_num));
    ensure_dir(out_dir);

    out_file = fullfile(out_dir, sprintf("CntStat_%s_%s.csv", cfg.data_file_name, cfg.count_level));
    mean_file = fullfile(out_dir, sprintf("CntStatMean_%s_%s.csv", cfg.data_file_name, cfg.count_level));

    writematrix(cntstat, out_file);
    writematrix(cntstat_mean, mean_file);

    fprintf("[%d keV] Wrote noisy CntStat: %s\n", energy_keV, out_file);
    fprintf("[%d keV] Wrote mean  CntStat: %s\n", energy_keV, mean_file);
    fprintf("[%d keV] noisy total = %.6g, mean total = %.6g\n", ...
        energy_keV, sum(cntstat, "all"), sum(cntstat_mean, "all"));
end

fprintf("\nDone.\n");

%% Local functions
function factor_dir = resolve_factor_dir(factor_roots, energy_keV, rotate_num)
    rel = sprintf("%dkeV_RotateNum%d", energy_keV, rotate_num);
    tried = strings(numel(factor_roots), 1);
    for i = 1:numel(factor_roots)
        candidate = fullfile(factor_roots(i), rel);
        tried(i) = string(candidate);
        if exist(candidate, "dir")
            factor_dir = char(candidate);
            return;
        end
    end
    error("Cannot find factor directory %s. Tried:\n%s", rel, strjoin(tried, newline));
end

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

function img = build_dual_energy_contrast_source(coor, cfg, energy_keV)
    x = coor(:, 1) - cfg.center_xy(1);
    y = coor(:, 2) - cfg.center_xy(2);
    z = coor(:, 3);

    back_r = cfg.back_rod_d / 2;
    back_mask = (x.^2 + y.^2) <= back_r^2 & abs(z) <= cfg.height / 2;

    energy_idx = find(cfg.energy_keV == energy_keV, 1);
    if isempty(energy_idx)
        error("Energy %d keV is not configured.", energy_keV);
    end

    img = zeros(size(coor, 1), 1, "single");

    if cfg.weight_mode == "macro"
        background_density = cfg.yield(energy_idx);
        rod_add_density_base = cfg.yield(1) * (cfg.act - 1);
    elseif cfg.weight_mode == "per_energy_hot"
        background_density = cfg.yield(energy_idx);
        rod_add_density_base = cfg.yield(energy_idx) * (cfg.act - 1);
    else
        error("Unknown cfg.weight_mode: %s", cfg.weight_mode);
    end

    img(back_mask) = single(background_density);

    for i = 1:numel(cfg.rod_d)
        if cfg.rod_energy_keV(i) ~= energy_keV
            continue;
        end

        theta = (i - 1) * pi / 3;
        cx = cfg.back_rod_d / 4 * cos(theta) + cfg.center_xy(1);
        cy = cfg.back_rod_d / 4 * sin(theta) + cfg.center_xy(2);
        rod_r = cfg.rod_d(i) / 2;

        rod_mask = ((coor(:, 1) - cx).^2 + (coor(:, 2) - cy).^2) <= rod_r^2 & ...
            abs(coor(:, 3)) <= cfg.height / 2;
        img(rod_mask) = img(rod_mask) + single(rod_add_density_base);
    end
end

function cntstat = apply_count_noise(cntstat_mean, noise_model)
    noise_model = string(noise_model);
    switch noise_model
        case "none"
            cntstat = round(cntstat_mean);
        case "poisson"
            if exist("poissrnd", "file") == 2
                cntstat = poissrnd(double(cntstat_mean));
            else
                warning("poissrnd is unavailable. Falling back to Gaussian approximation.");
                cntstat = round(double(cntstat_mean) + sqrt(double(cntstat_mean)) .* randn(size(cntstat_mean)));
            end
        case "gaussian"
            cntstat = round(double(cntstat_mean) + sqrt(double(cntstat_mean)) .* randn(size(cntstat_mean)));
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
