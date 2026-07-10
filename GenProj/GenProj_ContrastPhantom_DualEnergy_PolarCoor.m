%% GenProj_ContrastPhantom_DualEnergy_PolarCoor
% Forward-project the dual-energy 225Ac contrast phantom with SysMat_polar and
% add counting noise to generate single-photon CntStat files.
%
% This script matches the source model in:
%   Geant4Sim/ContrastPhantom_DualEnergy_Rotate_3D.m
%
% Outputs one CntStat/CntStatMean pair per configured energy and count level,
% by default 218/440 keV at 1e9, 1e10, and 1e11 total source counts.
%
% Notes:
%   - The Geant4 macro places the source at y = -245 mm in world coordinates.
%     Here the phantom is built in the reconstruction FOV coordinate system,
%     centered at (0, 0, 0), because coor_polar_full is defined around the FOV.
%   - Geant4 macro generation, GenProj, and the current 218/440 Factors all use
%     RotateNum20.

clear;
clc;

%% Configuration
cfg = struct();
cfg.repo_root = fileparts(fileparts(mfilename("fullpath")));
if strlength(string(cfg.repo_root)) == 0
    cfg.repo_root = pwd;
end

cfg.rotate_num = 20;
cfg.total_count_list = [1e9, 1e10, 1e11];
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

% Match ContrastPhantom_DualEnergy_Rotate_3D.m: build different x_Fr/x_Bi
% activity maps, normalize each map to unit spatial integral, and only then
% multiply by Y218/Y440. This preserves the displaced daughter distributions
% while enforcing the whole-run gamma yield ratio exactly in expectation.
cfg.weight_mode = "global_yield_after_energy_normalization";

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
img_activity_raw_all = zeros(pixel_num, numel(cfg.energy_keV), "single");
for id_energy = 1:numel(cfg.energy_keV)
    img_activity_raw_all(:, id_energy) = build_dual_energy_contrast_source( ...
        coor_polar_full, cfg, cfg.energy_keV(id_energy));
end

activity_sum = sum(double(img_activity_raw_all), 1);
if any(activity_sum <= 0)
    error("At least one daughter activity map is empty. Check phantom geometry.");
end

% Normalize in double precision so the strict whole-run yield check is not
% dominated by float32 accumulation error over all polar voxels.
img_activity_fraction_all = double(img_activity_raw_all) ./ activity_sum;
img_raw_all = img_activity_fraction_all .* reshape(cfg.yield, 1, []);
raw_sum_all = sum(img_raw_all, "all");
img_fraction_all = single(img_raw_all / raw_sum_all);
img_polar_all = img_fraction_all * single(cfg.total_count_list(1) / cfg.rotate_num);

actual_energy_fraction = sum(double(img_fraction_all), 1);
expected_energy_fraction = cfg.yield / sum(cfg.yield);
if max(abs(actual_energy_fraction - expected_energy_fraction)) > 1e-6
    error("Whole-run energy fractions do not match the configured gamma yields.");
end

fprintf("Dual-energy source model\n");
fprintf("  rotate_num             = %d\n", cfg.rotate_num);
fprintf("  total_count_list        = %s\n", mat2str(cfg.total_count_list));
for id_energy = 1:numel(cfg.energy_keV)
    fprintf("  %d keV source fraction  = %.6g (%.2f%%)\n", ...
        cfg.energy_keV(id_energy), ...
        sum(img_fraction_all(:, id_energy), "all"), ...
        100 * sum(img_fraction_all(:, id_energy), "all"));
end

phantom_out_dir = fullfile(cfg.output_root, "PhantomPreview");
ensure_dir(phantom_out_dir);
save(fullfile(phantom_out_dir, "ContrastPhantom_DualEnergy_225Ac_source_polar.mat"), ...
    "cfg", "coor_polar_full", "img_activity_raw_all", ...
    "img_activity_fraction_all", "img_raw_all", "img_fraction_all", ...
    "img_polar_all", "-v7.3");

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

    img_fraction = img_fraction_all(:, id_energy);
    cntstat_mean_per_source_count = zeros(cfg.rotate_num, detector_num, "single");

    for id_rotate = 1:cfg.rotate_num
        rot_idx = rot_mat_full(:, id_rotate);
        img_rot = img_fraction(rot_idx) / single(cfg.rotate_num);
        cntstat_mean_per_source_count(id_rotate, :) = (sysmat * img_rot).';
        fprintf("[%d keV] Unit-count forward projection %d/%d done. sum = %.6g\n", ...
            energy_keV, id_rotate, cfg.rotate_num, ...
            sum(cntstat_mean_per_source_count(id_rotate, :), "all"));
    end

    out_dir = fullfile(cfg.output_root, sprintf("%dkeV_RotateNum%d", energy_keV, cfg.rotate_num));
    ensure_dir(out_dir);

    for id_count = 1:numel(cfg.total_count_list)
        total_count = cfg.total_count_list(id_count);
        count_level = format_count_level(total_count);
        cntstat_mean = max(cntstat_mean_per_source_count * single(total_count), 0);
        cntstat = apply_count_noise(cntstat_mean, cfg.noise_model);

        out_file = fullfile(out_dir, sprintf("CntStat_%s_%s.csv", cfg.data_file_name, count_level));
        mean_file = fullfile(out_dir, sprintf("CntStatMean_%s_%s.csv", cfg.data_file_name, count_level));

        writematrix(cntstat, out_file);
        writematrix(cntstat_mean, mean_file);

        fprintf("[%d keV, %s] Wrote noisy CntStat: %s\n", energy_keV, count_level, out_file);
        fprintf("[%d keV, %s] Wrote mean  CntStat: %s\n", energy_keV, count_level, mean_file);
        fprintf("[%d keV, %s] noisy total = %.6g, mean total = %.6g\n", ...
            energy_keV, count_level, sum(cntstat, "all"), sum(cntstat_mean, "all"));
    end
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

    if cfg.weight_mode ~= "global_yield_after_energy_normalization"
        error("Unknown cfg.weight_mode: %s", cfg.weight_mode);
    end

    img(back_mask) = 1;

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
        img(rod_mask) = img(rod_mask) + single(cfg.act - 1);
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

function label = format_count_level(value)
    if ~isfinite(value) || value <= 0
        error("Count levels must be finite positive values; got %.6g.", value);
    end
    exponent = floor(log10(value));
    mantissa = value / 10^exponent;
    if abs(mantissa - round(mantissa)) < 1e-12
        mantissa_text = sprintf("%d", round(mantissa));
    else
        mantissa_text = sprintf("%.12g", mantissa);
    end
    label = sprintf("%se%d", mantissa_text, exponent);
end
