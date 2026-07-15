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
% "" selects JSCC; use "_SPECTEHENaI" for the EHE Pb/NaI response set.
% Batch jobs can override these without editing this file:
%   JSCC_RECON_FACTOR_DIR_SUFFIX
%   JSCC_RECON_CNTSTAT_DIR_SUFFIX
cfg.factor_dir_suffix = string(getenv("JSCC_RECON_FACTOR_DIR_SUFFIX"));
cfg.cntstat_dir_suffix = string(getenv("JSCC_RECON_CNTSTAT_DIR_SUFFIX"));
if strlength(cfg.cntstat_dir_suffix) == 0
    cfg.cntstat_dir_suffix = cfg.factor_dir_suffix;
end

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
    factor_dirs(energy_keV) = resolve_factor_dir( ...
        cfg.factor_roots, energy_keV, cfg.rotate_num, cfg.factor_dir_suffix);
end
cross_factor_dir = resolve_cross_factor_dir( ...
    cfg.factor_roots, 440, 218, cfg.rotate_num, cfg.factor_dir_suffix);

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

%% Load A218, A440 and C440->218, then forward-project the two observed windows
response_218 = load_response_factors(factor_dirs(218), pixel_num, cfg, "A218");
response_440 = load_response_factors(factor_dirs(440), pixel_num, cfg, "A440");
response_cross = load_response_factors(cross_factor_dir, pixel_num, cfg, "C440to218");
validate_response_compatibility(response_218, response_440, response_cross);

idx_218 = find(cfg.energy_keV == 218, 1);
idx_440 = find(cfg.energy_keV == 440, 1);
mean_218_direct_unit = forward_project_response( ...
    response_218, img_fraction_all(:, idx_218), cfg.rotate_num, "218 direct");
mean_218_cross_unit = forward_project_response( ...
    response_cross, img_fraction_all(:, idx_440), cfg.rotate_num, "440-to-218 cross-talk");
mean_440_unit = forward_project_response( ...
    response_440, img_fraction_all(:, idx_440), cfg.rotate_num, "440 direct");

out_218 = fullfile(cfg.output_root, build_energy_dir_name( ...
    218, cfg.rotate_num, cfg.cntstat_dir_suffix));
out_440 = fullfile(cfg.output_root, build_energy_dir_name( ...
    440, cfg.rotate_num, cfg.cntstat_dir_suffix));
ensure_dir(out_218);
ensure_dir(out_440);

projection_summary = repmat(struct(), numel(cfg.total_count_list), 1);
for id_count = 1:numel(cfg.total_count_list)
    total_count = cfg.total_count_list(id_count);
    count_level = format_count_level(total_count);

    mean_218_direct = max(mean_218_direct_unit * single(total_count), 0);
    mean_218_cross = max(mean_218_cross_unit * single(total_count), 0);
    mean_218_total = mean_218_direct + mean_218_cross;
    cnt_218_direct = apply_count_noise(mean_218_direct, cfg.noise_model);
    cnt_218_cross = apply_count_noise(mean_218_cross, cfg.noise_model);
    cnt_218_total = cnt_218_direct + cnt_218_cross;

    mean_440 = max(mean_440_unit * single(total_count), 0);
    cnt_440 = apply_count_noise(mean_440, cfg.noise_model);

    write_projection_set(out_218, cfg.data_file_name, count_level, ...
        cnt_218_total, mean_218_total, cnt_218_direct, mean_218_direct, ...
        cnt_218_cross, mean_218_cross);
    write_projection_set(out_440, cfg.data_file_name, count_level, ...
        cnt_440, mean_440, cnt_440, mean_440, [], []);

    projection_summary(id_count).count_level = count_level;
    projection_summary(id_count).total_source_photons = total_count;
    projection_summary(id_count).cnt218_direct = sum(cnt_218_direct, "all");
    projection_summary(id_count).cnt218_cross = sum(cnt_218_cross, "all");
    projection_summary(id_count).cnt218_total = sum(cnt_218_total, "all");
    projection_summary(id_count).cnt440 = sum(cnt_440, "all");
    projection_summary(id_count).cross_fraction_in_218 = ...
        projection_summary(id_count).cnt218_cross / max(projection_summary(id_count).cnt218_total, 1);

    fprintf("[%s] 218 direct=%g, cross=%g (%.3f%%), total=%g; 440=%g\n", ...
        count_level, projection_summary(id_count).cnt218_direct, ...
        projection_summary(id_count).cnt218_cross, ...
        100 * projection_summary(id_count).cross_fraction_in_218, ...
        projection_summary(id_count).cnt218_total, projection_summary(id_count).cnt440);
end

projection_manifest = struct();
projection_manifest.model = "y218=A218*x218+C440to218*x440; y440=A440*x440";
projection_manifest.noise_superposition = ...
    "218 direct and cross-talk are independently sampled, then added";
projection_manifest.factor_dir_suffix = cfg.factor_dir_suffix;
projection_manifest.gamma_yields = cfg.yield;
projection_manifest.yield_application = ...
    "applied once in emitted-photon source fractions; response matrices are per emitted photon";
projection_manifest.factor_A218 = response_218.factor_dir;
projection_manifest.factor_A440 = response_440.factor_dir;
projection_manifest.factor_C440to218 = response_cross.factor_dir;
projection_manifest.summary = projection_summary;
write_json(fullfile(cfg.output_root, sprintf("ProjectionManifest_%s%s.json", ...
    cfg.data_file_name, cfg.cntstat_dir_suffix)), projection_manifest);

fprintf("\nDone.\n");

%% Local functions
function factor_dir = resolve_factor_dir(factor_roots, energy_keV, rotate_num, suffix)
    rel = build_energy_dir_name(energy_keV, rotate_num, suffix);
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

function factor_dir = resolve_cross_factor_dir(factor_roots, source_keV, window_keV, rotate_num, suffix)
    rel = sprintf("%dkeV_to%dwin_RotateNum%d%s", ...
        source_keV, window_keV, rotate_num, string(suffix));
    factor_dir = resolve_factor_rel(factor_roots, rel);
end

function factor_dir = resolve_factor_rel(factor_roots, rel)
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

function name = build_energy_dir_name(energy_keV, rotate_num, suffix)
    name = sprintf("%dkeV_RotateNum%d%s", energy_keV, rotate_num, string(suffix));
end

function response = load_response_factors(factor_dir, pixel_num, cfg, expected_response)
    fprintf("\n[%s] Loading factors from %s\n", expected_response, factor_dir);
    validate_factor_manifest(factor_dir, expected_response);
    response = struct();
    response.name = expected_response;
    response.factor_dir = factor_dir;
    response.rotmat = load_named_array( ...
        fullfile(factor_dir, "RotMat_full.mat"), ...
        fullfile(factor_dir, "RotMat_full.csv"), "RotMat_full");
    if ~isequal(size(response.rotmat), [pixel_num, cfg.rotate_num])
        error("[%s] RotMat_full has shape %s; expected [%d %d].", ...
            expected_response, mat2str(size(response.rotmat)), pixel_num, cfg.rotate_num);
    end
    response.sysmat = read_sysmat_polar(fullfile(factor_dir, "SysMat_polar"), pixel_num);
    response.detector_num = size(response.sysmat, 1);
    fprintf("[%s] SysMat size = %d detectors x %d pixels\n", ...
        expected_response, size(response.sysmat, 1), size(response.sysmat, 2));
end

function validate_factor_manifest(factor_dir, expected_response)
    path = fullfile(factor_dir, "factor_manifest.json");
    if ~exist(path, "file")
        warning("Factor manifest is missing; validating by directory and dimensions only: %s", factor_dir);
        return;
    end
    manifest = jsondecode(fileread(path));
    if ~strcmp(string(manifest.response), expected_response)
        error("Factor response mismatch in %s: got %s, expected %s.", ...
            path, string(manifest.response), expected_response);
    end
end

function validate_response_compatibility(response_218, response_440, response_cross)
    if response_218.detector_num ~= response_cross.detector_num
        error("A218 and C440to218 detector counts differ: %d vs %d.", ...
            response_218.detector_num, response_cross.detector_num);
    end
    if response_218.detector_num ~= response_440.detector_num
        error("A218 and A440 detector counts differ: %d vs %d.", ...
            response_218.detector_num, response_440.detector_num);
    end
    if ~isequal(response_218.rotmat, response_cross.rotmat) || ...
            ~isequal(response_218.rotmat, response_440.rotmat)
        error("A218, A440 and C440to218 must use identical rotation mappings.");
    end
end

function mean_per_source = forward_project_response(response, image_fraction, rotate_num, label)
    mean_per_source = zeros(rotate_num, response.detector_num, "single");
    for id_rotate = 1:rotate_num
        rot_idx = response.rotmat(:, id_rotate);
        img_rot = image_fraction(rot_idx) / single(rotate_num);
        mean_per_source(id_rotate, :) = (response.sysmat * img_rot).';
        fprintf("[%s] projection %d/%d done; sum=%.6g\n", ...
            label, id_rotate, rotate_num, sum(mean_per_source(id_rotate, :), "all"));
    end
end

function write_projection_set(out_dir, dataset, count_level, cnt_total, mean_total, ...
        cnt_direct, mean_direct, cnt_cross, mean_cross)
    writematrix(cnt_total, fullfile(out_dir, sprintf("CntStat_%s_%s.csv", dataset, count_level)));
    writematrix(mean_total, fullfile(out_dir, sprintf("CntStatMean_%s_%s.csv", dataset, count_level)));
    writematrix(cnt_direct, fullfile(out_dir, sprintf("CntStatDirect_%s_%s.csv", dataset, count_level)));
    writematrix(mean_direct, fullfile(out_dir, sprintf("CntStatMeanDirect_%s_%s.csv", dataset, count_level)));
    if ~isempty(cnt_cross)
        writematrix(cnt_cross, fullfile(out_dir, sprintf("CntStatCrossTalk_%s_%s.csv", dataset, count_level)));
        writematrix(mean_cross, fullfile(out_dir, sprintf("CntStatMeanCrossTalk_%s_%s.csv", dataset, count_level)));
    end
end

function write_json(path, value)
    text = jsonencode(value, "PrettyPrint", true);
    file_id = fopen(path, "w");
    if file_id < 0
        error("Cannot write JSON file: %s", path);
    end
    cleanup = onCleanup(@() fclose(file_id));
    fwrite(file_id, text, "char");
    fwrite(file_id, newline, "char");
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
