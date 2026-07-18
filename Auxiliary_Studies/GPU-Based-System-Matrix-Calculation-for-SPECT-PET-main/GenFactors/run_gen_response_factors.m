function results = run_gen_response_factors(case_selectors, output_suffix, grid_options)
%RUN_GEN_RESPONSE_FACTORS Generate the six JSCC/EHE dual-energy responses.
%
% Final responses:
%   A218        = A(218-window <- 218-source), SysMat_withScatter
%   A440        = A(440-window <- 440-source), SysMat_withScatter
%   C440to218   = A(218-window <- 440-source), Scatter_SysMat only
%
% Each response is generated in a staging directory and validated before the
% corresponding project-root Factors directory is replaced.
% Optional selectors use "<system>/<response>", for example:
%   run_gen_response_factors("SPECTEHENaI/C440to218")
%   run_gen_response_factors(["JSCC/A218", "JSCC/A440", "JSCC/C440to218"], ...
%       "CenterPoint", struct('include_center_point', true))

    if nargin < 1
        case_selectors = strings(0, 1);
    end
    if nargin < 2 || isempty(output_suffix)
        output_suffix = "";
    end
    if nargin < 3 || isempty(grid_options)
        grid_options = struct('include_center_point', false);
    end
    if ~isfield(grid_options, 'include_center_point')
        grid_options.include_center_point = false;
    end
    grid_options.include_center_point = logical(grid_options.include_center_point);

    tool_dir = fileparts(mfilename('fullpath'));
    engine_root = fileparts(tool_dir);
    project_root = fileparts(fileparts(engine_root));
    runs_root = fullfile(engine_root, 'runs');
    factors_root = fullfile(project_root, 'Factors');
    rotate_num = 20;

    all_cases = build_cases(runs_root, factors_root, rotate_num, output_suffix);
    cases = select_cases(all_cases, case_selectors);
    require_case_inputs(cases);
    results = repmat(empty_result_summary(), numel(cases), 1);
    fprintf('Generating %d dual-energy response Factors under:\n  %s\n', ...
        numel(cases), factors_root);

    for case_idx = 1:numel(cases)
        item = cases(case_idx);
        fprintf('\n[%d/%d] %s / %s\n', case_idx, numel(cases), ...
            item.system_tag, item.response);
        timestamp = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmssSSS'));
        build_token = sprintf('%s_%d', timestamp, feature('getpid'));
        staging_dir = fullfile(factors_root, ['.build_' item.output_name '_' build_token]);
        cleanup_staging = onCleanup(@() remove_tree_if_exists(staging_dir));

        gen_factors(item.source_energy_keV, item.sysmat_file, ...
            item.params_detector_file, staging_dir, '', item.calibration, grid_options);
        summary = validate_factor_dir(staging_dir, item, rotate_num, grid_options);
        write_factor_manifest(staging_dir, item, summary, rotate_num, grid_options, output_suffix);
        install_factor_dir(staging_dir, item.output_dir);
        clear cleanup_staging;

        summary.output_dir = item.output_dir;
        results(case_idx, 1) = summary;
        fprintf('Installed validated Factors: %s\n', item.output_dir);
    end

    available = cases(arrayfun(@(item) isfolder(item.output_dir), cases));
    validate_response_geometry(available);
    fprintf('\nSelected response Factors generated and available cross-geometry checks passed.\n');
end


function selected = select_cases(cases, selectors)
    if ischar(selectors)
        selectors = string({selectors});
    else
        selectors = string(selectors(:));
    end
    selectors = strip(selectors(:));
    selectors(selectors == "") = [];
    if isempty(selectors)
        selected = cases;
        return;
    end

    case_ids = string({cases.system_tag}).' + "/" + string({cases.response}).';
    output_names = string({cases.output_name}).';
    selected_mask = false(numel(cases), 1);
    for idx = 1:numel(selectors)
        matches = strcmpi(selectors(idx), case_ids) | ...
            strcmpi(selectors(idx), output_names);
        if ~any(matches)
            error('run_gen_response_factors:UnknownSelector', ...
                'Unknown case selector "%s". Available selectors:\n%s', ...
                selectors(idx), strjoin(case_ids, newline));
        end
        selected_mask = selected_mask | matches;
    end
    selected = cases(selected_mask);
end


function require_case_inputs(cases)
    missing = strings(0, 1);
    for idx = 1:numel(cases)
        if ~isfile(cases(idx).sysmat_file)
            missing(end + 1, 1) = "input system matrix: " + ...
                string(cases(idx).sysmat_file); %#ok<AGROW>
        end
        if ~isfile(cases(idx).params_detector_file)
            missing(end + 1, 1) = "Params_Detector.dat: " + ...
                string(cases(idx).params_detector_file); %#ok<AGROW>
        end
    end
    if ~isempty(missing)
        error('run_gen_response_factors:MissingInputs', ...
            'Required inputs are missing; no Factors were changed:\n%s', ...
            strjoin(missing, newline));
    end
end


function summary = empty_result_summary()
    summary = struct( ...
        'system_tag', '', ...
        'response', '', ...
        'detector_num', 0, ...
        'pixel_num', 0, ...
        'points_per_layer', 0, ...
        'rotate_num', 0, ...
        'sysmat_polar_bytes', 0, ...
        'sysmat_tmp_bytes', 0, ...
        'output_dir', '');
end


function cases = build_cases(runs_root, factors_root, rotate_num, output_suffix)
    combined_name = 'SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat';
    cross_name = 'Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat';

    specs = {
        'JSCC', 'A218',       'JSCC_218keV',                   combined_name, 218, 218, '';
        'JSCC', 'A440',       'JSCC_440keV',                   combined_name, 440, 440, '';
        'JSCC', 'C440to218',  'JSCC_440keV_to_218keVwin',     cross_name,    440, 218, '';
        'SPECTEHENaI', 'A218',      'EHE_PbNaI_218keV',              combined_name, 218, 218, '_SPECTEHENaI';
        'SPECTEHENaI', 'A440',      'EHE_PbNaI_440keV',              combined_name, 440, 440, '_SPECTEHENaI';
        'SPECTEHENaI', 'C440to218', 'EHE_PbNaI_440keV_to_218keVwin', cross_name,    440, 218, '_SPECTEHENaI'
    };

    cases = repmat(struct(), size(specs, 1), 1);
    for idx = 1:size(specs, 1)
        source_energy = specs{idx, 5};
        window_energy = specs{idx, 6};
        suffix = specs{idx, 7};
        if strcmp(specs{idx, 2}, 'C440to218')
            output_name = sprintf('%dkeV_to%dwin_RotateNum%d%s', ...
                source_energy, window_energy, rotate_num, suffix);
        else
            output_name = sprintf('%dkeV_RotateNum%d%s', ...
                source_energy, rotate_num, suffix);
        end

        run_dir = fullfile(runs_root, specs{idx, 3});
        cases(idx).system_tag = specs{idx, 1};
        cases(idx).response = specs{idx, 2};
        cases(idx).run_name = specs{idx, 3};
        cases(idx).source_energy_keV = source_energy;
        cases(idx).window_energy_keV = window_energy;
        cases(idx).matrix_kind = ternary(strcmp(specs{idx, 2}, 'C440to218'), ...
            'scatter_only_cross_window', 'photopeak_plus_scatter');
        cases(idx).sysmat_file = fullfile(run_dir, specs{idx, 4});
        cases(idx).params_detector_file = fullfile(run_dir, 'Params_Detector.dat');
        output_name = append_output_suffix(output_name, output_suffix);
        cases(idx).output_name = output_name;
        cases(idx).output_dir = fullfile(factors_root, output_name);
        cases(idx).calibration = response_calibration( ...
            cases(idx).system_tag, cases(idx).response);
    end
end


function summary = validate_factor_dir(factor_dir, item, rotate_num, grid_options)
    detector = readmatrix(fullfile(factor_dir, 'Detector.csv'));
    coor = readmatrix(fullfile(factor_dir, 'coor_polar_full.csv'));
    rotmat = readmatrix(fullfile(factor_dir, 'RotMat_full.csv'));
    rotmat_inv = readmatrix(fullfile(factor_dir, 'RotMatInv_full.csv'));
    detector_num = size(detector, 1);
    pixel_num = size(coor, 1);

    expected_points_per_layer = 1280 + double(grid_options.include_center_point);
    expected_pixel_num = expected_points_per_layer * 20;
    if pixel_num ~= expected_pixel_num
        error('run_gen_response_factors:BadPixelCount', ...
            '%s generated %d pixels; expected %d.', factor_dir, pixel_num, expected_pixel_num);
    end
    if ~isequal(size(rotmat), [pixel_num, rotate_num]) || ...
            ~isequal(size(rotmat_inv), [pixel_num, rotate_num])
        error('run_gen_response_factors:BadRotationShape', ...
            '%s has invalid rotation-map dimensions.', factor_dir);
    end
    if any(sort(rotmat(:, 1)) ~= (1:pixel_num).') || ...
            any(sort(rotmat_inv(:, 1)) ~= (1:pixel_num).')
        error('run_gen_response_factors:BadRotationPermutation', ...
            '%s rotation maps are not permutations.', factor_dir);
    end

    polar_info = dir(fullfile(factor_dir, 'SysMat_polar'));
    cart_info = dir(fullfile(factor_dir, 'SysMat_tmp'));
    expected_polar_bytes = double(detector_num) * double(pixel_num) * 4;
    expected_cart_bytes = double(detector_num) * 51 * 51 * 20 * 4;
    if isempty(polar_info) || double(polar_info.bytes) ~= expected_polar_bytes
        error('run_gen_response_factors:BadPolarSize', ...
            '%s SysMat_polar has the wrong byte count.', factor_dir);
    end
    if isempty(cart_info) || double(cart_info.bytes) ~= expected_cart_bytes
        error('run_gen_response_factors:BadCartesianSize', ...
            '%s SysMat_tmp has the wrong byte count.', factor_dir);
    end

    expected_detector_num = ternary(strcmp(item.system_tag, 'JSCC'), 10496, 2312);
    if detector_num ~= expected_detector_num
        error('run_gen_response_factors:BadDetectorCount', ...
            '%s generated %d detector rows; expected %d.', ...
            factor_dir, detector_num, expected_detector_num);
    end

    summary = struct();
    summary.system_tag = item.system_tag;
    summary.response = item.response;
    summary.detector_num = detector_num;
    summary.pixel_num = pixel_num;
    summary.points_per_layer = expected_points_per_layer;
    summary.rotate_num = rotate_num;
    summary.sysmat_polar_bytes = expected_polar_bytes;
    summary.sysmat_tmp_bytes = expected_cart_bytes;
end


function write_factor_manifest(factor_dir, item, summary, rotate_num, grid_options, output_suffix)
    input_info = dir(item.sysmat_file);
    manifest = struct();
    manifest.format_version = 1;
    manifest.system_tag = item.system_tag;
    manifest.response = item.response;
    manifest.response_notation = response_notation(item.response);
    manifest.source_energy_keV = item.source_energy_keV;
    manifest.window_energy_keV = item.window_energy_keV;
    manifest.matrix_kind = item.matrix_kind;
    manifest.per_emitted_source_photon = true;
    manifest.includes_225Ac_gamma_yield = false;
    manifest.rotate_num = rotate_num;
    manifest.detector_num = summary.detector_num;
    manifest.pixel_num = summary.pixel_num;
    manifest.grid = struct( ...
        'include_center_point', logical(grid_options.include_center_point), ...
        'points_per_layer', summary.points_per_layer, ...
        'center_point_index', ternary(grid_options.include_center_point, 1, 0), ...
        'output_suffix', char(string(output_suffix)));
    manifest.input_run = item.run_name;
    manifest.input_matrix = item.sysmat_file;
    manifest.input_matrix_bytes = double(input_info.bytes);
    manifest.cross_talk_usage = cross_talk_usage(item.response);
    manifest.calibration = item.calibration;
    manifest.branching_ratio_included = false;
    manifest.generated_at = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));

    text_value = jsonencode(manifest, 'PrettyPrint', true);
    file_id = fopen(fullfile(factor_dir, 'factor_manifest.json'), 'w');
    if file_id < 0
        error('Cannot write factor manifest in %s.', factor_dir);
    end
    cleanup = onCleanup(@() fclose(file_id));
    fwrite(file_id, text_value, 'char');
    fwrite(file_id, newline, 'char');
end


function output_name = append_output_suffix(output_name, output_suffix)
    suffix = strip(string(output_suffix));
    suffix = strip(suffix, 'left', '_');
    if strlength(suffix) > 0
        output_name = output_name + "_" + suffix;
    end
    output_name = char(output_name);
end


function install_factor_dir(staging_dir, output_dir)
    backup_dir = '';
    if isfolder(output_dir)
        timestamp = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmssSSS'));
        backup_dir = [output_dir '.backup_' timestamp];
        [ok, message] = movefile(output_dir, backup_dir);
        if ~ok
            error('Cannot move existing Factors directory to backup: %s', message);
        end
    end

    try
        [ok, message] = movefile(staging_dir, output_dir);
        if ~ok
            error('Cannot install generated Factors: %s', message);
        end
    catch exception
        if ~isempty(backup_dir) && isfolder(backup_dir) && ~isfolder(output_dir)
            movefile(backup_dir, output_dir);
        end
        rethrow(exception);
    end

    if ~isempty(backup_dir)
        rmdir(backup_dir, 's');
    end
end


function validate_response_geometry(cases)
    systems = unique({cases.system_tag}, 'stable');
    for system_idx = 1:numel(systems)
        selected = cases(strcmp({cases.system_tag}, systems{system_idx}));
        reference_detector = fileread(fullfile(selected(1).output_dir, 'Detector.csv'));
        reference_coor = fileread(fullfile(selected(1).output_dir, 'coor_polar_full.csv'));
        reference_rot = fileread(fullfile(selected(1).output_dir, 'RotMat_full.csv'));
        for idx = 2:numel(selected)
            if ~strcmp(reference_detector, fileread(fullfile(selected(idx).output_dir, 'Detector.csv'))) || ...
                    ~strcmp(reference_coor, fileread(fullfile(selected(idx).output_dir, 'coor_polar_full.csv'))) || ...
                    ~strcmp(reference_rot, fileread(fullfile(selected(idx).output_dir, 'RotMat_full.csv')))
                error('run_gen_response_factors:GeometryMismatch', ...
                    '%s response Factors do not share identical geometry/order.', systems{system_idx});
            end
        end
    end
end


function value = response_notation(response)
    switch response
        case 'A218'
            value = 'A(218-window <- 218-source)';
        case 'A440'
            value = 'A(440-window <- 440-source)';
        case 'C440to218'
            value = 'C(218-window <- 440-source)';
        otherwise
            error('Unknown response: %s', response);
    end
end


function value = cross_talk_usage(response)
    if strcmp(response, 'C440to218')
        value = 'Use only as the additive 440-source contribution in the 218-window forward model.';
    else
        value = 'Direct photopeak-window response.';
    end
end


function calibration = response_calibration(system_tag, response)
    calibration = struct();
    calibration.enabled = false;
    calibration.name = 'none';
    calibration.scope = 'none';
    calibration.source = '';
    calibration.layer_y_mm = [];
    calibration.layer_scale = [];
    calibration.expected_active_rows = [];
    calibration.center_point_only = false;

    if ~strcmp(system_tag, 'JSCC')
        return;
    end

    calibration.enabled = true;
    calibration.name = ['JSCC_' response '_G4Center_LayerScale_20260716'];
    calibration.scope = 'active_detector_rows';
    calibration.source = [ ...
        '1e9 Geant4 center-point response; detector-local position-integrated matrices'];
    calibration.layer_y_mm = [30, 60, 90, 120];
    calibration.expected_active_rows = [512, 768, 1024, 8192];
    calibration.center_point_only = true;

    switch response
        case 'A218'
            calibration.layer_scale = [0.8740232, 0.8793926, 0.8719679, 0.8708826];
        case 'A440'
            calibration.layer_scale = [0.8720313, 0.8912363, 0.8847924, 0.8691287];
        case 'C440to218'
            calibration.layer_scale = [1.1411090, 1.1636691, 1.2201934, 1.2372030];
        otherwise
            error('run_gen_response_factors:UnknownJSCCCalibration', ...
                'No JSCC calibration is defined for response %s.', response);
    end
end


function remove_tree_if_exists(path)
    if isfolder(path)
        rmdir(path, 's');
    end
end


function value = ternary(condition, true_value, false_value)
    if condition
        value = true_value;
    else
        value = false_value;
    end
end
