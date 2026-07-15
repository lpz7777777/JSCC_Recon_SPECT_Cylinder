function result = export_run_sysmat_for_amide(run_selector, matrix_selector, output_dir, varargin)
%EXPORT_RUN_SYSMAT_FOR_AMIDE Export a runs/* matrix in Factors/SysMat_tmp layout.
%
% result = export_run_sysmat_for_amide(RUN, MATRIX)
% result = export_run_sysmat_for_amide(RUN, MATRIX, OUTPUT_DIR, ...)
%
% RUN may be a runs/<case> directory or a case name such as
% 'EHE_PbNaI_440keV'. MATRIX may be a file name within RUN or an absolute
% path. The output is float32 little-endian with dimension order:
%
%   [image_x, image_y, image_z, selected_detector, rotation]
%
% By default, detector records with flag==1 are retained, matching the
% SysMat_tmp filtering performed by GenFactors/gen_factors.m.
%
% Name-value options:
%   FilterScintillator - retain only detector flag==1 (default true)
%   Overwrite          - replace an existing export (default false)
%   ChunkDetectors     - input detector rows per streaming block (default 64)

    if nargin < 2
        error('export_run_sysmat_for_amide:MissingInput', ...
            'Both run_selector and matrix_selector are required.');
    end
    if nargin < 3
        output_dir = '';
    end

    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'FilterScintillator', true, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'Overwrite', false, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'ChunkDetectors', 64, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 1 && fix(x) == x);
    parse(parser, varargin{:});
    options = parser.Results;

    tool_dir = fileparts(mfilename('fullpath'));
    repo_root = fileparts(tool_dir);
    run_dir = resolve_run_dir(repo_root, run_selector);
    matrix_path = resolve_matrix_path(run_dir, matrix_selector);

    detector_path = fullfile(run_dir, 'Params_Detector.dat');
    image_path = fullfile(run_dir, 'Params_Image.dat');
    require_file(detector_path, 'Params_Detector.dat');
    require_file(image_path, 'Params_Image.dat');
    require_file(matrix_path, 'system matrix');

    detector_raw = read_float32_file(detector_path);
    image_raw = read_float32_file(image_path);
    if numel(image_raw) < 7
        error('export_run_sysmat_for_amide:BadImageParams', ...
            'Params_Image.dat has only %d float32 values.', numel(image_raw));
    end

    num_detector = checked_integer(detector_raw(1), 'detector count');
    required_detector_values = 1 + 12 * num_detector;
    if numel(detector_raw) < required_detector_values
        error('export_run_sysmat_for_amide:BadDetectorParams', ...
            'Params_Detector.dat needs %d float32 values but contains %d.', ...
            required_detector_values, numel(detector_raw));
    end
    detector = reshape(detector_raw(2:required_detector_values), 12, num_detector).';

    image_size = [ ...
        checked_integer(image_raw(1), 'image X size'), ...
        checked_integer(image_raw(2), 'image Y size'), ...
        checked_integer(image_raw(3), 'image Z size')];
    voxel_size_mm = double(image_raw(4:6)).';
    num_rotation = checked_integer(image_raw(7), 'rotation count');
    num_voxel = prod(double(image_size));
    expected_elements = num_voxel * double(num_detector) * double(num_rotation);
    expected_bytes = expected_elements * 4;

    matrix_info = dir(matrix_path);
    if double(matrix_info.bytes) ~= expected_bytes
        error('export_run_sysmat_for_amide:WrongMatrixSize', ...
            ['Matrix is incomplete or belongs to different parameters. ' ...
             'Expected %.0f bytes, found %.0f bytes: %s'], ...
            expected_bytes, double(matrix_info.bytes), matrix_path);
    end

    if options.FilterScintillator
        selected_mask = round(detector(:, 12)) == 1;
        filter_description = 'Params_Detector flag == 1';
    else
        selected_mask = true(num_detector, 1);
        filter_description = 'all detector records';
    end
    selected_indices = find(selected_mask);
    num_selected = numel(selected_indices);
    if num_selected == 0
        error('export_run_sysmat_for_amide:NoSelectedDetector', ...
            'The selected detector filter retained zero rows.');
    end

    [~, run_name] = fileparts(run_dir);
    [~, matrix_stem] = fileparts(matrix_path);
    if isempty(output_dir)
        output_dir = fullfile(repo_root, 'AmideExports', run_name, matrix_stem);
    else
        output_dir = make_absolute(repo_root, output_dir);
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    output_path = fullfile(output_dir, 'SysMat_tmp');
    partial_path = fullfile(output_dir, 'SysMat_tmp.partial');
    if exist(output_path, 'file') && ~options.Overwrite
        error('export_run_sysmat_for_amide:OutputExists', ...
            'Output exists. Use ''Overwrite'', true to replace it: %s', output_path);
    end
    if exist(partial_path, 'file')
        if options.Overwrite
            delete(partial_path);
        else
            error('export_run_sysmat_for_amide:PartialExists', ...
                'A partial export exists. Use ''Overwrite'', true to replace it: %s', partial_path);
        end
    end

    fprintf('=====================================\n');
    fprintf('Export system matrix for AMIDE\n');
    fprintf('  Run:       %s\n', run_name);
    fprintf('  Matrix:    %s\n', matrix_path);
    fprintf('  Input:     [%d, %d, %d, %d, %d]\n', ...
        image_size(1), image_size(2), image_size(3), num_detector, num_rotation);
    fprintf('  Filter:    %s (%d/%d detector rows)\n', ...
        filter_description, num_selected, num_detector);
    fprintf('  Output:    %s\n', output_path);
    fprintf('=====================================\n');

    stats = stream_matrix(matrix_path, partial_path, num_voxel, num_detector, ...
        num_rotation, selected_mask, options.ChunkDetectors);
    output_elements = num_voxel * double(num_selected) * double(num_rotation);
    output_bytes = output_elements * 4;
    partial_info = dir(partial_path);
    if isempty(partial_info) || double(partial_info.bytes) ~= output_bytes
        if exist(partial_path, 'file'); delete(partial_path); end
        error('export_run_sysmat_for_amide:WrongOutputSize', ...
            'Expected %.0f output bytes but wrote %.0f.', ...
            output_bytes, get_file_bytes(partial_info));
    end
    if exist(output_path, 'file')
        delete(output_path);
    end
    movefile(partial_path, output_path, 'f');

    write_detector_map(fullfile(output_dir, 'DetectorIndex.csv'), ...
        detector, selected_indices);

    metadata = struct();
    metadata.format = 'float32 little-endian raw';
    metadata.dimension_order = {'image_x', 'image_y', 'image_z', ...
        'selected_detector', 'rotation'};
    metadata.dimensions = [double(image_size), double(num_selected), double(num_rotation)];
    metadata.amide_dimensions_xyzt = [double(image_size), ...
        double(num_selected) * double(num_rotation)];
    metadata.voxel_size_mm = voxel_size_mm;
    metadata.byte_offset = 0;
    metadata.scale_factor = 1;
    metadata.run_name = run_name;
    metadata.input_matrix = matrix_path;
    metadata.input_dimensions = [double(image_size), double(num_detector), double(num_rotation)];
    metadata.input_bytes = expected_bytes;
    metadata.detector_filter = filter_description;
    metadata.selected_detector_count = num_selected;
    metadata.output_file = output_path;
    metadata.output_bytes = output_bytes;
    metadata.nonfinite_replaced_with_zero = stats.nonfinite;
    metadata.negative_values = stats.negative;
    metadata.nonzero_values = stats.nonzero;
    metadata.minimum = stats.minimum;
    metadata.maximum = stats.maximum;

    write_metadata_text(fullfile(output_dir, 'AMIDE_IMPORT.txt'), metadata);
    write_metadata_json(fullfile(output_dir, 'metadata.json'), metadata);
    save(fullfile(output_dir, 'metadata.mat'), 'metadata');

    result = metadata;
    result.detector_index_file = fullfile(output_dir, 'DetectorIndex.csv');
    result.metadata_text_file = fullfile(output_dir, 'AMIDE_IMPORT.txt');

    fprintf('Export complete: %.0f bytes\n', output_bytes);
    fprintf('  Nonzero=%g, min=%.9g, max=%.9g, cleaned nonfinite=%g\n', ...
        stats.nonzero, stats.minimum, stats.maximum, stats.nonfinite);
    fprintf('  AMIDE settings: X=%d Y=%d Z=%d T=%d, float32, little-endian, offset=0\n', ...
        image_size(1), image_size(2), image_size(3), num_selected * num_rotation);
end


function stats = stream_matrix(input_path, partial_path, num_voxel, num_detector, ...
        num_rotation, selected_mask, chunk_detectors)
    input_id = fopen(input_path, 'rb', 'ieee-le');
    if input_id < 0
        error('export_run_sysmat_for_amide:OpenInput', 'Cannot open %s.', input_path);
    end
    output_id = fopen(partial_path, 'wb', 'ieee-le');
    if output_id < 0
        fclose(input_id);
        error('export_run_sysmat_for_amide:OpenOutput', 'Cannot open %s.', partial_path);
    end

    stats = struct('nonfinite', 0, 'negative', 0, 'nonzero', 0, ...
        'minimum', inf, 'maximum', -inf);
    total_chunks = num_rotation * ceil(num_detector / chunk_detectors);
    completed_chunks = 0;
    next_report = 10;

    try
        for rotation = 1:num_rotation
            for first_detector = 1:chunk_detectors:num_detector
                detector_count = min(chunk_detectors, num_detector - first_detector + 1);
                [block, count] = fread(input_id, [num_voxel, detector_count], 'single=>single');
                expected_count = num_voxel * detector_count;
                if count ~= expected_count
                    error('export_run_sysmat_for_amide:ShortRead', ...
                        'Expected %.0f values but read %.0f at detector %d, rotation %d.', ...
                        expected_count, count, first_detector, rotation);
                end

                local_mask = selected_mask(first_detector:first_detector + detector_count - 1);
                selected = block(:, local_mask);
                bad = ~isfinite(selected);
                bad_count = nnz(bad);
                if bad_count > 0
                    selected(bad) = 0;
                    stats.nonfinite = stats.nonfinite + bad_count;
                end
                stats.negative = stats.negative + nnz(selected < 0);
                stats.nonzero = stats.nonzero + nnz(selected);
                if ~isempty(selected)
                    stats.minimum = min(stats.minimum, double(min(selected(:))));
                    stats.maximum = max(stats.maximum, double(max(selected(:))));
                end
                written = fwrite(output_id, selected, 'single');
                if written ~= numel(selected)
                    error('export_run_sysmat_for_amide:ShortWrite', ...
                        'Expected to write %.0f values but wrote %.0f.', numel(selected), written);
                end

                completed_chunks = completed_chunks + 1;
                percent = floor(100 * completed_chunks / total_chunks);
                if percent >= next_report
                    fprintf('  Progress: %d%%\n', percent);
                    next_report = next_report + 10;
                end
            end
        end
        fclose(input_id);
        fclose(output_id);
    catch exception
        fclose(input_id);
        fclose(output_id);
        if exist(partial_path, 'file'); delete(partial_path); end
        rethrow(exception);
    end

    if isinf(stats.minimum)
        stats.minimum = 0;
        stats.maximum = 0;
    end
end


function run_dir = resolve_run_dir(repo_root, selector)
    selector = char(selector);
    if isfolder(selector)
        run_dir = make_absolute(pwd, selector);
    else
        run_dir = fullfile(repo_root, 'runs', selector);
    end
    if ~isfolder(run_dir)
        error('export_run_sysmat_for_amide:RunNotFound', ...
            'Run directory not found: %s', run_dir);
    end
end


function matrix_path = resolve_matrix_path(run_dir, selector)
    selector = char(selector);
    if isfile(selector)
        matrix_path = make_absolute(pwd, selector);
    else
        matrix_path = fullfile(run_dir, selector);
    end
end


function path = make_absolute(base_dir, path)
    path = char(path);
    if isempty(path); path = base_dir; return; end
    if ~is_absolute_path(path)
        path = fullfile(base_dir, path);
    end
end


function result = is_absolute_path(path)
    % Windows absolute paths can begin with a drive letter or a backslash.
    % This also covers UNC paths, so all public path arguments are usable.
    result = startsWith(path, '/') || startsWith(path, '\') || ...
        ~isempty(regexp(path, '^[A-Za-z]:[\\/]', 'once'));
end


function values = read_float32_file(path)
    file_id = fopen(path, 'rb', 'ieee-le');
    if file_id < 0
        error('export_run_sysmat_for_amide:OpenParams', 'Cannot open %s.', path);
    end
    values = fread(file_id, inf, 'single=>single');
    fclose(file_id);
end


function value = checked_integer(raw, description)
    value = round(double(raw));
    if ~isfinite(value) || value < 1 || abs(double(raw) - value) > 1e-3
        error('export_run_sysmat_for_amide:BadDimension', ...
            'Invalid %s: %.9g.', description, double(raw));
    end
end


function require_file(path, description)
    if ~isfile(path)
        error('export_run_sysmat_for_amide:MissingFile', ...
            'Missing %s: %s', description, path);
    end
end


function bytes = get_file_bytes(info)
    if isempty(info); bytes = 0; else; bytes = double(info.bytes); end
end


function write_detector_map(path, detector, selected_indices)
    file_id = fopen(path, 'w');
    if file_id < 0
        error('export_run_sysmat_for_amide:DetectorMap', 'Cannot write %s.', path);
    end
    fprintf(file_id, 'output_index,input_detector_index,x_mm,y_local_mm,z_mm,flag\n');
    for output_index = 1:numel(selected_indices)
        input_index = selected_indices(output_index);
        fprintf(file_id, '%d,%d,%.9g,%.9g,%.9g,%.9g\n', ...
            output_index, input_index, detector(input_index, 1), ...
            detector(input_index, 2), detector(input_index, 3), detector(input_index, 12));
    end
    fclose(file_id);
end


function write_metadata_text(path, metadata)
    file_id = fopen(path, 'w');
    if file_id < 0
        error('export_run_sysmat_for_amide:MetadataText', 'Cannot write %s.', path);
    end
    dims = metadata.dimensions;
    amide_dims = metadata.amide_dimensions_xyzt;
    voxel = metadata.voxel_size_mm;
    fprintf(file_id, 'AMIDE raw data import settings\n\n');
    fprintf(file_id, 'Data file: SysMat_tmp\n');
    fprintf(file_id, 'Data type: 32-bit floating point (float32)\n');
    fprintf(file_id, 'Byte order: little-endian\n');
    fprintf(file_id, 'Header/byte offset: 0\n');
    fprintf(file_id, 'Scale factor: 1\n');
    fprintf(file_id, 'Dimensions in file [X Y Z detector rotation]: [%d %d %d %d %d]\n', dims);
    fprintf(file_id, 'AMIDE raw dimensions [X Y Z T]: [%d %d %d %d]\n', amide_dims);
    fprintf(file_id, 'Voxel size [X Y Z] mm: [%.9g %.9g %.9g]\n', voxel);
    fprintf(file_id, 'T frame index = detector + selected_detector_count * rotation (zero-based).\n');
    fprintf(file_id, 'DetectorIndex.csv maps each detector frame to its original detector row.\n');
    fprintf(file_id, 'Input matrix: %s\n', metadata.input_matrix);
    fprintf(file_id, 'Detector filter: %s\n', metadata.detector_filter);
    fprintf(file_id, 'Nonfinite values replaced with zero: %.0f\n', metadata.nonfinite_replaced_with_zero);
    fprintf(file_id, 'Negative values: %.0f\n', metadata.negative_values);
    fprintf(file_id, 'Minimum: %.9g\nMaximum: %.9g\n', metadata.minimum, metadata.maximum);
    fclose(file_id);
end


function write_metadata_json(path, metadata)
    try
        text = jsonencode(metadata, 'PrettyPrint', true);
    catch
        text = jsonencode(metadata);
    end
    file_id = fopen(path, 'w');
    if file_id < 0
        error('export_run_sysmat_for_amide:MetadataJson', 'Cannot write %s.', path);
    end
    fwrite(file_id, text, 'char');
    fwrite(file_id, newline, 'char');
    fclose(file_id);
end
