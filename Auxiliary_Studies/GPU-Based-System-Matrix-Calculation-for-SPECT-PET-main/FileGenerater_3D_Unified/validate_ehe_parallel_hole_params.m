function report = validate_ehe_parallel_hole_params(output_root, write_visuals)
%VALIDATE_EHE_PARALLEL_HOLE_PARAMS Validate serialized EHE Pb/NaI parameters.
% The checks operate on Params_*.dat, so file truncation and layout errors are
% covered in addition to the geometric construction itself. Plot export is
% optional because graphics backends can block in headless batch sessions.

    this_dir = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(output_root)
        output_root = fullfile(this_dir, 'output');
    end
    if nargin < 2 || isempty(write_visuals)
        write_visuals = false;
    end
    output_root = char(output_root);
    cfg = config_geometry();
    cfg.geometry_type = 'ConventionalSPECT';
    expected_front_y = cfg.fov.common_front_face_y;
    expected_ehe_origin_y = cfg.conv.fov2collimator0;

    case_names = {'EHE_PbNaI_218keV', 'EHE_PbNaI_440keV', ...
                  'EHE_PbNaI_440keV_to_218keVwin'};
    expected_energy = [218, 440, 440];
    expected_forced = [0, 0, 1];
    expected_combined = [1, 1, 0];
    expected_local_scatter = [1, 1; 1, 1; 1, 0];

    cases = repmat(struct(), 1, numel(case_names));
    for i = 1:numel(case_names)
        case_dir = fullfile(output_root, case_names{i});
        cases(i).name = case_names{i};
        cases(i).det_raw = read_float32(fullfile(case_dir, 'Params_Detector.dat'));
        cases(i).col_raw = read_float32(fullfile(case_dir, 'Params_Collimator.dat'));
        cases(i).img = read_float32(fullfile(case_dir, 'Params_Image.dat'));
        cases(i).phy = read_float32(fullfile(case_dir, 'Params_Physics.dat'));

        cases(i).num_detectors = round(cases(i).det_raw(1));
        cases(i).num_holes = round(cases(i).col_raw(11));
        must(numel(cases(i).det_raw) == 1 + 12 * cases(i).num_detectors, ...
             '%s detector file length is inconsistent with its header.', case_names{i});
        must(numel(cases(i).col_raw) == 100 + 9 * cases(i).num_holes, ...
             '%s collimator file length is inconsistent with its header.', case_names{i});
        must(numel(cases(i).img) == 12, '%s image file must contain 12 floats.', case_names{i});
        must(numel(cases(i).phy) == 12, '%s physics file must contain 12 floats.', case_names{i});

        cases(i).det = reshape(cases(i).det_raw(2:end), 12, []).';
        cases(i).holes = reshape(cases(i).col_raw(101:end), 9, []).';
        must(cases(i).num_detectors == prod(cfg.conv.unit_num), ...
             '%s detector count mismatch.', case_names{i});
        must(cases(i).num_holes == cfg.conv.collimator.hole_rows * cfg.conv.collimator.hole_cols, ...
             '%s hole count mismatch.', case_names{i});
        must(abs(cases(i).phy(8) - expected_energy(i)) < 1e-5, '%s source energy mismatch.', case_names{i});
        must(cases(i).phy(5) == expected_forced(i), '%s forced-window flag mismatch.', case_names{i});
        must(cases(i).phy(4) == expected_combined(i), '%s combined-matrix flag mismatch.', case_names{i});
        must(isequal(reshape(cases(i).phy(11:12), 1, []), expected_local_scatter(i, :)), ...
             '%s detector-local scatter switches mismatch.', case_names{i});
        must(abs(cases(i).img(12) - expected_ehe_origin_y) < 1e-5, ...
             '%s EHE local-Y origin does not preserve the JSCC front-face reference.', case_names{i});
    end

    % Geometry must be identical among energy/window cases. Attenuation and
    % energy-resolution fields are expected to differ.
    det_geometry_cols = [1:6, 11:12];
    hole_geometry_cols = [1:5, 9];
    for i = 2:numel(cases)
        must(isequal(cases(1).det(:, det_geometry_cols), cases(i).det(:, det_geometry_cols)), ...
             'Detector geometry differs between %s and %s.', cases(1).name, cases(i).name);
        must(isequal(cases(1).holes(:, hole_geometry_cols), cases(i).holes(:, hole_geometry_cols)), ...
             'Collimator geometry differs between %s and %s.', cases(1).name, cases(i).name);
        must(isequal(cases(1).img, cases(i).img), 'Image geometry differs between cases.');
    end

    det = cases(1).det;
    holes = cases(1).holes;
    col = cases(1).col_raw;
    radius = cfg.conv.collimator.hole_diameter / 2;
    expected_pitch = cfg.conv.collimator.hole_diameter + cfg.conv.collimator.septal_thickness;
    centers = holes(:, [1, 4]);
    nearest = nearest_neighbor_distances(centers);
    measured_pitch = median(nearest);
    measured_septum = min(nearest) - 2 * radius;

    must(all(abs(holes(:, 5) - radius) < 1e-6), 'Hole radii do not match the configured diameter.');
    must(abs(measured_pitch - expected_pitch) < 1e-4, 'Triangular-lattice pitch mismatch.');
    must(abs(measured_septum - cfg.conv.collimator.septal_thickness) < 1e-4, ...
         'Minimum edge-to-edge septal thickness mismatch.');
    must(all(nearest >= 2 * radius), 'At least two collimator holes overlap.');
    must(all(abs(centers(:, 1)) + radius <= col(12)/2 + 1e-5), 'A hole exceeds the collimator X boundary.');
    must(all(abs(centers(:, 2)) + radius <= col(14)/2 + 1e-5), 'A hole exceeds the collimator Z boundary.');
    must(abs((holes(1, 3) - holes(1, 2)) - cfg.conv.collimator_thickness) < 1e-5, ...
         'Hole length does not equal collimator thickness.');
    must(all(abs(holes(:, 6:8)) < 1e-7, 'all'), ...
         'Collimator apertures must contain air/vacuum, but a hole attenuation coefficient is nonzero.');
    must(col(16) > 0 && col(17) > 0 && col(18) > 0, ...
         'The EHE collimator plate must retain nonzero Pb attenuation coefficients.');
    must(all(det(:, 12) == 1), 'Conventional detector contains non-scintillator flags.');
    must(all(abs(det(:, 4:6) - cfg.conv.unit_size) < 1e-6, 'all'), 'Detector crystal dimensions mismatch.');

    detector_front_y = min(det(:, 2) - det(:, 5)/2);
    collimator_back_y = max(holes(:, 3));
    measured_gap = detector_front_y - collimator_back_y;
    must(abs(measured_gap - cfg.conv.detector_gap_y) < 1e-5, 'Detector-to-collimator gap mismatch.');

    absolute_collimator_front_y = cases(1).img(12) + min(holes(:, 2));
    absolute_collimator_back_y = cases(1).img(12) + max(holes(:, 3));
    fov_y_spacing = cfg.fov.y_axis(2) - cfg.fov.y_axis(1);
    fov_positive_boundary_y = max(cfg.fov.y_axis) + fov_y_spacing / 2 + cfg.fov.shift_y;
    fov_to_collimator_clearance = absolute_collimator_front_y - fov_positive_boundary_y;
    must(abs(absolute_collimator_front_y - expected_front_y) < 1e-5, ...
         'EHE collimator front face does not match the JSCC first-layer detector front face.');
    must(fov_to_collimator_clearance > 0, ...
         'EHE collimator overlaps the positive-Y boundary of the image FOV.');

    report = struct();
    report.status = 'PASS';
    report.detector_count = size(det, 1);
    report.hole_count = size(holes, 1);
    report.hole_diameter_mm = 2 * radius;
    report.expected_center_pitch_mm = expected_pitch;
    report.measured_nearest_pitch_min_mm = min(nearest);
    report.measured_nearest_pitch_median_mm = measured_pitch;
    report.measured_nearest_pitch_max_mm = max(nearest);
    report.measured_minimum_septum_mm = measured_septum;
    report.collimator_size_x_y_z_mm = [col(12), col(13), col(14)];
    report.hole_center_x_range_mm = [min(centers(:, 1)), max(centers(:, 1))];
    report.hole_center_z_range_mm = [min(centers(:, 2)), max(centers(:, 2))];
    report.detector_center_x_range_mm = [min(det(:, 1)), max(det(:, 1))];
    report.detector_center_z_range_mm = [min(det(:, 3)), max(det(:, 3))];
    report.detector_to_collimator_gap_mm = measured_gap;
    report.shared_jscc_detector_ehe_collimator_front_mm = expected_front_y;
    report.fov_positive_boundary_mm = fov_positive_boundary_y;
    report.fov_to_collimator_clearance_mm = fov_to_collimator_clearance;
    report.fov_to_collimator_reference_mm = cases(1).img(12);
    report.fov_to_collimator_front_mm = absolute_collimator_front_y;
    report.fov_to_collimator_back_mm = absolute_collimator_back_y;
    report.physics_218 = cases(1).phy(:).';
    report.physics_440 = cases(2).phy(:).';
    report.physics_440_to_218_window = cases(3).phy(:).';

    validation_dir = fullfile(output_root, 'EHE_validation');
    if ~exist(validation_dir, 'dir')
        mkdir(validation_dir);
    end
    write_report(fullfile(validation_dir, 'EHE_parallel_hole_validation.txt'), report);
    if write_visuals
        write_visualization(fullfile(validation_dir, 'EHE_parallel_hole_validation.png'), ...
                            fullfile(validation_dir, 'EHE_parallel_hole_validation.pdf'), ...
                            cfg, cases(1), nearest, measured_septum);
    end

    fprintf('EHE parallel-hole validation: PASS\n');
    fprintf('  detectors=%d, holes=%d, pitch=%.6f mm, minimum septum=%.6f mm\n', ...
            report.detector_count, report.hole_count, measured_pitch, measured_septum);
    fprintf('  shared front=%.6f mm, FOV clearance=%.6f mm\n', ...
            report.shared_jscc_detector_ehe_collimator_front_mm, ...
            report.fov_to_collimator_clearance_mm);
    fprintf('  results: %s\n', validation_dir);
end


function values = read_float32(path)
    fid = fopen(path, 'rb');
    if fid < 0
        error('validate_ehe_parallel_hole_params:MissingFile', 'Cannot open %s.', path);
    end
    cleaner = onCleanup(@() fclose(fid));
    values = fread(fid, inf, 'float32=>double').';
end


function nearest = nearest_neighbor_distances(points)
    n = size(points, 1);
    nearest = inf(n, 1);
    for i = 1:n
        delta = points - points(i, :);
        d2 = sum(delta .* delta, 2);
        d2(i) = inf;
        nearest(i) = sqrt(min(d2));
    end
end


function must(condition, message, varargin)
    if ~condition
        error('validate_ehe_parallel_hole_params:ValidationFailed', message, varargin{:});
    end
end


function write_report(path, r)
    fid = fopen(path, 'w');
    if fid < 0
        error('Cannot write validation report %s.', path);
    end
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, 'EHE parallel-hole Params validation\n');
    fprintf(fid, 'status = %s\n', r.status);
    fprintf(fid, 'detector_count = %d\n', r.detector_count);
    fprintf(fid, 'hole_count = %d\n', r.hole_count);
    fprintf(fid, 'hole_diameter_mm = %.9g\n', r.hole_diameter_mm);
    fprintf(fid, 'expected_center_pitch_mm = %.9g\n', r.expected_center_pitch_mm);
    fprintf(fid, 'measured_nearest_pitch_min_median_max_mm = %.9g, %.9g, %.9g\n', ...
            r.measured_nearest_pitch_min_mm, r.measured_nearest_pitch_median_mm, r.measured_nearest_pitch_max_mm);
    fprintf(fid, 'measured_minimum_septum_mm = %.9g\n', r.measured_minimum_septum_mm);
    fprintf(fid, 'collimator_size_x_y_z_mm = %.9g, %.9g, %.9g\n', r.collimator_size_x_y_z_mm);
    fprintf(fid, 'hole_center_x_range_mm = %.9g, %.9g\n', r.hole_center_x_range_mm);
    fprintf(fid, 'hole_center_z_range_mm = %.9g, %.9g\n', r.hole_center_z_range_mm);
    fprintf(fid, 'detector_center_x_range_mm = %.9g, %.9g\n', r.detector_center_x_range_mm);
    fprintf(fid, 'detector_center_z_range_mm = %.9g, %.9g\n', r.detector_center_z_range_mm);
    fprintf(fid, 'detector_to_collimator_gap_mm = %.9g\n', r.detector_to_collimator_gap_mm);
    fprintf(fid, 'shared_jscc_detector_ehe_collimator_front_mm = %.9g\n', ...
            r.shared_jscc_detector_ehe_collimator_front_mm);
    fprintf(fid, 'fov_positive_boundary_mm = %.9g\n', r.fov_positive_boundary_mm);
    fprintf(fid, 'fov_to_collimator_clearance_mm = %.9g\n', r.fov_to_collimator_clearance_mm);
    fprintf(fid, 'fov_to_collimator_reference_mm = %.9g\n', r.fov_to_collimator_reference_mm);
    fprintf(fid, 'fov_to_collimator_front_mm = %.9g\n', r.fov_to_collimator_front_mm);
    fprintf(fid, 'fov_to_collimator_back_mm = %.9g\n', r.fov_to_collimator_back_mm);
    fprintf(fid, 'Physics_218 = %s\n', mat2str(r.physics_218, 9));
    fprintf(fid, 'Physics_440 = %s\n', mat2str(r.physics_440, 9));
    fprintf(fid, 'Physics_440_to_218_window = %s\n', mat2str(r.physics_440_to_218_window, 9));
    fprintf(fid, '\nInterpretation: septal_thickness is the nearest edge-to-edge Pb thickness.\n');
    fprintf(fid, 'Therefore center_pitch = hole_diameter + septal_thickness.\n');
end


function write_visualization(png_path, pdf_path, cfg, data, nearest, measured_septum)
    holes = data.holes;
    det = data.det;
    img = data.img;
    x = holes(:, 1);
    z = holes(:, 4);
    r = holes(:, 5);
    f = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1500, 1000]);
    cleaner = onCleanup(@() close(f));
    tl = tiledlayout(f, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'EHE Parallel-Hole Params Validation', 'FontWeight', 'bold');

    ax = nexttile(tl, 1);
    scatter(ax, x, z, 10, 'filled', 'MarkerFaceColor', [0.15, 0.42, 0.65]);
    rectangle(ax, 'Position', [-data.col_raw(12)/2, -data.col_raw(14)/2, data.col_raw(12), data.col_raw(14)], ...
              'EdgeColor', [0.15, 0.15, 0.15], 'LineWidth', 1.5);
    axis(ax, 'equal'); grid(ax, 'on'); xlabel(ax, 'X (mm)'); ylabel(ax, 'Z (mm)');
    title(ax, sprintf('All %d hole centers and plate boundary', size(holes, 1)));

    ax = nexttile(tl, 2);
    [~, center_id] = min(x.^2 + z.^2);
    d = hypot(x - x(center_id), z - z(center_id));
    ids = find(d <= 8);
    hold(ax, 'on');
    theta = linspace(0, 2*pi, 100);
    for id = ids(:).'
        plot(ax, x(id) + r(id)*cos(theta), z(id) + r(id)*sin(theta), 'Color', [0.1, 0.35, 0.6]);
        plot(ax, x(id), z(id), '.', 'Color', [0.1, 0.1, 0.1]);
    end
    axis(ax, 'equal'); grid(ax, 'on'); xlabel(ax, 'X (mm)'); ylabel(ax, 'Z (mm)');
    title(ax, sprintf('Central unit cells: diameter %.1f, pitch %.1f, septum %.1f mm', ...
          2*r(1), median(nearest), measured_septum));

    ax = nexttile(tl, 3);
    histogram(ax, nearest, 30, 'FaceColor', [0.25, 0.55, 0.35]);
    xline(ax, cfg.conv.collimator.hole_diameter + cfg.conv.collimator.septal_thickness, ...
          '--r', 'configured pitch', 'LineWidth', 1.5);
    grid(ax, 'on'); xlabel(ax, 'Nearest-neighbor center distance (mm)'); ylabel(ax, 'Hole count');
    title(ax, 'Nearest-neighbor consistency');

    ax = nexttile(tl, 4);
    fov_front = 0;
    col_front = img(12) + min(holes(:, 2));
    col_back = img(12) + max(holes(:, 3));
    detector_front = img(12) + min(det(:, 2) - det(:, 5)/2);
    detector_back = img(12) + max(det(:, 2) + det(:, 5)/2);
    hold(ax, 'on');
    patch(ax, [fov_front fov_front col_front col_front], [-1 1 1 -1], [0.3 0.7 0.4], ...
          'FaceAlpha', 0.18, 'EdgeColor', 'none');
    patch(ax, [col_front col_back col_back col_front], [-0.75 -0.75 0.75 0.75], [0.25 0.25 0.28], ...
          'FaceAlpha', 0.75, 'EdgeColor', 'k');
    patch(ax, [detector_front detector_back detector_back detector_front], [-0.6 -0.6 0.6 0.6], ...
          [0.95 0.72 0.15], 'FaceAlpha', 0.9, 'EdgeColor', 'k');
    xline(ax, col_front, ':k'); xline(ax, col_back, ':k');
    ylim(ax, [-1.2, 1.2]); xlim(ax, [0, detector_back + 20]); grid(ax, 'on');
    xlabel(ax, 'Global Y from FOV center (mm)'); yticks(ax, []);
    title(ax, sprintf('Side profile: FOV-to-front %.1f mm, collimator %.1f mm, NaI %.1f mm', ...
          col_front, cfg.conv.collimator_thickness, cfg.conv.unit_size(2)));
    legend(ax, {'FOV-to-collimator space', 'Pb collimator', 'NaI detector'}, 'Location', 'southoutside');

    % exportgraphics with a vector PDF can hang in headless MATLAB sessions.
    % print uses the noninteractive renderer and is stable for batch generation.
    print(f, png_path, '-dpng', '-r180');
    print(f, pdf_path, '-dpdf', '-vector', '-bestfit');
end
