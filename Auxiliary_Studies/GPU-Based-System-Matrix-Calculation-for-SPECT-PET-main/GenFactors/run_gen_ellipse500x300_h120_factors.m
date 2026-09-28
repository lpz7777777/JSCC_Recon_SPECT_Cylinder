function results = run_gen_ellipse500x300_h120_factors()
% Convert the three isolated, uncalibrated ellipse experiment matrices.
    engine = fileparts(fileparts(mfilename('fullpath')));
    repo = fileparts(fileparts(engine));
    folder = fullfile(repo, 'experiments', 'ELLIPSE500x300_H120');
    cfg = jsondecode(fileread(fullfile(folder, 'config.json')));
    r = 6:6:252;
    theta = zeros(size(r));
    for i = 1:numel(r)
        if r(i) <= 150
            theta(i) = 20 * ceil(i / (25/4));
        else
            selected = cfg.outer_ring_counts(:,1) <= r(i) & ...
                       r(i) <= cfg.outer_ring_counts(:,2);
            assert(sum(selected) == 1, 'Missing or ambiguous outer ring');
            theta(i) = cfg.outer_ring_counts(selected,3);
        end
    end
    dz = cfg.z_spacing_mm;
    h = cfg.height_mm;
    grid = struct('include_center_point', true, ...
        'apply_polar_volume_weighting', true, 'write_cartesian_tmp', false, ...
        'run_name_suffix', '_pe_v4_ELLIPSE500x300_H120', ...
        'calibration_profile', 'none', ...
        'z_axis', (-h/2+dz/2):dz:(h/2-dz/2), ...
        'xy_axis', cfg.xy_axis_mm(1):cfg.xy_axis_mm(2):cfg.xy_axis_mm(3), ...
        'radial_centers_mm', r, 'theta_per_ring', theta, ...
        'factors_root', fullfile(folder, 'generated', 'FactorsRaw'));
    assert(sum(theta)+1 == cfg.points_per_layer, 'Grid count mismatch');
    results = run_gen_response_factors( ...
        ["JSCC/A218", "JSCC/A440", "JSCC/C440to218"], "", grid);
end
