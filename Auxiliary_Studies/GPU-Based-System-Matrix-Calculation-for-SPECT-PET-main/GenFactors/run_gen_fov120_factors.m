function results = run_gen_fov120_factors()
% Isolated raw FOV120 Factors. Calibration is fitted later from fresh MC.
    engine = fileparts(fileparts(mfilename('fullpath')));
    repo = fileparts(fileparts(engine));
    cfg = jsondecode(fileread(fullfile(repo, 'experiments', 'FOV120', 'config.json')));
    dz = cfg.z_spacing_mm;
    h = cfg.height_mm;
    grid = struct('include_center_point', true, ...
        'apply_polar_volume_weighting', true, 'write_cartesian_tmp', false, ...
        'run_name_suffix', '_pe_v4_FOV120', 'calibration_profile', 'none', ...
        'z_axis', (-h/2+dz/2):dz:(h/2-dz/2), ...
        'factors_root', fullfile(repo, 'experiments', 'FOV120', 'generated', 'FactorsRaw'));
    results = run_gen_response_factors(["JSCC/A218", "JSCC/A440", "JSCC/C440to218"], "", grid);
end
