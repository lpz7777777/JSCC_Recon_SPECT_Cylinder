function validate_local_scatter_params()
% Validate the two detector-local scatter flags without touching runs/.

    root = fileparts(fileparts(mfilename('fullpath')));
    generator_dir = fullfile(root, 'FileGenerater_3D_Unified');
    addpath(generator_dir);

    cfg = config_geometry();
    cfg.fov.x_axis = single([-0.5, 0.5]);
    cfg.fov.y_axis = single([-0.5, 0.5]);
    cfg.fov.z_axis = single([-0.5, 0.5]);
    cfg.enable_compton = true;
    cfg.save_compton_only = true;
    cfg.save_combined_sysmat = true;
    cfg.use_same_energy_window = false;
    cfg.energy_window_lower_keV = 0;
    cfg.energy_window_upper_keV = 0;
    cfg.compute_geo_relationship = true;

    detector = single([1, 0, 10, 0, 4, 10, 4, ...
        0.0938786, 0.0580227, 0.0358559, 0.199033216, 0, 1]);
    collimator = single(zeros(1, 109));

    temporary_root = tempname;
    cleanup = onCleanup(@() remove_temporary_root(temporary_root));
    mkdir(temporary_root);

    direct_dir = fullfile(temporary_root, 'direct');
    cfg.enable_detector_recoil_escape = true;
    cfg.enable_self_scatter_photopeak = true;
    write_dat_files(direct_dir, cfg, 218, detector, collimator);
    direct = read_float32(fullfile(direct_dir, 'Params_Physics.dat'));
    assert(numel(direct) == 12, 'Direct Params_Physics.dat must contain 12 float32 values.');
    assert(all(direct(11:12) == single([1; 1])), ...
        'Direct response must enable recoil-escape and same-crystal Compton+PE.');

    cross_dir = fullfile(temporary_root, 'cross');
    cfg.save_combined_sysmat = false;
    cfg.use_same_energy_window = true;
    cfg.energy_window_lower_keV = 196.30538;
    cfg.energy_window_upper_keV = 239.69462;
    cfg.enable_detector_recoil_escape = true;
    cfg.enable_self_scatter_photopeak = false;
    write_dat_files(cross_dir, cfg, 440, detector, collimator);
    cross = read_float32(fullfile(cross_dir, 'Params_Physics.dat'));
    assert(numel(cross) == 12, 'Cross Params_Physics.dat must contain 12 float32 values.');
    assert(all(cross(11:12) == single([1; 0])), ...
        '440-to-218 response must enable recoil-escape and disable same-crystal Compton+PE.');
    assert(cross(5) == 1 && abs(double(cross(6)) - 196.30538) < 1e-4 ...
        && abs(double(cross(7)) - 239.69462) < 1e-4 && cross(8) == 440, ...
        '440-to-218 source energy or forced energy window is incorrect.');

    jscc_source = fileread(fullfile(generator_dir, ...
        'generate_jscc_218_440_response_params.m'));
    ehe_source = fileread(fullfile(generator_dir, ...
        'generate_ehe_pb_nai_218_440_response_params.m'));
    validate_generator_switch_defaults(jscc_source, 'JSCC');
    validate_generator_switch_defaults(ehe_source, 'EHE');

    fprintf(['PASS validate_local_scatter_params: direct flags=[%g %g], ' ...
        'cross flags=[%g %g], serialized bytes=48\n'], ...
        direct(11), direct(12), cross(11), cross(12));
end


function values = read_float32(path)
    info = dir(path);
    assert(~isempty(info) && info.bytes == 48, ...
        'Params_Physics.dat must be exactly 48 bytes.');
    fid = fopen(path, 'rb');
    assert(fid >= 0, 'Cannot open %s.', path);
    cleanup = onCleanup(@() fclose(fid));
    values = fread(fid, inf, 'single=>single');
end


function validate_generator_switch_defaults(source, name)
    required = {
        'cases(1).enable_detector_recoil_escape = true;', ...
        'cases(1).enable_self_scatter_photopeak = true;', ...
        'cases(2).enable_detector_recoil_escape = true;', ...
        'cases(2).enable_self_scatter_photopeak = true;', ...
        'cases(3).enable_detector_recoil_escape = true;', ...
        'cases(3).enable_self_scatter_photopeak = false;', ...
        'cfg.enable_detector_recoil_escape = case_cfg.enable_detector_recoil_escape;', ...
        'cfg.enable_self_scatter_photopeak = case_cfg.enable_self_scatter_photopeak;'};
    for idx = 1:numel(required)
        assert(contains(source, required{idx}), ...
            '%s generator is missing expected switch propagation: %s', ...
            name, required{idx});
    end
end


function remove_temporary_root(path)
    if exist(path, 'dir')
        rmdir(path, 's');
    end
end
