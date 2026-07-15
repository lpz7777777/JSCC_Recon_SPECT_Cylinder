function generate_ehe_pb_nai_218_440_response_params()
% GENERATE_EHE_PB_NAI_218_440_RESPONSE_PARAMS
% Generate parameter sets for a conventional SPECT camera with:
%   - Siemens Symbia EHE-style parallel-hole collimator geometry
%   - Pb collimator material
%   - NaI scintillator crystal material
%
% Generated response cases:
%   1) EHE_PbNaI_218keV              : 218 keV source, automatic 218 photopeak window
%   2) EHE_PbNaI_440keV              : 440 keV source, automatic 440 photopeak window
%   3) EHE_PbNaI_440keV_to_218keVwin : 440 keV source, forced 218 keV energy window
%
% The third case is the cross-talk response A(218-window <- 440-source).
% Use its Scatter_SysMat output as the 440-to-218-window contribution.

    this_dir = fileparts(mfilename('fullpath'));
    old_dir = pwd;
    cleanup = onCleanup(@() cd(old_dir));
    cd(this_dir);

    cfg = config_geometry();
    cfg.geometry_type = 'ConventionalSPECT';
    cfg.detector_material = 'NaI';
    cfg.collimator_material = 'Pb';
    cfg.shield_material = 'Pb';
    cfg.fov.fov2collimator0 = cfg.conv.fov2collimator0;
    cfg.energy_list_keV = [218, 440];
    cfg.enable_compton = true;
    cfg.save_compton_only = true;
    cfg.compute_geo_relationship = true;

    output_root = fullfile(this_dir, cfg.output_root);
    runs_root = fullfile(this_dir, '..', 'runs');
    if ~exist(output_root, 'dir')
        mkdir(output_root);
    end
    if ~exist(runs_root, 'dir')
        mkdir(runs_root);
    end

    window_energy_keV = 218;
    window_res = calc_energy_resolution(cfg, window_energy_keV);
    window_lower_keV = (1 - window_res / 2.0) * window_energy_keV;
    window_upper_keV = (1 + window_res / 2.0) * window_energy_keV;

    cases = struct([]);
    cases(1).output_name = 'EHE_PbNaI_218keV';
    cases(1).source_energy_keV = 218;
    cases(1).force_window = false;
    cases(1).window_lower_keV = 0;
    cases(1).window_upper_keV = 0;
    cases(1).save_combined_sysmat = true;
    cases(1).enable_detector_recoil_escape = true;
    cases(1).enable_self_scatter_photopeak = true;
    cases(1).description = 'EHE Pb/NaI conventional SPECT, 218 keV source, automatic 218 photopeak window';

    cases(2).output_name = 'EHE_PbNaI_440keV';
    cases(2).source_energy_keV = 440;
    cases(2).force_window = false;
    cases(2).window_lower_keV = 0;
    cases(2).window_upper_keV = 0;
    cases(2).save_combined_sysmat = true;
    cases(2).enable_detector_recoil_escape = true;
    cases(2).enable_self_scatter_photopeak = true;
    cases(2).description = 'EHE Pb/NaI conventional SPECT, 440 keV source, automatic 440 photopeak window';

    cases(3).output_name = 'EHE_PbNaI_440keV_to_218keVwin';
    cases(3).source_energy_keV = 440;
    cases(3).force_window = true;
    cases(3).window_lower_keV = window_lower_keV;
    cases(3).window_upper_keV = window_upper_keV;
    cases(3).save_combined_sysmat = false;
    cases(3).enable_detector_recoil_escape = true;
    cases(3).enable_self_scatter_photopeak = false;
    cases(3).description = 'EHE Pb/NaI conventional SPECT, 440 keV source, forced 218 keV window; use Scatter_SysMat as cross-talk';

    fprintf('=====================================\n');
    fprintf('EHE Pb/NaI 218/440 response parameter generation\n');
    fprintf('Geometry:    %s\n', cfg.geometry_type);
    fprintf('Detector:    %s\n', cfg.detector_material);
    fprintf('Collimator:  %s\n', cfg.collimator_material);
    fprintf('Common front face from FOV center: %.6f mm\n', cfg.fov.common_front_face_y);
    fprintf('EHE local-Y origin from FOV center: %.6f mm\n', cfg.fov.fov2collimator0);
    fprintf('Output root: %s\n', output_root);
    fprintf('Runs root:   %s\n', runs_root);
    fprintf('218 keV forced window for cross-talk: [%.6f, %.6f] keV\n', ...
            window_lower_keV, window_upper_keV);
    fprintf('=====================================\n\n');

    for idx = 1:numel(cases)
        write_case(cfg, cases(idx), output_root, runs_root);
    end

    validate_ehe_parallel_hole_params(output_root);

    fprintf('\nDone. Regenerated EHE Pb/NaI Params_*.dat for 218, 440, and 440-to-218-window cases.\n');
end


function write_case(base_cfg, case_cfg, output_root, runs_root)
    cfg = base_cfg;
    source_energy_keV = case_cfg.source_energy_keV;

    cfg.use_same_energy_window = case_cfg.force_window;
    cfg.energy_window_lower_keV = case_cfg.window_lower_keV;
    cfg.energy_window_upper_keV = case_cfg.window_upper_keV;
    cfg.save_combined_sysmat = case_cfg.save_combined_sysmat;
    cfg.enable_detector_recoil_escape = case_cfg.enable_detector_recoil_escape;
    cfg.enable_self_scatter_photopeak = case_cfg.enable_self_scatter_photopeak;

    energy_res = calc_energy_resolution(cfg, source_energy_keV);
    det_coeff = material_db(cfg.detector_material, source_energy_keV);
    col_coeff = material_db(cfg.collimator_material, source_energy_keV);
    coeffs = struct('scintillator', det_coeff, 'highz', col_coeff);

    det_params = build_detector(cfg, coeffs, source_energy_keV, energy_res);
    col_params = build_collimator(cfg, col_coeff, source_energy_keV);

    outdir = fullfile(output_root, case_cfg.output_name);
    write_dat_files(outdir, cfg, source_energy_keV, det_params, col_params);
    write_case_note(outdir, cfg, case_cfg, energy_res, det_params, col_params);

    rundir = fullfile(runs_root, case_cfg.output_name);
    if ~exist(rundir, 'dir')
        mkdir(rundir);
    end
    copyfile(fullfile(outdir, 'Params_*.dat'), rundir);
    copyfile(fullfile(outdir, 'Params_README.txt'), rundir);

    fprintf('Generated %-32s source=%g keV  R=%.6f  combined=%d\n', ...
            case_cfg.output_name, source_energy_keV, energy_res, case_cfg.save_combined_sysmat);
    fprintf('  detector crystals: %.0f\n', det_params(1));
    fprintf('  collimator holes:  %.0f\n', col_params(11));
    if case_cfg.force_window
        fprintf('  forced window: [%.6f, %.6f] keV\n', ...
                case_cfg.window_lower_keV, case_cfg.window_upper_keV);
    else
        fprintf('  automatic photopeak window: [%.6f, %.6f] keV\n', ...
                (1 - energy_res / 2.0) * source_energy_keV, ...
                (1 + energy_res / 2.0) * source_energy_keV);
    end
    fprintf('  output: %s\n', outdir);
    fprintf('  runs:   %s\n\n', rundir);
end


function energy_res = calc_energy_resolution(cfg, energy_keV)
    energy_res = cfg.energy_resolution_ref * sqrt(cfg.energy_resolution_ref_keV / energy_keV);
end


function write_case_note(outdir, cfg, case_cfg, energy_res, det_params, col_params)
    note_path = fullfile(outdir, 'Params_README.txt');
    fid = fopen(note_path, 'w');
    if fid < 0
        error('generate_ehe_pb_nai_218_440_response_params: cannot write %s', note_path);
    end
    cleaner = onCleanup(@() fclose(fid));

    fprintf(fid, '%s\n', case_cfg.output_name);
    fprintf(fid, '%s\n\n', case_cfg.description);
    fprintf(fid, 'geometry_type = %s\n', cfg.geometry_type);
    fprintf(fid, 'collimator_geometry = Siemens Symbia EHE-style triangular-lattice parallel-hole\n');
    fprintf(fid, 'collimator_material = %s\n', cfg.collimator_material);
    fprintf(fid, 'collimator_hole_material = Air/Vacuum\n');
    fprintf(fid, 'detector_material = %s\n', cfg.detector_material);
    fprintf(fid, 'source_energy_keV = %.9g\n', case_cfg.source_energy_keV);
    fprintf(fid, 'relative_FWHM_at_source_energy = %.9g\n', energy_res);
    fprintf(fid, 'use_forced_energy_window = %d\n', case_cfg.force_window);
    fprintf(fid, 'energy_window_lower_keV = %.9g\n', case_cfg.window_lower_keV);
    fprintf(fid, 'energy_window_upper_keV = %.9g\n', case_cfg.window_upper_keV);
    fprintf(fid, 'save_combined_sysmat = %d\n', case_cfg.save_combined_sysmat);
    fprintf(fid, 'enable_detector_recoil_escape = %d\n', case_cfg.enable_detector_recoil_escape);
    fprintf(fid, 'enable_self_scatter_photopeak = %d\n', case_cfg.enable_self_scatter_photopeak);
    fprintf(fid, 'detector_crystal_count = %.0f\n', det_params(1));
    fprintf(fid, 'collimator_hole_count = %.0f\n', col_params(11));
    fprintf(fid, 'collimator_thickness_mm = %.9g\n', cfg.conv.collimator_thickness);
    fprintf(fid, 'hole_diameter_mm = %.9g\n', cfg.conv.collimator.hole_diameter);
    fprintf(fid, 'septal_thickness_mm = %.9g\n\n', cfg.conv.collimator.septal_thickness);
    fprintf(fid, 'shared_JSCC_detector_and_EHE_collimator_front_face_mm = %.9g\n', ...
            cfg.fov.common_front_face_y);
    fprintf(fid, 'cuda_fov_to_local_y_origin_mm = %.9g\n', cfg.fov.fov2collimator0);
    fprintf(fid, 'collimator_front_face_mm = %.9g\n', ...
            cfg.fov.fov2collimator0 - cfg.conv.collimator_thickness / 2);
    fprintf(fid, 'collimator_back_face_mm = %.9g\n\n', ...
            cfg.fov.fov2collimator0 + cfg.conv.collimator_thickness / 2);
    if case_cfg.force_window
        fprintf(fid, 'This is a cross-talk parameter set. Run ScatterGen with the 440 keV PE matrix and use Scatter_SysMat_*.sysmat as A(218-window <- 440-source).\n');
        fprintf(fid, 'Do not use SysMat_withScatter for this cross-talk term, because PEGen does not apply the forced low-energy window to the direct PE matrix.\n');
        fprintf(fid, 'The matrix is per emitted 440 keV photon and does not include 225Ac branching ratio.\n');
    else
        fprintf(fid, 'This is a standard photopeak-window parameter set. Use PE and Scatter/combined outputs according to the reconstruction model.\n');
        fprintf(fid, 'The matrix is per emitted photon at source_energy_keV and does not include 225Ac branching ratio.\n');
    end
end
