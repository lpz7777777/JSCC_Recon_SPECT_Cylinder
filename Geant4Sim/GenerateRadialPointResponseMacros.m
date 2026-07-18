%% GenerateRadialPointResponseMacros
% Generate pure-energy point-source macros for diagnosing radial JSCC
% system-matrix/Geant4 mismatch near the FOV center.
%
% The macros target Geant4Code_CntStatResponseStudy.  They preserve the
% production JSCC geometry and output source-separated CntStat files, while
% using only one GPS source per run.  Coordinates are in the same source plane
% convention as ContrastPhantom_DualEnergy_Rotate_3D.m:
%   Factor (x, y, z) -> Geant4 (x, y - 245, z) mm.
%
% Output layout:
%   Geant4Sim/Macro/RadialPointResponse_JSCC/<label>.mac
%   Geant4Sim/Macro/RadialPointResponse_JSCC/radial_point_response_full_fov_worker.mac
%   Geant4Sim/Macro/RadialPointResponse_JSCC/radial_point_manifest.csv
%
% Every worker executes the same generated radial_point_response_full_fov_worker.mac.
% The CSV writers append one row per beamOn, in manifest order. Merge matching
% rows from cfg.worker_count worker directories after the Slurm array finishes.

clear;

cfg = struct();
cfg.fov_center_y_mm = -245;
% Match every radial ring in the polar Factors, from the center to the FOV edge.
cfg.radii_mm = 0:6:150;
cfg.angles_deg = [0, 90, 180, 270];
cfg.energies_keV = [218, 440];
cfg.worker_count = 100;
cfg.events_per_worker_configuration = 1e6;
cfg.source_z_mm = 0;
cfg.batch_macro_name = "radial_point_response_full_fov_worker.mac";

if cfg.worker_count <= 0 || cfg.worker_count ~= floor(cfg.worker_count)
    error("cfg.worker_count must be a positive integer.");
end
if cfg.events_per_worker_configuration <= 0 || ...
        cfg.events_per_worker_configuration ~= floor(cfg.events_per_worker_configuration)
    error("cfg.events_per_worker_configuration must be a positive integer.");
end
target_events_per_configuration = cfg.worker_count * cfg.events_per_worker_configuration;

script_dir = fileparts(mfilename("fullpath"));
output_dir = fullfile(script_dir, "Macro", "RadialPointResponse_JSCC");
if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

old_macros = dir(fullfile(output_dir, "radial_*.mac"));
for file_idx = 1:numel(old_macros)
    delete(fullfile(old_macros(file_idx).folder, old_macros(file_idx).name));
end

row_template = struct("label", "", "energy_keV", 0, "radius_mm", 0, ...
    "angle_deg", 0, "factor_x_mm", 0, "factor_y_mm", 0, ...
    "factor_z_mm", 0, "geant4_x_mm", 0, "geant4_y_mm", 0, ...
    "geant4_z_mm", 0, "primary_events_per_worker", 0, ...
    "worker_count", 0, "target_primary_events", 0);
angle_counts = repmat(numel(cfg.angles_deg), size(cfg.radii_mm));
angle_counts(cfg.radii_mm == 0) = 1;
row_count = sum(angle_counts) * numel(cfg.energies_keV);
rows = repmat(row_template, 1, row_count);
row_idx = 0;

for radius_mm = cfg.radii_mm
    if radius_mm == 0
        angles_deg = 0;
    else
        angles_deg = cfg.angles_deg;
    end

    for angle_deg = angles_deg
        factor_x_mm = radius_mm * cosd(angle_deg);
        factor_y_mm = radius_mm * sind(angle_deg);
        geant4_x_mm = factor_x_mm;
        geant4_y_mm = cfg.fov_center_y_mm + factor_y_mm;

        for energy_keV = cfg.energies_keV
            label = sprintf("radial_r%03d_t%03d_E%03d", ...
                radius_mm, angle_deg, energy_keV);
            macro_path = fullfile(output_dir, label + ".mac");
            write_macro(macro_path, label, energy_keV, radius_mm, angle_deg, ...
                factor_x_mm, factor_y_mm, cfg.source_z_mm, geant4_x_mm, ...
                geant4_y_mm, cfg.source_z_mm, cfg.events_per_worker_configuration);

            row_idx = row_idx + 1;
            rows(row_idx) = struct( ...
                "label", label, ...
                "energy_keV", energy_keV, ...
                "radius_mm", radius_mm, ...
                "angle_deg", angle_deg, ...
                "factor_x_mm", factor_x_mm, ...
                "factor_y_mm", factor_y_mm, ...
                "factor_z_mm", cfg.source_z_mm, ...
                "geant4_x_mm", geant4_x_mm, ...
                "geant4_y_mm", geant4_y_mm, ...
                "geant4_z_mm", cfg.source_z_mm, ...
                "primary_events_per_worker", cfg.events_per_worker_configuration, ...
                "worker_count", cfg.worker_count, ...
                "target_primary_events", target_events_per_configuration);
        end
    end
end

manifest = struct2table(rows);
writetable(manifest, fullfile(output_dir, "radial_point_manifest.csv"));
write_batch_macro(fullfile(output_dir, cfg.batch_macro_name), rows);

fprintf("Generated %d pure-energy radial point-response macros in:\n  %s\n", ...
    height(manifest), output_dir);
fprintf("Per worker configuration: %.0f primary photons.\n", ...
    cfg.events_per_worker_configuration);
fprintf("Workers: %d; target per configuration: %.0f; total: %.0f primaries.\n", ...
    cfg.worker_count, target_events_per_configuration, ...
    height(manifest) * target_events_per_configuration);
fprintf("Recommended worker macro: %s\n", ...
    fullfile(output_dir, cfg.batch_macro_name));


function write_macro(macro_path, label, energy_keV, radius_mm, angle_deg, ...
        factor_x_mm, factor_y_mm, factor_z_mm, geant4_x_mm, geant4_y_mm, ...
        geant4_z_mm, primary_events)
    fid = fopen(char(macro_path), 'w');
    if fid < 0
        error("Cannot create macro: %s", macro_path);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, "# JSCC radial point-response study: %s\n", label);
    fprintf(fid, "# Factor coordinate (mm): (%.4f, %.4f, %.4f)\n", ...
        factor_x_mm, factor_y_mm, factor_z_mm);
    fprintf(fid, "# Geant4 GPS coordinate (mm): (%.4f, %.4f, %.4f)\n", ...
        geant4_x_mm, geant4_y_mm, geant4_z_mm);
    fprintf(fid, "# radius=%.1f mm, azimuth=%.1f deg, pure %d keV, beamOn=%d\n\n", ...
        radius_mm, angle_deg, energy_keV, primary_events);
    fprintf(fid, "/gps/particle gamma\n");
    fprintf(fid, "/gps/energy %d keV\n", energy_keV);
    fprintf(fid, "/gps/pos/type Point\n");
    fprintf(fid, "/gps/pos/centre %.4f %.4f %.4f mm\n", ...
        geant4_x_mm, geant4_y_mm, geant4_z_mm);
    fprintf(fid, "/gps/ang/type iso\n");
    fprintf(fid, "/gps/ang/mintheta 0 deg\n");
    fprintf(fid, "/gps/ang/maxtheta 180 deg\n\n");
    fprintf(fid, "/run/beamOn %d\n", primary_events);
end


function write_batch_macro(macro_path, rows)
    fid = fopen(char(macro_path), 'w');
    if fid < 0
        error("Cannot create batch macro: %s", macro_path);
    end
    cleanup = onCleanup(@() fclose(fid));

    fprintf(fid, "# JSCC radial point-response study: all configurations\n");
    fprintf(fid, "# One /run/beamOn per block. CSV row order follows radial_point_manifest.csv.\n");
    fprintf(fid, "# Run this macro once in one clean directory; do not append a second run.\n");
    fprintf(fid, "# Exactly one GPS source is reused and overwritten for every block.\n\n");

    for idx = 1:numel(rows)
        row = rows(idx);
        fprintf(fid, "# row %d: %s | factor=(%.4f, %.4f, %.4f) mm\n", ...
            idx, row.label, row.factor_x_mm, row.factor_y_mm, row.factor_z_mm);
        fprintf(fid, "/gps/particle gamma\n");
        fprintf(fid, "/gps/number 1\n");
        fprintf(fid, "/gps/energy %d keV\n", row.energy_keV);
        fprintf(fid, "/gps/pos/type Point\n");
        fprintf(fid, "/gps/pos/centre %.4f %.4f %.4f mm\n", ...
            row.geant4_x_mm, row.geant4_y_mm, row.geant4_z_mm);
        fprintf(fid, "/gps/ang/type iso\n");
        fprintf(fid, "/gps/ang/mintheta 0 deg\n");
        fprintf(fid, "/gps/ang/maxtheta 180 deg\n");
        fprintf(fid, "/run/beamOn %d\n\n", row.primary_events_per_worker);
    end
end
