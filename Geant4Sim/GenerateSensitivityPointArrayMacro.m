function manifest = GenerateSensitivityPointArrayMacro(varargin)
%GENERATESENSITIVITYPOINTARRAYMACRO Generate one equal-weight GPS point-array macro.
%
% With no arguments, this function reads:
%   Factors/440keV_RotateNum20/coor_polar_full.csv
%
% and writes one 440 keV macro containing all 25600 polar-grid positions:
%   Geant4Sim/Macro/SensitivityPointArray_440keV_RotateNum20/
%       SensitivityPointArray_440keV_RotateNum20_View01.mac
%
% The first point uses the default GPS source, whose intensity is 1. Every
% subsequent point starts with "/gps/source/add 1". GPS multiple-vertex mode
% is disabled, so each Geant4 event selects exactly one point source. The
% macro contains exactly one /run/beamOn command at the end.
%
% Example:
%   GenerateSensitivityPointArrayMacro
%
%   GenerateSensitivityPointArrayMacro( ...
%       'PrimaryEventsPerPoint', 40000, ...
%       'OutputFileName', 'SensitivityPointArray_440keV_1p024e9.mac', ...
%       'Overwrite', true);

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
default_factor_dir = fullfile(repo_root, 'Factors', '440keV_RotateNum20');
default_output_dir = fullfile( ...
    script_dir, 'Macro', 'SensitivityPointArray_440keV_RotateNum20');

parser = inputParser;
parser.FunctionName = mfilename;
addParameter(parser, 'FactorDir', default_factor_dir, @isTextScalar);
addParameter(parser, 'CoordinateCsv', '', @isTextScalar);
addParameter(parser, 'OutputDir', default_output_dir, @isTextScalar);
addParameter(parser, 'OutputFileName', ...
    'SensitivityPointArray_440keV_RotateNum20_View01.mac', @isTextScalar);
addParameter(parser, 'EnergyKeV', 440, @isPositiveScalar);
addParameter(parser, 'RotateNum', 20, @isPositiveIntegerScalar);
addParameter(parser, 'RotationIndex', 1, @isPositiveIntegerScalar);
addParameter(parser, 'FovCenterMm', [0, -245, 0], @isThreeElementFiniteVector);
addParameter(parser, 'PrimaryEventsPerPoint', 20000, @isPositiveIntegerScalar);
addParameter(parser, 'ExpectedPointCount', 25600, @isNonnegativeIntegerScalar);
addParameter(parser, 'Overwrite', false, @isLogicalScalar);
parse(parser, varargin{:});
cfg = parser.Results;

factor_dir = char(cfg.FactorDir);
coordinate_csv = char(cfg.CoordinateCsv);
if isempty(coordinate_csv)
    coordinate_csv = fullfile(factor_dir, 'coor_polar_full.csv');
end
output_dir = char(cfg.OutputDir);
output_file_name = char(cfg.OutputFileName);
fov_center_mm = reshape(double(cfg.FovCenterMm), 1, 3);

if cfg.RotationIndex > cfg.RotateNum
    error('RotationIndex (%d) cannot exceed RotateNum (%d).', ...
        cfg.RotationIndex, cfg.RotateNum);
end
if ~isfile(coordinate_csv)
    error('Coordinate CSV does not exist: %s', coordinate_csv);
end

coordinates_local_mm = readmatrix(coordinate_csv, 'FileType', 'text');
if size(coordinates_local_mm, 2) < 3
    error('Coordinate CSV must contain at least three columns: %s', coordinate_csv);
end
coordinates_local_mm = double(coordinates_local_mm(:, 1:3));
if isempty(coordinates_local_mm) || any(~isfinite(coordinates_local_mm), 'all')
    error('Coordinate CSV is empty or contains non-finite values: %s', coordinate_csv);
end

point_count = size(coordinates_local_mm, 1);
if cfg.ExpectedPointCount > 0 && point_count ~= cfg.ExpectedPointCount
    error('Expected %d source positions, but found %d in %s.', ...
        cfg.ExpectedPointCount, point_count, coordinate_csv);
end
unique_point_count = size(unique(coordinates_local_mm, 'rows'), 1);
if unique_point_count ~= point_count
    error('Coordinate CSV contains %d duplicate rows.', point_count - unique_point_count);
end

beam_on = double(point_count) * double(cfg.PrimaryEventsPerPoint);
geant4_beam_on_max = double(intmax('int32'));
if beam_on > geant4_beam_on_max
    error(['Requested /run/beamOn %.0f exceeds the standard Geant4 signed ', ...
        '32-bit command limit %.0f. Reduce PrimaryEventsPerPoint to %d or less.'], ...
        beam_on, geant4_beam_on_max, floor(geant4_beam_on_max / point_count));
end

phi_rad = (double(cfg.RotationIndex) - 1) * 2 * pi / double(cfg.RotateNum);
cos_phi = cos(phi_rad);
sin_phi = sin(phi_rad);
x_local = coordinates_local_mm(:, 1);
y_local = coordinates_local_mm(:, 2);
coordinates_global_mm = zeros(size(coordinates_local_mm));
% Match the existing phantom convention: rotate local x-y by -phi, then
% translate the local FOV origin to the Geant4 center [0, -245, 0] mm.
coordinates_global_mm(:, 1) = x_local * cos_phi + y_local * sin_phi + fov_center_mm(1);
coordinates_global_mm(:, 2) = y_local * cos_phi - x_local * sin_phi + fov_center_mm(2);
coordinates_global_mm(:, 3) = coordinates_local_mm(:, 3) + fov_center_mm(3);

if ~isfolder(output_dir)
    mkdir(output_dir);
end
macro_path = fullfile(output_dir, output_file_name);
manifest_path = fullfile(output_dir, 'generation_manifest.json');
if isfile(macro_path) && ~cfg.Overwrite
    error('Output macro already exists. Use ''Overwrite'', true to replace it: %s', macro_path);
end

temporary_macro_path = [macro_path, '.tmp'];
fid = fopen(temporary_macro_path, 'w');
if fid < 0
    error('Cannot open output macro for writing: %s', temporary_macro_path);
end
try
    fprintf(fid, '# Equal-weight polar-grid point array for Compton Sensi_d\n');
    fprintf(fid, '# Energy: %.10g keV\n', cfg.EnergyKeV);
    fprintf(fid, '# Coordinate source: %s\n', normalizePathForComment(coordinate_csv));
    fprintf(fid, '# Point count: %d\n', point_count);
    fprintf(fid, '# Every GPS source has intensity 1\n');
    fprintf(fid, '# Source 1 uses the GPS default intensity 1; sources 2..%d use /gps/source/add 1\n', point_count);
    fprintf(fid, '# GPS multiplevertex=false: exactly one source/primary vertex is selected per event\n');
    fprintf(fid, '# Source selection is uniform in expectation; realized per-point counts are multinomial\n');
    fprintf(fid, '# Local-to-Geant4 FOV center: [%.10g, %.10g, %.10g] mm\n', fov_center_mm);
    fprintf(fid, '# Rotation: %d/%d, phi=%.10g deg, using the existing -phi convention\n', ...
        cfg.RotationIndex, cfg.RotateNum, phi_rad * 180 / pi);
    fprintf(fid, '# Expected primary events per point: %d\n', cfg.PrimaryEventsPerPoint);
    fprintf(fid, '# Total primary events: %.0f\n', beam_on);
    fprintf(fid, '# The macro changes only the GPS source; detector/world materials remain those in Geant4Code\n\n');
    fprintf(fid, '/gps/source/multiplevertex false\n\n');

    for point_index = 1:point_count
        if point_index > 1
            fprintf(fid, '/gps/source/add 1\n');
        end
        fprintf(fid, '# source %d/%d, local=(%.10f, %.10f, %.10f) mm\n', ...
            point_index, point_count, coordinates_local_mm(point_index, :));
        fprintf(fid, '/gps/particle gamma\n');
        fprintf(fid, '/gps/number 1\n');
        fprintf(fid, '/gps/energy %.10g keV\n', cfg.EnergyKeV);
        fprintf(fid, '/gps/pos/type Point\n');
        fprintf(fid, '/gps/pos/centre %.10f %.10f %.10f mm\n', ...
            coordinates_global_mm(point_index, :));
        fprintf(fid, '/gps/ang/type iso\n');
        fprintf(fid, '/gps/ang/mintheta 0 deg\n');
        fprintf(fid, '/gps/ang/maxtheta 180 deg\n\n');
    end

    fprintf(fid, '# Exactly one run command; all point sources participate in this run.\n');
    fprintf(fid, '/run/beamOn %.0f\n', beam_on);
catch exception
    fclose(fid);
    rethrow(exception);
end
if fclose(fid) ~= 0
    error('Failed to close completed macro: %s', temporary_macro_path);
end

[move_ok, move_message] = movefile(temporary_macro_path, macro_path, 'f');
if ~move_ok
    error('Failed to move completed macro into place: %s', move_message);
end

manifest = struct;
manifest.generator = mfilename;
manifest.generated_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss Z'));
manifest.coordinate_csv = coordinate_csv;
manifest.factor_dir = factor_dir;
manifest.macro_path = macro_path;
manifest.energy_keV = double(cfg.EnergyKeV);
manifest.point_count = point_count;
manifest.source_weight = 1;
manifest.multiple_vertex = false;
manifest.primary_events_per_point_expected = double(cfg.PrimaryEventsPerPoint);
manifest.beam_on = beam_on;
manifest.source_photons_for_sensitivity = beam_on;
manifest.rotate_num = double(cfg.RotateNum);
manifest.rotation_index = double(cfg.RotationIndex);
manifest.rotation_phi_deg = phi_rad * 180 / pi;
manifest.fov_center_mm = fov_center_mm;
manifest.local_coordinate_min_mm = min(coordinates_local_mm, [], 1);
manifest.local_coordinate_max_mm = max(coordinates_local_mm, [], 1);
manifest.global_coordinate_min_mm = min(coordinates_global_mm, [], 1);
manifest.global_coordinate_max_mm = max(coordinates_global_mm, [], 1);
manifest.unique_z_count = numel(unique(coordinates_local_mm(:, 3)));
manifest.position_type = 'Point';
manifest.angular_distribution = 'isotropic';
manifest.material_geometry = 'unchanged; controlled only by Geant4Code/DetectorConstruction.cc';
manifest.source_sampling_note = [ ...
    'All GPS intensities are equal. Each event selects one source uniformly in expectation; ', ...
    'the realized number selected from each point is not forced to be exactly equal.'];

manifest_text = jsonencode(manifest, 'PrettyPrint', true);
manifest_fid = fopen(manifest_path, 'w');
if manifest_fid < 0
    error('Cannot write generation manifest: %s', manifest_path);
end
try
    fprintf(manifest_fid, '%s\n', manifest_text);
catch exception
    fclose(manifest_fid);
    rethrow(exception);
end
if fclose(manifest_fid) ~= 0
    error('Failed to close generation manifest: %s', manifest_path);
end

macro_info = dir(macro_path);
fprintf('Generated equal-weight %.10g keV sensitivity point-array macro.\n', cfg.EnergyKeV);
fprintf('  Coordinates : %s\n', coordinate_csv);
fprintf('  Sources     : %d (all GPS intensity 1)\n', point_count);
fprintf('  beamOn      : %.0f\n', beam_on);
fprintf('  Mean/source : %d primary events (expected)\n', cfg.PrimaryEventsPerPoint);
fprintf('  Macro       : %s (%.2f MiB)\n', macro_path, macro_info.bytes / 2^20);
fprintf('  Manifest    : %s\n', manifest_path);
end


function result = isTextScalar(value)
result = ischar(value) || (isstring(value) && isscalar(value));
end


function result = isPositiveScalar(value)
result = isnumeric(value) && isscalar(value) && isfinite(value) && value > 0;
end


function result = isPositiveIntegerScalar(value)
result = isPositiveScalar(value) && value == floor(value);
end


function result = isNonnegativeIntegerScalar(value)
result = isnumeric(value) && isscalar(value) && isfinite(value) && ...
    value >= 0 && value == floor(value);
end


function result = isThreeElementFiniteVector(value)
result = isnumeric(value) && numel(value) == 3 && all(isfinite(value), 'all');
end


function result = isLogicalScalar(value)
result = (islogical(value) || isnumeric(value)) && isscalar(value) && ...
    ismember(value, [false, true]);
end


function path_text = normalizePathForComment(path_text)
path_text = strrep(path_text, '\', '/');
end
