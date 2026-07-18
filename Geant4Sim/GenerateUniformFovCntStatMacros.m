function manifests = GenerateUniformFovCntStatMacros(varargin)
%GENERATEUNIFORMFOVCNTSTATMACROS Generate 218/440 equal-weight full-FOV macros.
% Each Geant4 event selects exactly one of the 25620 CenterPoint polar-grid
% positions. Equal GPS source intensities make every position equally likely.

script_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(script_dir);
parser = inputParser;
addParameter(parser, 'PrimaryEventsPerPointPerWorker', 4000, ...
    @(x) isnumeric(x) && isscalar(x) && x > 0 && x == floor(x));
addParameter(parser, 'WorkerCount', 100, ...
    @(x) isnumeric(x) && isscalar(x) && x > 0 && x == floor(x));
addParameter(parser, 'Overwrite', true, ...
    @(x) (islogical(x) || isnumeric(x)) && isscalar(x));
parse(parser, varargin{:});
cfg = parser.Results;

output_root = fullfile(script_dir, 'Macro', 'UniformFovCntStat_CenterPoint');
energies = [218, 440];
manifest_cells = cell(numel(energies), 1);

for idx = 1:numel(energies)
    energy = energies(idx);
    factor_dir = fullfile(repo_root, 'Factors', ...
        sprintf('%dkeV_RotateNum20_CenterPoint', energy));
    output_dir = fullfile(output_root, sprintf('%dkeV', energy));
    output_name = sprintf('UniformFovCntStat_%dkeV_CenterPoint_worker.mac', energy);
    manifest = GenerateSensitivityPointArrayMacro( ...
        'FactorDir', factor_dir, ...
        'OutputDir', output_dir, ...
        'OutputFileName', output_name, ...
        'EnergyKeV', energy, ...
        'PrimaryEventsPerPoint', cfg.PrimaryEventsPerPointPerWorker, ...
        'ExpectedPointCount', 25620, ...
        'Overwrite', logical(cfg.Overwrite));

    manifest.worker_count = double(cfg.WorkerCount);
    manifest.target_primary_events = ...
        manifest.beam_on * double(cfg.WorkerCount);
    manifest.target_primary_events_per_point_expected = ...
        double(cfg.PrimaryEventsPerPointPerWorker) * double(cfg.WorkerCount);
    manifest_cells{idx} = manifest;
    manifest_path = fullfile(output_dir, 'uniform_fov_manifest.json');
    fid = fopen(manifest_path, 'w');
    if fid < 0
        error('Cannot write uniform-FOV manifest: %s', manifest_path);
    end
    fprintf(fid, '%s\n', jsonencode(manifest, 'PrettyPrint', true));
    fclose(fid);
end
manifests = vertcat(manifest_cells{:});

fprintf('Generated two uniform-FOV worker macros under:\n  %s\n', output_root);
fprintf('Workers=%d, expected events/point/worker=%d, target events/point=%d\n', ...
    cfg.WorkerCount, cfg.PrimaryEventsPerPointPerWorker, ...
    cfg.WorkerCount * cfg.PrimaryEventsPerPointPerWorker);
end
