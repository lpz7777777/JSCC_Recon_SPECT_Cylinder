function summaries = bingxing_UniformFovCntStat(run_root, output_root, worker_ids)
%BINGXING_UNIFORMFOVCNTSTAT Merge pure-218 and pure-440 uniform-FOV runs.

if nargin < 1 || isempty(run_root), run_root = pwd; end
if nargin < 2 || isempty(output_root)
    output_root = fullfile(run_root, 'merged');
end
if nargin < 3 || isempty(worker_ids), worker_ids = 1:100; end

energies = [218, 440];
summary_cells = cell(numel(energies), 1);
for idx = 1:numel(energies)
    energy_name = sprintf('%dkeV', energies(idx));
    summary_cells{idx} = merge_uniform_fov_cntstat( ...
        fullfile(run_root, energy_name), worker_ids, ...
        fullfile(output_root, energy_name));
end
summaries = vertcat(summary_cells{:});

save(fullfile(output_root, 'MergeSummaryBothEnergies.mat'), 'summaries');
fprintf('Merged uniform-FOV CntStat for 218 and 440 keV.\n');
end
