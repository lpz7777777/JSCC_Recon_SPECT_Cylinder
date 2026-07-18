function summary = merge_uniform_fov_cntstat(worker_root, worker_ids, output_dir)
%MERGE_UNIFORM_FOV_CNTSTAT Sum one-row CntStat-only worker outputs.

if nargin < 1 || isempty(worker_root), worker_root = pwd; end
if nargin < 2 || isempty(worker_ids), worker_ids = 1:100; end
if nargin < 3 || isempty(output_dir)
    output_dir = fullfile(worker_root, 'merged_UniformFovCntStat');
end

worker_root = char(worker_root);
output_dir = char(output_dir);
worker_ids = reshape(worker_ids, 1, []);
files = {'CntStat_218.csv', 'CntStat_440.csv'};
expected_bins = 10496;
totals = zeros(numel(files), expected_bins, 'uint64');
total_primary = uint64(0);

for worker_id = worker_ids
    worker_dir = fullfile(worker_root, sprintf('%d', worker_id));
    if ~isfolder(worker_dir)
        error('UniformFovMerge:MissingWorker', 'Missing worker directory: %s', worker_dir);
    end
    for file_idx = 1:numel(files)
        path = fullfile(worker_dir, files{file_idx});
        values = readmatrix(path, 'OutputType', 'double');
        values = values(:).';
        if numel(values) ~= expected_bins
            error('UniformFovMerge:Shape', '%s has %d bins; expected %d.', ...
                path, numel(values), expected_bins);
        end
        if any(~isfinite(values)) || any(values < 0) || any(values ~= round(values))
            error('UniformFovMerge:Counts', '%s contains invalid counts.', path);
        end
        totals(file_idx, :) = totals(file_idx, :) + uint64(values);
    end

    primary_path = fullfile(worker_dir, 'PrimaryCount.csv');
    primary = readmatrix(primary_path, 'OutputType', 'double');
    if ~isscalar(primary) || ~isfinite(primary) || primary <= 0 || ...
            primary ~= round(primary)
        error('UniformFovMerge:PrimaryCount', ...
            '%s must contain one positive integer.', primary_path);
    end
    total_primary = total_primary + uint64(primary);
end

if ~isfolder(output_dir), mkdir(output_dir); end
for file_idx = 1:numel(files)
    writematrix(totals(file_idx, :), fullfile(output_dir, files{file_idx}));
end
writematrix(total_primary, fullfile(output_dir, 'PrimaryCount.csv'));

summary = struct();
summary.worker_root = worker_root;
summary.output_dir = output_dir;
summary.worker_ids = worker_ids;
summary.worker_count = numel(worker_ids);
summary.detector_bins = expected_bins;
summary.total_primary = total_primary;
summary.total_cntstat_218 = sum(totals(1, :));
summary.total_cntstat_440 = sum(totals(2, :));
summary.created = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
save(fullfile(output_dir, 'MergeSummary.mat'), 'summary');
fprintf('Merged %d CntStat-only workers into %s\n', numel(worker_ids), output_dir);
fprintf('CntStat totals: 218=%g, 440=%g\n', ...
    double(summary.total_cntstat_218), double(summary.total_cntstat_440));
fprintf('Primary total: %g\n', double(summary.total_primary));
end
