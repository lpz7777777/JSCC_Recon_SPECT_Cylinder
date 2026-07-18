function summary = bingxing_CntStatResponseStudy_Rows(worker_root, worker_ids, output_dir, expected_rows)
%BINGXING_CNTSTATRESPONSESTUDY_ROWS Row-preserving response-study merge.
% Each worker runs the same multi-beamOn macro. Row j represents the same
% manifest configuration in every worker and is summed only across workers.
% List.csv is intentionally not merged because radial response analysis uses
% source-separated CntStat and primary-count files only.

if nargin < 1 || isempty(worker_root), worker_root = pwd; end
if nargin < 2 || isempty(worker_ids), worker_ids = 1:100; end
if nargin < 3 || isempty(output_dir)
    output_dir = fullfile(worker_root, 'merged_RadialPointResponse');
end
if nargin < 4 || isempty(expected_rows), expected_rows = 202; end

worker_root = char(worker_root);
output_dir = char(output_dir);
worker_ids = reshape(worker_ids, 1, []);
expected_bins = 10496;
count_files = {'CntStat_218.csv'; 'CntStat_440.csv'; ...
    'CntStat218_From218.csv'; 'CntStat218_From440.csv'; ...
    'CntStat440_From440.csv'};
primary_files = {'PrimaryCount218.csv'; 'PrimaryCount440.csv'; 'PrimaryCountOther.csv'};

merged_counts = cell(numel(count_files), 1);
for idx = 1:numel(count_files)
    merged_counts{idx} = zeros(expected_rows, expected_bins, 'uint64');
end
merged_primaries = zeros(expected_rows, numel(primary_files), 'uint64');

for worker_id = worker_ids
    worker_dir = fullfile(worker_root, sprintf('%d', worker_id));
    if ~isfolder(worker_dir)
        error('ResponseStudyRows:MissingWorker', 'Missing worker directory: %s', worker_dir);
    end

    worker_counts = cell(numel(count_files), 1);
    for idx = 1:numel(count_files)
        path = fullfile(worker_dir, count_files{idx});
        worker_counts{idx} = read_count_matrix(path, expected_rows, expected_bins);
        merged_counts{idx} = merged_counts{idx} + worker_counts{idx};
    end
    assert_cntstat_closure(worker_counts, count_files, sprintf('worker %d', worker_id));

    for idx = 1:numel(primary_files)
        path = fullfile(worker_dir, primary_files{idx});
        merged_primaries(:, idx) = merged_primaries(:, idx) + ...
            read_count_matrix(path, expected_rows, 1);
    end
end

assert_cntstat_closure(merged_counts, count_files, 'merged result');
if any(merged_primaries(:, 3) ~= 0)
    error('ResponseStudyRows:UnexpectedPrimary', 'PrimaryCountOther contains nonzero rows.');
end

if ~isfolder(output_dir), mkdir(output_dir); end
for idx = 1:numel(count_files)
    writematrix(merged_counts{idx}, fullfile(output_dir, count_files{idx}));
end
for idx = 1:numel(primary_files)
    writematrix(merged_primaries(:, idx), fullfile(output_dir, primary_files{idx}));
end

summary = struct();
summary.worker_root = worker_root;
summary.output_dir = output_dir;
summary.worker_ids = worker_ids;
summary.worker_count = numel(worker_ids);
summary.expected_rows = expected_rows;
summary.expected_bins = expected_bins;
summary.primary_count_218 = merged_primaries(:, 1);
summary.primary_count_440 = merged_primaries(:, 2);
summary.primary_count_other = merged_primaries(:, 3);
summary.created = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss'));
save(fullfile(output_dir, 'MergeSummaryRows.mat'), 'summary');
fprintf('Merged %d workers while preserving %d manifest rows.\n', ...
    summary.worker_count, expected_rows);
end


function values = read_count_matrix(path, expected_rows, expected_columns)
    values = readmatrix(path, 'OutputType', 'double');
    if expected_columns == 1
        values = values(:);
    end
    if ~isequal(size(values), [expected_rows, expected_columns])
        error('ResponseStudyRows:Shape', '%s has size %s; expected [%d %d].', ...
            path, mat2str(size(values)), expected_rows, expected_columns);
    end
    if any(~isfinite(values), 'all') || any(values < 0, 'all') || ...
            any(values ~= round(values), 'all')
        error('ResponseStudyRows:InvalidValues', '%s has invalid count values.', path);
    end
    values = uint64(values);
end


function assert_cntstat_closure(counts, names, label)
    idx = @(name) find(strcmp(names, name), 1);
    closure = counts{idx('CntStat_218.csv')} - ...
        counts{idx('CntStat218_From218.csv')} - ...
        counts{idx('CntStat218_From440.csv')};
    if any(closure ~= 0, 'all')
        error('ResponseStudyRows:Closure', '%s failed CntStat_218 closure.', label);
    end
end
