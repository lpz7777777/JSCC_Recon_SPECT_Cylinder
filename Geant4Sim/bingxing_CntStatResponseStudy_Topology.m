function summary = bingxing_CntStatResponseStudy_Topology(worker_root, worker_ids, output_dir, merge_list)
%BINGXING_CNTSTATRESPONSESTUDY_TOPOLOGY Merge parallel response-study outputs.
%
% summary = bingxing_CntStatResponseStudy_Topology()
% summary = bingxing_CntStatResponseStudy_Topology(worker_root, worker_ids)
% summary = bingxing_CntStatResponseStudy_Topology(..., output_dir, merge_list)
%
% Each worker must have its own numeric directory below worker_root, for
% example worker_root/1, worker_root/2, ... . All rows in an individual CSV
% are summed, so this also handles a worker that performed several beamOn
% commands. List.csv is concatenated as text without loading it into RAM.

if nargin < 1 || isempty(worker_root)
    worker_root = pwd;
end
if nargin < 2 || isempty(worker_ids)
    worker_ids = 1:100;
end
if nargin < 3 || isempty(output_dir)
    output_dir = fullfile(worker_root, 'merged_CntStatResponseStudy');
end
if nargin < 4 || isempty(merge_list)
    merge_list = true;
end

worker_root = char(worker_root);
output_dir = char(output_dir);
worker_ids = reshape(worker_ids, 1, []);
validateattributes(worker_ids, {'numeric'}, ...
    {'real', 'finite', 'integer', 'positive', 'vector'}, mfilename, 'worker_ids');
validateattributes(merge_list, {'logical', 'numeric'}, {'scalar'}, mfilename, 'merge_list');
merge_list = logical(merge_list);

expected_bins = 10496;
count_files = {
    'CntStat_440.csv'
    'CntStat_218.csv'
    'CntStat218_From218.csv'
    'CntStat218_From440.csv'
    'CntStat440_From440.csv'
    'CntStat218_From440_FirstCrystal.csv'
    'CntStat218_From440_OtherCrystal.csv'
    'CntStat218_From440_Hit1.csv'
    'CntStat218_From440_Hit2.csv'
    'CntStat218_From440_Hit3Plus.csv'
    'CntStat218_From440_FirstCrystal_Compton0.csv'
    'CntStat218_From440_FirstCrystal_Compton1.csv'
    'CntStat218_From440_FirstCrystal_Compton2Plus.csv'
    };
primary_files = {
    'PrimaryCount218.csv'
    'PrimaryCount440.csv'
    'PrimaryCountOther.csv'
    };
required_files = [count_files; primary_files];

count_totals = cell(size(count_files));
for file_index = 1:numel(count_files)
    count_totals{file_index} = zeros(1, expected_bins);
end
primary_totals = zeros(size(primary_files));
included_workers = [];
rows_per_worker = zeros(0, numel(required_files));
missing_directories = [];

for worker_id = worker_ids
    worker_dir = fullfile(worker_root, sprintf('%d', worker_id));
    if ~isfolder(worker_dir)
        missing_directories(end + 1) = worker_id; %#ok<AGROW>
        continue;
    end

    missing_files = required_files(~cellfun(@(name) ...
        isfile(fullfile(worker_dir, name)), required_files));
    if merge_list && ~isfile(fullfile(worker_dir, 'List.csv'))
        missing_files{end + 1, 1} = 'List.csv'; %#ok<AGROW>
    end
    if ~isempty(missing_files)
        error('ResponseStudyMerge:IncompleteWorker', ...
            'Worker %d is incomplete. Missing: %s', ...
            worker_id, strjoin(missing_files, ', '));
    end

    worker_counts = cell(size(count_files));
    worker_rows = zeros(1, numel(required_files));
    for file_index = 1:numel(count_files)
        source_path = fullfile(worker_dir, count_files{file_index});
        [worker_counts{file_index}, worker_rows(file_index)] = ...
            read_and_sum_count_rows(source_path, expected_bins);
    end

    worker_primaries = zeros(size(primary_files));
    for file_index = 1:numel(primary_files)
        source_path = fullfile(worker_dir, primary_files{file_index});
        [worker_primaries(file_index), worker_rows(numel(count_files) + file_index)] = ...
            read_and_sum_scalar_file(source_path);
    end

    assert_response_closure(worker_counts, count_files, ...
        sprintf('worker %d', worker_id));
    if worker_primaries(3) ~= 0
        error('ResponseStudyMerge:UnexpectedPrimary', ...
            'Worker %d has %g unclassified primary events.', ...
            worker_id, worker_primaries(3));
    end

    for file_index = 1:numel(count_files)
        count_totals{file_index} = count_totals{file_index} + worker_counts{file_index};
    end
    primary_totals = primary_totals + worker_primaries;
    included_workers(end + 1) = worker_id; %#ok<AGROW>
    rows_per_worker(end + 1, :) = worker_rows; %#ok<AGROW>
end

if isempty(included_workers)
    error('ResponseStudyMerge:NoWorkers', ...
        'No complete worker directories were found under %s.', worker_root);
end

assert_response_closure(count_totals, count_files, 'merged result');
if ~isfolder(output_dir)
    mkdir(output_dir);
end

for file_index = 1:numel(count_files)
    write_integer_csv_row(fullfile(output_dir, count_files{file_index}), ...
        count_totals{file_index});
end
for file_index = 1:numel(primary_files)
    write_integer_csv_row(fullfile(output_dir, primary_files{file_index}), ...
        primary_totals(file_index));
end

if merge_list
    list_paths = arrayfun(@(worker_id) fullfile(worker_root, ...
        sprintf('%d', worker_id), 'List.csv'), included_workers, ...
        'UniformOutput', false);
    merge_text_files(list_paths, fullfile(output_dir, 'List.csv'));
end

idx = @(name) find(strcmp(count_files, name), 1);
closure = struct();
closure.cnt218_max_abs_bin = max(abs( ...
    count_totals{idx('CntStat_218.csv')} ...
    - count_totals{idx('CntStat218_From218.csv')} ...
    - count_totals{idx('CntStat218_From440.csv')}));
closure.first_other_max_abs_bin = max(abs( ...
    count_totals{idx('CntStat218_From440.csv')} ...
    - count_totals{idx('CntStat218_From440_FirstCrystal.csv')} ...
    - count_totals{idx('CntStat218_From440_OtherCrystal.csv')}));
closure.hit_multiplicity_max_abs_bin = max(abs( ...
    count_totals{idx('CntStat218_From440.csv')} ...
    - count_totals{idx('CntStat218_From440_Hit1.csv')} ...
    - count_totals{idx('CntStat218_From440_Hit2.csv')} ...
    - count_totals{idx('CntStat218_From440_Hit3Plus.csv')}));
closure.first_compton_max_abs_bin = max(abs( ...
    count_totals{idx('CntStat218_From440_FirstCrystal.csv')} ...
    - count_totals{idx('CntStat218_From440_FirstCrystal_Compton0.csv')} ...
    - count_totals{idx('CntStat218_From440_FirstCrystal_Compton1.csv')} ...
    - count_totals{idx('CntStat218_From440_FirstCrystal_Compton2Plus.csv')}));
closure.cnt440_minus_from440_total = sum( ...
    count_totals{idx('CntStat_440.csv')} ...
    - count_totals{idx('CntStat440_From440.csv')});

summary = struct();
summary.worker_root = worker_root;
summary.output_dir = output_dir;
summary.requested_worker_ids = worker_ids;
summary.included_worker_ids = included_workers;
summary.missing_worker_directories = missing_directories;
summary.rows_per_worker = rows_per_worker;
summary.count_files = count_files;
summary.primary_files = primary_files;
summary.primary_count_218 = primary_totals(1);
summary.primary_count_440 = primary_totals(2);
summary.primary_count_other = primary_totals(3);
summary.total_primary_events = sum(primary_totals);
summary.total_cntstat_218_bins = sum(count_totals{idx('CntStat_218.csv')});
summary.total_cntstat_440_bins = sum(count_totals{idx('CntStat_440.csv')});
summary.total_440_to_218_bins = sum(count_totals{idx('CntStat218_From440.csv')});
summary.closure = closure;
summary.list_merged = merge_list;
summary.created = char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));

save(fullfile(output_dir, 'MergeSummary.mat'), 'summary');
write_summary_text(fullfile(output_dir, 'MergeSummary.txt'), summary);

fprintf('Merged %d workers into %s\n', numel(included_workers), output_dir);
fprintf('Primary counts: 218=%g, 440=%g, other=%g, total=%g\n', ...
    summary.primary_count_218, summary.primary_count_440, ...
    summary.primary_count_other, summary.total_primary_events);
fprintf('440-to-218 accepted detector-bin counts: %g\n', ...
    summary.total_440_to_218_bins);
fprintf('All exact closure checks passed.\n');
end


function [total, row_count] = read_and_sum_count_rows(path, expected_bins)
values = readmatrix(path, 'OutputType', 'double');
if isempty(values)
    error('ResponseStudyMerge:EmptyFile', 'Count file is empty: %s', path);
end
if isvector(values) && numel(values) == expected_bins
    values = reshape(values, 1, expected_bins);
end
if size(values, 2) ~= expected_bins
    error('ResponseStudyMerge:DetectorCount', ...
        '%s has %d columns; expected %d.', path, size(values, 2), expected_bins);
end
validate_count_values(values, path);
row_count = size(values, 1);
total = sum(values, 1);
end


function [total, row_count] = read_and_sum_scalar_file(path)
values = readmatrix(path, 'OutputType', 'double');
values = values(:);
if isempty(values)
    error('ResponseStudyMerge:EmptyFile', 'Primary-count file is empty: %s', path);
end
validate_count_values(values, path);
row_count = numel(values);
total = sum(values);
end


function validate_count_values(values, path)
if any(~isfinite(values), 'all') || any(values < 0, 'all') ...
        || any(abs(values - round(values)) > 0, 'all')
    error('ResponseStudyMerge:InvalidCounts', ...
        '%s contains non-finite, negative, or non-integer values.', path);
end
end


function assert_response_closure(counts, names, label)
idx = @(name) find(strcmp(names, name), 1);
checks = {
    counts{idx('CntStat_218.csv')} ...
        - counts{idx('CntStat218_From218.csv')} ...
        - counts{idx('CntStat218_From440.csv')}, ...
        'CntStat_218 = From218 + From440'
    counts{idx('CntStat218_From440.csv')} ...
        - counts{idx('CntStat218_From440_FirstCrystal.csv')} ...
        - counts{idx('CntStat218_From440_OtherCrystal.csv')}, ...
        'From440 = FirstCrystal + OtherCrystal'
    counts{idx('CntStat218_From440.csv')} ...
        - counts{idx('CntStat218_From440_Hit1.csv')} ...
        - counts{idx('CntStat218_From440_Hit2.csv')} ...
        - counts{idx('CntStat218_From440_Hit3Plus.csv')}, ...
        'From440 = Hit1 + Hit2 + Hit3Plus'
    counts{idx('CntStat218_From440_FirstCrystal.csv')} ...
        - counts{idx('CntStat218_From440_FirstCrystal_Compton0.csv')} ...
        - counts{idx('CntStat218_From440_FirstCrystal_Compton1.csv')} ...
        - counts{idx('CntStat218_From440_FirstCrystal_Compton2Plus.csv')}, ...
        'FirstCrystal = Compton0 + Compton1 + Compton2Plus'
    };
for check_index = 1:size(checks, 1)
    maximum_error = max(abs(checks{check_index, 1}));
    if maximum_error ~= 0
        error('ResponseStudyMerge:ClosureFailure', ...
            '%s failed closure "%s"; max absolute bin error = %g.', ...
            label, checks{check_index, 2}, maximum_error);
    end
end
end


function write_integer_csv_row(path, values)
temporary_path = [path, '.tmp'];
file_id = fopen(temporary_path, 'w');
if file_id < 0
    error('ResponseStudyMerge:OutputOpen', 'Cannot open %s.', temporary_path);
end
cleanup = onCleanup(@() fclose_if_open(file_id));
fprintf(file_id, '%.0f', values(1));
fprintf(file_id, ',%.0f', values(2:end));
fprintf(file_id, '\n');
fclose(file_id);
clear cleanup;
movefile(temporary_path, path, 'f');
end


function merge_text_files(source_paths, destination_path)
temporary_path = [destination_path, '.tmp'];
output_id = fopen(temporary_path, 'wb');
if output_id < 0
    error('ResponseStudyMerge:OutputOpen', 'Cannot open %s.', temporary_path);
end
cleanup = onCleanup(@() fclose_if_open(output_id));
previous_ended_with_newline = true;
buffer_bytes = 8 * 1024 * 1024;

for path_index = 1:numel(source_paths)
    input_id = fopen(source_paths{path_index}, 'rb');
    if input_id < 0
        error('ResponseStudyMerge:InputOpen', 'Cannot open %s.', source_paths{path_index});
    end
    input_cleanup = onCleanup(@() fclose_if_open(input_id));
    first_chunk = true;
    while true
        chunk = fread(input_id, buffer_bytes, '*uint8');
        if isempty(chunk)
            break;
        end
        if first_chunk && ~previous_ended_with_newline
            fwrite(output_id, uint8(newline), 'uint8');
        end
        fwrite(output_id, chunk, 'uint8');
        previous_ended_with_newline = chunk(end) == 10 || chunk(end) == 13;
        first_chunk = false;
    end
    fclose(input_id);
    clear input_cleanup;
end

fclose(output_id);
clear cleanup;
movefile(temporary_path, destination_path, 'f');
end


function write_summary_text(path, summary)
file_id = fopen(path, 'w');
if file_id < 0
    error('ResponseStudyMerge:OutputOpen', 'Cannot open %s.', path);
end
cleanup = onCleanup(@() fclose_if_open(file_id));
fprintf(file_id, 'Created: %s\n', summary.created);
fprintf(file_id, 'Worker root: %s\n', summary.worker_root);
fprintf(file_id, 'Output directory: %s\n', summary.output_dir);
fprintf(file_id, 'Included workers (%d): %s\n', ...
    numel(summary.included_worker_ids), mat2str(summary.included_worker_ids));
fprintf(file_id, 'Missing worker directories: %s\n', ...
    mat2str(summary.missing_worker_directories));
fprintf(file_id, 'PrimaryCount218: %.0f\n', summary.primary_count_218);
fprintf(file_id, 'PrimaryCount440: %.0f\n', summary.primary_count_440);
fprintf(file_id, 'PrimaryCountOther: %.0f\n', summary.primary_count_other);
fprintf(file_id, 'Total primary events: %.0f\n', summary.total_primary_events);
fprintf(file_id, 'Total CntStat 218 bin counts: %.0f\n', summary.total_cntstat_218_bins);
fprintf(file_id, 'Total CntStat 440 bin counts: %.0f\n', summary.total_cntstat_440_bins);
fprintf(file_id, 'Total 440-to-218 bin counts: %.0f\n', summary.total_440_to_218_bins);
fprintf(file_id, 'Closure Cnt218: %.0f\n', summary.closure.cnt218_max_abs_bin);
fprintf(file_id, 'Closure first/other: %.0f\n', summary.closure.first_other_max_abs_bin);
fprintf(file_id, 'Closure hit multiplicity: %.0f\n', ...
    summary.closure.hit_multiplicity_max_abs_bin);
fprintf(file_id, 'Closure first-crystal Compton: %.0f\n', ...
    summary.closure.first_compton_max_abs_bin);
fprintf(file_id, 'Cnt440 minus From440 total: %.0f\n', ...
    summary.closure.cnt440_minus_from440_total);
fprintf(file_id, 'List merged: %d\n', summary.list_merged);
fclose(file_id);
clear cleanup;
end


function fclose_if_open(file_id)
if file_id > 0
    try
        fclose(file_id);
    catch
    end
end
end
