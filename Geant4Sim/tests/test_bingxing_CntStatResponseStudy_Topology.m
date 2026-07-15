function test_bingxing_CntStatResponseStudy_Topology
test_root = tempname;
mkdir(test_root);
cleanup = onCleanup(@() rmdir(test_root, 's'));

expected_bins = 10496;
for worker_id = 1:2
    worker_dir = fullfile(test_root, sprintf('%d', worker_id));
    mkdir(worker_dir);
    direct_218 = worker_id * ones(1, expected_bins);
    cross = (worker_id + 1) * ones(1, expected_bins);
    direct_440 = (worker_id + 2) * ones(1, expected_bins);
    first = ones(1, expected_bins);
    other = cross - first;
    hit_1 = first;
    hit_2 = other;
    zeros_row = zeros(1, expected_bins);

    write_rows(worker_dir, 'CntStat_440.csv', direct_440);
    write_rows(worker_dir, 'CntStat_218.csv', direct_218 + cross);
    write_rows(worker_dir, 'CntStat218_From218.csv', direct_218);
    write_rows(worker_dir, 'CntStat218_From440.csv', cross);
    write_rows(worker_dir, 'CntStat440_From440.csv', direct_440);
    write_rows(worker_dir, 'CntStat218_From440_FirstCrystal.csv', first);
    write_rows(worker_dir, 'CntStat218_From440_OtherCrystal.csv', other);
    write_rows(worker_dir, 'CntStat218_From440_Hit1.csv', hit_1);
    write_rows(worker_dir, 'CntStat218_From440_Hit2.csv', hit_2);
    write_rows(worker_dir, 'CntStat218_From440_Hit3Plus.csv', zeros_row);
    write_rows(worker_dir, 'CntStat218_From440_FirstCrystal_Compton0.csv', zeros_row);
    write_rows(worker_dir, 'CntStat218_From440_FirstCrystal_Compton1.csv', first);
    write_rows(worker_dir, 'CntStat218_From440_FirstCrystal_Compton2Plus.csv', zeros_row);
    writematrix([10 * worker_id; 20 * worker_id], ...
        fullfile(worker_dir, 'PrimaryCount218.csv'));
    writematrix([30 * worker_id; 40 * worker_id], ...
        fullfile(worker_dir, 'PrimaryCount440.csv'));
    writematrix([0; 0], fullfile(worker_dir, 'PrimaryCountOther.csv'));
    writematrix([worker_id, 0.2, worker_id + 1, 0.22, 1], ...
        fullfile(worker_dir, 'List.csv'));
end

output_dir = fullfile(test_root, 'merged');
summary = bingxing_CntStatResponseStudy_Topology( ...
    test_root, 1:3, output_dir, true);

assert(isequal(summary.included_worker_ids, [1, 2]));
assert(isequal(summary.missing_worker_directories, 3));
assert(summary.primary_count_218 == 90);
assert(summary.primary_count_440 == 210);
assert(summary.primary_count_other == 0);
assert(summary.closure.cnt218_max_abs_bin == 0);
assert(summary.closure.first_other_max_abs_bin == 0);
assert(summary.closure.hit_multiplicity_max_abs_bin == 0);
assert(summary.closure.first_compton_max_abs_bin == 0);
assert(all(readmatrix(fullfile(output_dir, 'CntStat218_From440.csv')) == 5));
assert(size(readmatrix(fullfile(output_dir, 'List.csv')), 1) == 2);
assert(isfile(fullfile(output_dir, 'MergeSummary.mat')));
assert(isfile(fullfile(output_dir, 'MergeSummary.txt')));

clear cleanup;
fprintf('test_bingxing_CntStatResponseStudy_Topology: PASS\n');
end


function write_rows(worker_dir, filename, row)
% Two appended rows exercise per-worker row summation.
writematrix([row; zeros(size(row))], fullfile(worker_dir, filename));
end
