function tests = test_export_run_sysmat_for_amide
%TEST_EXPORT_RUN_SYSMAT_FOR_AMIDE Regression tests for path handling/export.
    tests = functiontests(localfunctions);
end


function testAbsoluteWindowsStylePaths(testCase)
    test_dir = fileparts(mfilename('fullpath'));
    tool_dir = fileparts(test_dir);
    addpath(tool_dir);
    path_cleanup = onCleanup(@() rmpath(tool_dir));
    root_dir = tempname;
    run_dir = fullfile(root_dir, 'runs', 'TestRun');
    output_dir = fullfile(root_dir, 'exports', 'result');
    mkdir(run_dir);
    cleanup = onCleanup(@() remove_tree_if_exists(root_dir));

    write_single(fullfile(run_dir, 'Params_Detector.dat'), single([ ...
        2, ...
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, ...
        1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2]));
    write_single(fullfile(run_dir, 'Params_Image.dat'), ...
        single([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]));
    matrix_path = fullfile(run_dir, 'matrix.sysmat');
    write_single(matrix_path, single([3, 7]));

    result = export_run_sysmat_for_amide(run_dir, matrix_path, output_dir, ...
        'FilterScintillator', true);

    verifyEqual(testCase, result.selected_detector_count, 1);
    verifyEqual(testCase, result.dimensions, [1, 1, 1, 1, 1]);
    verifyEqual(testCase, read_single(fullfile(output_dir, 'SysMat_tmp')), single(3));
    verifyTrue(testCase, isfile(fullfile(output_dir, 'DetectorIndex.csv')));
end


function write_single(path, values)
    file_id = fopen(path, 'wb', 'ieee-le');
    assert(file_id >= 0, 'Cannot write %s.', path);
    cleanup = onCleanup(@() fclose(file_id));
    fwrite(file_id, values, 'single');
end


function values = read_single(path)
    file_id = fopen(path, 'rb', 'ieee-le');
    assert(file_id >= 0, 'Cannot read %s.', path);
    cleanup = onCleanup(@() fclose(file_id));
    values = fread(file_id, inf, 'single=>single');
end


function remove_tree_if_exists(path)
    if isfolder(path)
        rmdir(path, 's');
    end
end
