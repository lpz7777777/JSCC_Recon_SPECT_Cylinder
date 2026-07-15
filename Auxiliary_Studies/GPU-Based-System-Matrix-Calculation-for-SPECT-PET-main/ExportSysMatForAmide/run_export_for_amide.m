function result = run_export_for_amide()
%RUN_EXPORT_FOR_AMIDE Interactively select a run and matrix for AMIDE export.

    tool_dir = fileparts(mfilename('fullpath'));
    repo_root = fileparts(tool_dir);
    runs_root = fullfile(repo_root, 'runs');

    entries = dir(runs_root);
    entries = entries([entries.isdir]);
    keep = false(size(entries));
    for index = 1:numel(entries)
        name = entries(index).name;
        if startsWith(name, '.'); continue; end
        candidate = fullfile(runs_root, name);
        keep(index) = isfile(fullfile(candidate, 'Params_Detector.dat')) ...
            && isfile(fullfile(candidate, 'Params_Image.dat')) ...
            && ~isempty(dir(fullfile(candidate, '*.sysmat')));
    end
    entries = entries(keep);
    if isempty(entries)
        error('run_export_for_amide:NoRuns', ...
            'No runs directory contains parameters and a .sysmat file.');
    end
    [~, order] = sort(lower({entries.name}));
    entries = entries(order);

    fprintf('\nAvailable runs\n');
    fprintf('--------------\n');
    for index = 1:numel(entries)
        fprintf('  %2d. %s\n', index, entries(index).name);
    end
    run_index = read_choice('Select run number: ', numel(entries));
    run_name = entries(run_index).name;
    run_dir = fullfile(runs_root, run_name);

    matrices = dir(fullfile(run_dir, '*.sysmat'));
    [~, order] = sort(lower({matrices.name}));
    matrices = matrices(order);
    fprintf('\nMatrices in %s\n', run_name);
    fprintf('------------------------------\n');
    for index = 1:numel(matrices)
        fprintf('  %2d. %-72s %9.2f MiB\n', index, matrices(index).name, ...
            double(matrices(index).bytes) / 1024 / 1024);
    end
    matrix_index = read_choice('Select matrix number: ', numel(matrices));
    matrix_name = matrices(matrix_index).name;
    [~, matrix_stem] = fileparts(matrix_name);

    default_output = fullfile(repo_root, 'AmideExports', run_name, matrix_stem);
    fprintf('\nDefault output directory:\n  %s\n', default_output);
    custom_output = strtrim(input('Press Enter to use it, or type another output path: ', 's'));
    if isempty(custom_output)
        output_dir = default_output;
    else
        output_dir = custom_output;
    end

    overwrite = false;
    if isfile(fullfile(output_dir, 'SysMat_tmp')) ...
            || isfile(fullfile(output_dir, 'SysMat_tmp.partial'))
        answer = lower(strtrim(input('Output already exists. Overwrite it? [y/N]: ', 's')));
        overwrite = strcmp(answer, 'y') || strcmp(answer, 'yes');
        if ~overwrite
            error('run_export_for_amide:Cancelled', 'Export cancelled; existing output was not changed.');
        end
    end

    result = export_run_sysmat_for_amide(run_dir, matrix_name, output_dir, ...
        'FilterScintillator', true, 'Overwrite', overwrite);
end


function choice = read_choice(prompt, maximum)
    raw = strtrim(input(prompt, 's'));
    choice = str2double(raw);
    if ~isfinite(choice) || choice < 1 || choice > maximum || fix(choice) ~= choice
        error('run_export_for_amide:BadChoice', ...
            'Selection must be an integer from 1 to %d.', maximum);
    end
end
