function run_gen_ellipse500x300_h120_params()
% Generate new response parameters without changing existing FOV120 runs.
    engine = fileparts(fileparts(mfilename('fullpath')));
    repo = fileparts(fileparts(engine));
    cfg = fullfile(repo, 'experiments', 'ELLIPSE500x300_H120', 'config.json');
    generate_jscc_218_440_response_params('_pe_v4_ELLIPSE500x300_H120', cfg);
end
