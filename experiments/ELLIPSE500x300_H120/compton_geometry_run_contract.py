"""Explicit R1 imaging contract; legacy geometry regression retains its q3 cut."""
import json


def validate_run(study, *, regression, pilot, dry_run, iterations, save_step,
                 dataset, level, channels, sensitivity, geometry_sha, kernel_sha,
                 digest, config_directory):
    if (study['study'] != 'compton_response_geometry_v3' or
        study['variant'] != 'R1_stable_point' or study['group'] != 'ideal_first_scatter_v2' or
        study['max_min_standardized_arm'] != 3 or study['quality_domain'] != 'full_circle_132040' or
        study['iterations'] != 2000 or study['save_step'] != 50 or
        (dataset, level, channels) != ('NEMA_Body_H60', '1e9', 'compton-jscc') or
        geometry_sha != study['geometry_sha256'] or kernel_sha != study['kernel_sha256'] or
        sensitivity is None):
        raise ValueError('Frozen stable geometry imaging contract differs')
    if regression and (iterations, save_step) != (50, 50):
        raise ValueError('Geometry regression requires exactly 50 iterations')
    if pilot and (iterations, save_step) != (10, 10):
        raise ValueError('Geometry pilot requires exactly 10 iterations')
    if not (regression or pilot or dry_run) and (iterations, save_step) != (2000, 50):
        raise ValueError('Geometry formal imaging is bounded to 2000/save50')
    # Both mean/spatial validations must have actually passed; q/K tests alone
    # or a previously invalid unrotated source comparison cannot unlock imaging.
    for name, sha in study['validation_evidence_sha256'].items():
        path = config_directory/name
        if digest(path) != sha or json.loads(path.read_text())['status'] != 'PASSED':
            raise ValueError('Stable geometry independent validation is missing or on HOLD')
    # Unlike response_mismatch_cut3_v1, R0 already used q3. Turning it off in
    # this 50-iteration regression would compare a different event collection.
    return ('legacy' if regression else 'stable_float64'), 3.0
