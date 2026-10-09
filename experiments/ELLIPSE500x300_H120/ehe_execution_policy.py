"""Bind explicit human continuation to unchanged, physically unvalidated inputs."""
from pathlib import Path
from ehe_common import RESPONSES, digest, read


def physical_permission(physical, counts, responses, release_manifest, policy_path=None):
    gate_path = Path(physical) / 'physical_gate.json'
    gate = read(gate_path)
    if gate['passed']:
        if policy_path is not None:
            raise ValueError('A HOLD continuation must bind the original failed gate')
        return dict(physical_calibration_passed=True, physical_policy_sha256=None,
                    execution_authorized_under_known_physics_mismatch=False)
    if policy_path is None:
        raise ValueError('Physical HOLD requires explicit bound human continuation')
    policy_path = Path(policy_path)
    policy = read(policy_path)
    if (policy.get('kind') != 'explicit_human_original_method_continuation'
            or policy.get('study') != 'ehe_spect_5e9_200'
            or policy.get('authorized_modes') != ['validation', 'formal']
            or policy.get('formal_iterations') != 200
            or policy.get('physical_calibration_passed') is not False
            or policy.get('acknowledges_all_recorded_hold_diagnostics') is not True
            or policy.get('preserve_original_scientific_algorithm') is not True):
        raise ValueError('Invalid continuation scope')
    producer = release_manifest.get('producer_release_key', release_manifest['release_key'])
    if policy['producer_release_key'] != producer:
        raise ValueError('Continuation scientific release differs')
    if policy['physical_gate_sha256'] != digest(gate_path):
        raise ValueError('Continuation gate identity differs')
    if policy['hold_count'] != gate['hold_count'] or policy['undetermined_bins'] != gate['undetermined_bins']:
        raise ValueError('Continuation diagnostic scope differs')
    if policy['counts_sha256'] != digest(Path(counts) / 'collection.json'):
        raise ValueError('Continuation observation identity differs')
    factors = {n: digest(Path(responses) / n / 'factor_manifest.json') for n in RESPONSES}
    if policy['factor_sha256'] != factors:
        raise ValueError('Continuation Factor identity differs')
    if policy['source_sha256'] != gate['source_sha256']:
        raise ValueError('Continuation source identity differs')
    if policy['physical_audit_sha256'] != gate['files']['physical_audit.csv']:
        raise ValueError('Continuation original audit identity differs')
    if policy['baseline_helper_sha256'] != {
            n: release_manifest['sha256'][n]
            for n in ('torch_active_operator.py', 'single_checkpoint_mlem.py')}:
        raise ValueError('Continuation original MLEM identity differs')
    return dict(physical_calibration_passed=False,
                physical_policy_sha256=digest(policy_path),
                execution_authorized_under_known_physics_mismatch=True,
                original_hold_count=gate['hold_count'],
                original_undetermined_bins=gate['undetermined_bins'])
