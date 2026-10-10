"""Bounded postprocessing only after independently accepted new matrix-Poisson200."""
import argparse
import os
from pathlib import Path
import time

from ehe_common import read, write, digest, verify_files
from ehe_forward_poisson_5e10_workflow import DATA, REPORT, DOSE
from ehe_5e10_workflow import pid_alive


def finish():
    proof = read(REPORT / 'formal_summary.json')
    if not proof['passed'] or proof['iterations'] != 200 or proof['frames_per_channel'] != 20:
        raise ValueError('Strict full formal200 authority required')
    verify_files(DATA / 'results/formal', proof['files'])
    from plot_ehe_forward_poisson_5e10 import OUT, main as plot
    from verify_ehe_forward_poisson_5e10_figures import verify
    if OUT.exists() and not (OUT / 'artifact_manifest.json').exists():
        raise ValueError('Partial comparison preserved; no automatic overwrite')
    if not OUT.exists():
        plot()
    if not (REPORT / 'scientific_figure_acceptance.json').exists():
        verify()
    numeric = read(REPORT / 'scientific_figure_acceptance.json')
    if not numeric['passed']:
        raise ValueError('Actual scientific numeric QA required')
    generated = read(REPORT / 'generation_summary.json')
    comparison = read(OUT / 'comparison_report.json')
    run = read(DATA / 'results/formal/run_manifest.json')
    write(REPORT / 'numerical_delivery.json', dict(
        passed=True, expected_emitted_photons=DOSE,
        transport_performed=False, data_kind=generated['data_kind'],
        formal_job=read(REPORT / 'formal_job.json')['job'],
        formal_verification_job=read(REPORT / 'formal_acceptance_job.json')['job'],
        frames_per_channel=20, atomic_checkpoints=len(proof['checkpoints']),
        counts_sha256=proof['counts_sha256'],
        formal_authority_sha256=digest(REPORT / 'formal_summary.json'),
        artifact_manifest_sha256=digest(OUT / 'artifact_manifest.json'),
        scientific_qa_sha256=digest(REPORT / 'scientific_figure_acceptance.json'),
        visual_qa='Pending direct inspection of all nine actual exported scientific figures',
        final_delivery=False, physical_calibration_claim=False,
        postprocessing_source_sha256=digest(__file__)))
    lines = ['# New matrix-Poisson5e10 numerical results; direct visual QA pending', '',
        'Expected-source5e10 forward means, independent component Poisson noise,',
        'full-input validation10, original sequential MLEM200 and strict fetch are complete.',
        'Numeric scalar/3D ROI QA passed; direct visual inspection/final report remain pending.', '',
        f"New seeds: {generated['noise_seeds']}.",
        f"Expected source budget: {generated['budget']}.",
        f"Components: {generated['components']}.",
        f"Sampled windows: {generated['window_counts']}.",
        f"Generated cross fraction: {generated['generated_cross_fraction']}.",
        f"Fixed background from this own final440200: {comparison['fixed_cross_background_total']}.",
        f"Actual solver phase seconds: {run['phase_seconds']}.",
        f"Actual Slurm MaxRSS/AllocTRES bytes: {proof['slurm_maxrss_bytes']} / {run['allocation']['host_allocated_bytes']}.", '',
        'Nine figures compare18 routes: new matrix-Poisson5e10, accepted Geant4 5e10,',
        'matrix-Poisson5e9, Geant4 5e9 and six JSCC routes. EHE0-200 and JSCC0-10000',
        'retain their separate iteration ranges. Authoritative H60 3D truth/ROIs,',
        'crop0/no smoothing/no fitted gain/fixed emitted-source density scale.',
        'Model-generated observations do not independently calibrate physical response;',
        'same dose does not mean same response/noise realization or equal convergence.', '']
    (REPORT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print('NEW_MATRIX_POISSON5E10_NUMERIC_COMPLETE_VISUAL_QA_PENDING', flush=True)


def main(hours):
    path = DATA / 'postprocess_registration.json'
    if path.exists():
        previous = read(path)
        if previous['status'] == 'running' and pid_alive(previous['pid']):
            raise RuntimeError('Registered local postprocessor is alive')
        if previous['status'] == 'complete':
            return
    value = dict(pid=os.getpid(), status='running', started_epoch=time.time(),
                 source_sha256=digest(__file__), bounded_hours=hours,
                 advance_fetch_submit=False, recurring_automation=False)
    write(path, value)
    began = time.monotonic()
    try:
        while not (REPORT / 'formal_summary.json').exists():
            controller = read(DATA / 'controller.json')
            if controller['status'] == 'failed':
                raise RuntimeError('Scientific workflow failed; preserve all evidence and stop')
            if time.monotonic() - began >= hours * 3600:
                raise TimeoutError('Bounded wait elapsed; preserve all jobs/results')
            time.sleep(30)
        finish()
    except BaseException as exc:
        value.update(status='failed', error=str(exc), finished_epoch=time.time())
        write(path, value)
        raise
    else:
        value.update(status='complete', exit_code=0, finished_epoch=time.time())
        write(path, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=8)
    main(parser.parse_args().hours)
