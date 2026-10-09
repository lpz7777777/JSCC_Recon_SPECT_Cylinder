"""One-time bounded local postprocessing after actual strict formal acceptance."""
import argparse, os, time
from pathlib import Path
from ehe_common import read, write, digest, verify_files
from ehe_5e10_workflow import DATA, REPORT, registered, pid_alive


def finish():
    proof=read(REPORT/'formal_summary.json')
    if not proof['passed'] or proof['iterations']!=200 or proof['frames_per_channel']!=20:
        raise ValueError('Actual complete strict formal200 authority required')
    verify_files(DATA/'results/formal',proof['files'])
    from plot_ehe_5e10 import OUT, main as plot
    from verify_ehe_5e10_figures import verify
    if OUT.exists() and not (OUT/'artifact_manifest.json').exists():
        raise ValueError('Partial comparison retained; diagnose without replacing it')
    if not OUT.exists():plot()
    if not (REPORT/'scientific_figure_acceptance.json').exists():verify()
    numeric=read(REPORT/'scientific_figure_acceptance.json')
    if not numeric['passed']:raise ValueError('Actual numeric figure QA required')
    comparison=read(OUT/'comparison_report.json')
    transport=read(REPORT/'transport_acceptance.json')
    run=read(DATA/'results/formal/run_manifest.json')
    identity=dict(passed=True,actual_primary_photons=50000000000,
        transport_job=registered('transport')['job'],formal_job=registered('formal')['job'],
        formal_verification_job=registered('formal_acceptance')['job'],
        frames_per_channel=20,atomic_checkpoints=len(proof['checkpoints']),
        counts_sha256=proof['counts_sha256'],formal_authority_sha256=digest(REPORT/'formal_summary.json'),
        artifact_manifest_sha256=digest(OUT/'artifact_manifest.json'),
        scientific_qa_sha256=digest(REPORT/'scientific_figure_acceptance.json'),
        visual_qa='Pending direct inspection of all nine actual figures',final_delivery=False,
        physical_calibration_claim=False,postprocessing_source_sha256=digest(Path(__file__)))
    write(REPORT/'numerical_delivery.json',identity)
    lines=['# Actual numerical results; direct visual QA pending','',
        'The complete independent full-4pi 5e10 transport, complete-input validation10,',
        'original sequential MLEM200 and strict all-file acceptance/fetch have completed.',
        'The atlas and raw-array/3D-ROI numeric QA exist. Final visual review and final',
        'reviewed report/Git delivery remain pending; this is not final experiment delivery.','',
        f"Actual CPU transport job: {registered('transport')['job']}; formal GPU job: {registered('formal')['job']};",
        f"independent formal verifier: {registered('formal_acceptance')['job']}.",
        f"Actual initial primary counts [218,440,other]: {transport['primary_counts']}.",
        f"Actual energy-window counts: {transport['window_counts']}.",
        f"Measured tagged 440->218 fraction in the 218 window: {comparison['actual_cross_fraction']:.9%}.",
        f"Fixed background from this own440200 image: {comparison['fixed_cross_background_total']:.9f} counts.",
        f"Actual solver phase seconds: {run['phase_seconds']}.",
        f"Formal Slurm MaxRSS: {proof['slurm_maxrss_bytes']} bytes; actual allocation denominator: {run['allocation']['host_allocated_bytes']} bytes.",'',
        'See comparison/comparison_report.json and the 15-route overall MIP,',
        'four new three-route slice/MIP galleries, complete CNR/CRC and density/noise/',
        'integral/peak-position curves. EHE uses 0-200 and JSCC 0-10000, with own',
        'iteration axes and actual emitted-density scaling. No gain fit or smoothing.',
        'The tenfold dose difference, model-versus-transport distinction, original',
        'response discrepancy, material/coverage and 440-background budget differences',
        'remain explicit. No physical calibration or true-device performance is claimed.','']
    (REPORT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('EHE_5E10_NUMERIC_ATLAS_COMPLETE_VISUAL_REVIEW_PENDING',flush=True)


def main(hours):
    path=DATA/'postprocess_registration.json'
    if path.exists():
        previous=read(path)
        if previous['status']=='running' and pid_alive(previous['pid']):
            raise RuntimeError('Registered local postprocessing process already alive')
        if previous['status']=='complete':return
    value=dict(pid=os.getpid(),status='running',started_epoch=time.time(),
        source_sha256=digest(Path(__file__)),bounded_hours=hours,advance_fetch_submit=False)
    write(path,value);began=time.monotonic()
    try:
        while not (REPORT/'formal_summary.json').exists():
            controller=read(DATA/'controller.json')
            if controller['status']=='failed':raise RuntimeError('Registered workflow failed; preserve evidence and stop postprocessing')
            if time.monotonic()-began>hours*3600:raise TimeoutError('Postprocessing wait bound reached; no jobs/results cancelled')
            time.sleep(30)
        finish()
    except BaseException as exc:
        value.update(status='failed',error=str(exc),finished_epoch=time.time());write(path,value);raise
    else:
        value.update(status='complete',exit_code=0,finished_epoch=time.time());write(path,value)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=8)
    main(parser.parse_args().hours)
