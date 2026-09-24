"""Finish an already-running isolated matrix job with validated raw Factors."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fov_config import load_config,validate_factor_geometry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-pid',type=int,required=True)
    parser.add_argument('--timeout-hours',type=float,default=12)
    args=parser.parse_args()
    start=time.monotonic()
    log=ROOT/'matrix_pipeline.log'
    marker=ROOT/'experiments/FOV120/generated/factor_conversion_status.json'
    marker.parent.mkdir(parents=True,exist_ok=True)
    def status(stage,**extra):
        marker.write_text(json.dumps({'stage':stage,**extra},indent=2)+'\n')
    status('waiting_for_matrices',pipeline_pid=args.pipeline_pid)
    try:
        while True:
            text=log.read_text(errors='replace') if log.exists() else ''
            if 'All three FOV120 response matrices completed and size checked' in text:
                break
            if 'Traceback (most recent call last)' in text:
                raise RuntimeError('Matrix job failed; inspect matrix_pipeline.log')
            try:os.kill(args.pipeline_pid,0)
            except ProcessLookupError:
                raise RuntimeError('Matrix process exited without completion marker')
            if time.monotonic()-start>args.timeout_hours*3600:
                raise TimeoutError('Matrix completion wait timed out; matrix job was not stopped')
            time.sleep(30)
        status('converting_raw_factors')
        tool=ROOT/'Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main/GenFactors'
        statement="addpath('"+str(tool).replace("'","''")+"'); run_gen_fov120_factors;"
        subprocess.run(['matlab','-batch',statement],cwd=ROOT,check=True)
        factors=ROOT/'experiments/FOV120/generated/FactorsRaw'
        report=validate_factor_geometry({name:factors/name for name in
            ('218keV_RotateNum20','440keV_RotateNum20','440keV_to218win_RotateNum20')},
            load_config(),scan_matrix=True)
        status('raw_factors_complete',geometry=report,
               next_stage='Independent high-count calibration and Sensi_d; raw Factors are not calibrated')
    except Exception as exc:
        status('failed',error=str(exc))
        raise


if __name__=='__main__':main()
