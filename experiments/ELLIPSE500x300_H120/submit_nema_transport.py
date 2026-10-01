"""Submit a gated NEMA transport chain once, using the existing SSH agent."""
import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

from deploy_nema_simulation import REMOTE, remote, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("1e9", "5e9", "1e10"), required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    record_path = here / f"reports/NEMA_Body_H60/{args.level}/transport_jobs.json"
    if record_path.exists():
        raise FileExistsError(f"Submission record already exists: {record_path}")
    jobs_path = here / f"generated/NEMA_Body_H60/Simulation_{args.level}/jobs.json"
    logs = f"{REMOTE}/generated/NEMA_Body_H60/Simulation_{args.level}/logs"
    record = {"level": args.level, "submitted_utc": datetime.now(timezone.utc).isoformat(),
              "prepared_manifest_sha256": digest(jobs_path), "jobs": {}}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist each accepted ID immediately, so a partially submitted chain is reviewable.
    def submit(stage, options, script="maty_nema_array.sh"):
        command = (f"cd {shlex.quote(REMOTE)} && sbatch --parsable "
                   f"--job-name=NEMA_{args.level}_{stage} "
                   f"--output={shlex.quote(logs+'/'+stage+'.%A_%a.out')} "
                   f"--error={shlex.quote(logs+'/'+stage+'.%A_%a.err')} "
                   f"{options} {shlex.quote(script)}")
        job_id = remote(command).strip().split(";")[0]
        if not job_id.isdigit():
            raise ValueError(f"Invalid submission response: {job_id}")
        record["jobs"][stage] = job_id
        record_path.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        print(f"{stage}: {job_id}", flush=True)
        return job_id
    remote(f"cd {shlex.quote(REMOTE)} && bash -n maty_nema_array.sh && bash -n maty_nema_collect.sh")
    smoke = submit("smoke", f"--array=0 --export=ALL,NEMA_LEVEL={args.level},NEMA_SMOKE=1")
    first = submit("first", f"--array=0 --dependency=afterok:{smoke} --export=ALL,NEMA_LEVEL={args.level},NEMA_SMOKE=0")
    production = submit("production", f"--array=1-199%32 --dependency=afterok:{first} --export=ALL,NEMA_LEVEL={args.level},NEMA_SMOKE=0")
    submit("collect", f"--dependency=afterok:{first}:{production} --export=ALL,NEMA_LEVEL={args.level}", "maty_nema_collect.sh")


if __name__ == "__main__":
    main()
