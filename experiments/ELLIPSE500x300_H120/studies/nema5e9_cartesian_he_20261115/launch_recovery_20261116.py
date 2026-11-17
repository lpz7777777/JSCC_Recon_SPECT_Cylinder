"""Windows-safe deployment transport: Python program on stdin, not argv.

The first deployment attempt exceeded the Windows command-line limit before
starting SSH. Its frozen repair source is unchanged; no calculation was started.
"""
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'repair_20261116'))
import repair_workflow as recovery
from common import sha, write, now


def server_stdin(code,timeout=120):
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',recovery.SERVER,
         'env PYTHONPATH='+shlex.quote(recovery.SERVER_DEPS)+' '+shlex.quote(recovery.SERVER_PY)+' -B -']
    r=subprocess.run(cmd,input=code,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr or r.stdout)
    return r.stdout


if __name__=='__main__':
    recovery.server=server_stdin
    recovery.start()
    write(recovery.REPORT/'repair_launch_transport_20261116.json',dict(time=now(),
        entry_sha256=sha(__file__),transport='SSH stdin; no source changes',
        original_predeploy_error='WinError 206 before SSH process creation'))
