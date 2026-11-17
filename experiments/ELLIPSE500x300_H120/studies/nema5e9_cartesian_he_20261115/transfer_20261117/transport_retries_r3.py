"""Retry only recoverable I/O; never retry data-integrity or scientific failures."""
import json
import time
import socket
import subprocess
from pathlib import Path
import paramiko

class TransferIOError(OSError):pass

def transient(error):
    if isinstance(error,(paramiko.AuthenticationException,paramiko.BadHostKeyException)):
        return False
    return isinstance(error,(TransferIOError,TimeoutError,ConnectionError,EOFError,socket.timeout,
        paramiko.SSHException,subprocess.TimeoutExpired,OSError)) and not isinstance(error,(FileExistsError,PermissionError))

def retry_io(action,label,attempts=5,logger=None,sleep=time.sleep):
    for attempt in range(1,attempts+1):
        try:return action()
        except Exception as error:
            if not transient(error) or attempt==attempts:raise
            row=dict(label=label,attempt=attempt,error_type=type(error).__name__,error=str(error)[-1600:],delay_seconds=min(30,2**attempt))
            if logger:logger(row)
            print('TRANSFER_IO_RETRY '+json.dumps(row),flush=True)
            sleep(row['delay_seconds'])
