"""Bounded continuous SSH archive reads; no SFTP request-per-packet latency."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
from ehe_5e9_workflow import connection,q


def download_gpu_archive_stream(remote,partial,expected_sha256,expected_bytes,maximum_seconds=1800):
    if not remote.startswith('/data/run01/scxi717/lpz/') or not re.fullmatch(r'[A-Za-z0-9_./-]+',remote):
        raise ValueError('Only the authorized literal archive path may be streamed')
    partial=Path(partial)
    if partial.exists():raise FileExistsError('Existing partial preserved')
    source='''import sys,json,os
c=json.loads(sys.argv[1]);f=open(c['path'],'rb',buffering=0)
assert os.fstat(f.fileno()).st_size==c['bytes']
while True:
 b=f.read(1<<20)
 if not b:break
 sys.stdout.buffer.write(b)
sys.stdout.buffer.flush()
'''
    total=0;h=hashlib.sha256();started=time.monotonic();last_report=0
    with connection('gpu') as client:
        transport=client.get_transport()
        transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        transport.set_keepalive(20)
        channel=transport.open_session(window_size=4<<20,max_packet_size=32768,timeout=30)
        channel.settimeout(45)
        try:
            command='/data/home/scxi717/.conda/envs/torch/bin/python -B -u -c '+q(source)+' '+q(json.dumps(dict(path=remote,bytes=expected_bytes)))
            channel.exec_command(command)
            with partial.open('xb') as dst:
                while True:
                    b=channel.recv(128<<10)
                    if not b:break
                    total+=len(b)
                    if total>expected_bytes:raise ValueError('Archive stream exceeds registered byte count')
                    h.update(b);dst.write(b)
                    if total-last_report>=128<<20:
                        print('STRICT_ARCHIVE_STREAM_PROGRESS',total,expected_bytes,flush=True);last_report=total
                    if time.monotonic()-started>maximum_seconds:raise TimeoutError('Bounded archive stream duration')
                dst.flush();os.fsync(dst.fileno())
            error=b''
            while channel.recv_stderr_ready():error+=channel.recv_stderr(4096)
            status=channel.recv_exit_status()
            if status!=0:raise RuntimeError('Archive sender exit '+str(status)+': '+error[-2000:].decode('utf8','replace'))
        finally:channel.close()
    if total!=expected_bytes or h.hexdigest()!=expected_sha256:
        raise ValueError('Continuous archive exact byte/SHA closure failed; partial retained')
    result=dict(passed=True,bytes=total,sha256=h.hexdigest(),elapsed_seconds=time.monotonic()-started,
                channel_window_bytes=4<<20,read_buffer_bytes=128<<10,unbounded_prefetch=False)
    print('STRICT_ARCHIVE_STREAM_COMPLETE',json.dumps(result),flush=True)
    return result
