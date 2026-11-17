"""Serial bounded SFTP reads; never start an unbounded prefetch thread."""
import hashlib
import os
from pathlib import Path
import time
import contextlib
import re
import subprocess
import shutil
import json
import zlib
import base64
import socket


@contextlib.contextmanager
def sftp_bounded(host):
    import paramiko
    from ehe_5e9_workflow import connection
    with connection(host) as client:
        session=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=64<<10,max_packet_size=16<<10)
        session.get_channel().settimeout(30)
        try:yield session
        finally:session.close()


def download_serial(session, remote, partial, expected_sha256, timeout=30, maximum_seconds=600):
    partial=Path(partial)
    if partial.exists():raise FileExistsError('Interrupted transfer retained: '+str(partial))
    session.get_channel().settimeout(timeout)
    size=session.stat(remote).st_size
    h=hashlib.sha256();count=0;started=time.monotonic()
    with session.open(remote,'rb') as src, partial.open('xb') as dst:
        # SFTPFile.read without prefetch: requests are bounded and completed in order.
        while True:
            chunk=src.read(1<<20)
            if not chunk:break
            dst.write(chunk);h.update(chunk);count+=len(chunk)
            if time.monotonic()-started>maximum_seconds:raise TimeoutError('Bounded strict transfer timeout')
        dst.flush();os.fsync(dst.fileno())
    if count!=size or h.hexdigest()!=expected_sha256:
        raise ValueError('Strict sequential transfer byte count/SHA mismatch; partial retained')
    print('STRICT_SERIAL_MEMBER',partial.name,count,flush=True)
    return dict(bytes=count,sha256=h.hexdigest(),elapsed_seconds=time.monotonic()-started,prefetch=False)


def download_cpu(remote,partial,expected_sha256,maximum_seconds=600):
    """Bounded small compressed SSH replies for the Maty download channel."""
    partial=Path(partial)
    if partial.exists():raise FileExistsError('Interrupted transfer retained: '+str(partial))
    if not remote.startswith('/') or not re.fullmatch(r'[A-Za-z0-9_./-]+',remote):
        raise ValueError('Only a registered literal absolute remote member is permitted')
    from ehe_5e9_workflow import q,connection
    source="""import sys,json,pathlib,hashlib,zlib,base64
c=json.loads(sys.argv[1]);b=pathlib.Path(c['path']).read_bytes();s=hashlib.sha256(b).hexdigest()
assert s==c['sha256'];z=zlib.compress(b,1)
for i in range(0,len(z),512):
 piece=z[i:i+512]
 print('CHUNK_JSON '+json.dumps(dict(offset=i,compressed_bytes=len(z),bytes=len(b),sha256=s,data=base64.b64encode(piece).decode())),flush=True)
 assert sys.stdin.readline().strip()=='NEXT'
print('TRANSFER_DONE '+s,flush=True)
"""
    started=time.monotonic();offset=0;count=0;h=hashlib.sha256();unpack=zlib.decompressobj()
    with connection('maty') as client, partial.open('xb') as dst:
        client.get_transport().sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        channel=client.get_transport().open_session(window_size=64<<10,max_packet_size=4096,timeout=30)
        channel.settimeout(30)
        try:
            channel.exec_command('python3 -B -u -c '+q(source)+' '+q(json.dumps(dict(path=remote,sha256=expected_sha256))))
            stdout=channel.makefile('rb')
            while True:
                raw=stdout.readline(2048).decode('utf8')
                if not raw.startswith('CHUNK_JSON '):raise ValueError('Incomplete bounded transfer reply')
                row=json.loads(raw[len('CHUNK_JSON '):])
                if row['offset']!=offset or row['sha256']!=expected_sha256:raise ValueError('Transfer chunk identity differs')
                chunk=base64.b64decode(row['data'],validate=True)
                if not chunk and offset<row['compressed_bytes']:raise ValueError('Truncated compressed member')
                b=unpack.decompress(chunk);dst.write(b);h.update(b);count+=len(b);offset+=len(chunk)
                channel.sendall(b'NEXT\n')
                if offset%65536==0:print('STRICT_CHUNK_PROGRESS',partial.name,offset,row['compressed_bytes'],flush=True)
                if time.monotonic()-started>maximum_seconds:raise TimeoutError('Bounded chunked transfer timeout')
                if offset==row['compressed_bytes']:break
                if offset>row['compressed_bytes']:raise ValueError('Compressed member overrun')
            if stdout.readline().decode('utf8').strip()!='TRANSFER_DONE '+expected_sha256 or channel.recv_exit_status()!=0:
                raise ValueError('Transfer sender did not exit successfully')
            b=unpack.flush();dst.write(b);h.update(b);count+=len(b);dst.flush();os.fsync(dst.fileno())
        finally:channel.close()
    if not unpack.eof or unpack.unused_data or count!=row['bytes'] or h.hexdigest()!=expected_sha256:
        raise ValueError('Strict chunked size/SHA mismatch; partial retained')
    print('STRICT_CHUNKED_MEMBER',partial.name,count,flush=True)
    return dict(bytes=count,sha256=h.hexdigest(),elapsed_seconds=time.monotonic()-started,
                backend='acknowledged 512-byte compressed SSH replies, complete member SHA',compressed_bytes=offset)
