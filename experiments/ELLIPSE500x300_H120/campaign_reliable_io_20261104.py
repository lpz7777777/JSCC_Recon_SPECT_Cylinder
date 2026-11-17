"""Read-only acknowledged RPC and SHA-bound recovery; never retry submissions."""
import base64
import contextlib
import hashlib
import json
from pathlib import Path
import socket
import time
import zlib

import paramiko

from ehe_5e9_workflow import connection, q


class ReadOnlyUnavailable(RuntimeError):
    """No fresh read-only evidence; keep jobs intact and defer admission."""


TRANSIENT = (TimeoutError, ConnectionError, EOFError, paramiko.SSHException)


def decode_piece(row, offset, identity):
    if row['offset'] != offset or (row['bytes'], row['sha256']) != identity:
        raise ValueError('RPC chunk identity or offset differs')
    piece = base64.b64decode(row['data'], validate=True)
    if not piece or offset + len(piece) > row['compressed_bytes']:
        raise ValueError('RPC compressed byte count differs')
    return piece


def decode_message(packed, identity):
    unpack = zlib.decompressobj()
    raw = unpack.decompress(packed, identity[0]+1)
    if (not unpack.eof or unpack.unused_data or unpack.unconsumed_tail or len(raw) != identity[0]
            or hashlib.sha256(raw).hexdigest() != identity[1]):
        raise ValueError('Read-only RPC complete size/SHA/framing differs')
    return raw


def python_json(host, interpreter, source, arguments=None, timeout=90, attempts=2):
    """Only read-only callers: compressed <=512-byte messages, acknowledged."""
    sender = '''import sys,json,subprocess,zlib,hashlib,base64
c=json.loads(sys.argv[1])
try:
 p=subprocess.run([c['interpreter'],'-B','-c',c['source'],json.dumps(c['arguments'])],capture_output=True,timeout=c['timeout'])
 r=dict(rc=p.returncode,out=p.stdout.decode('utf8','replace'),err=p.stderr.decode('utf8','replace'))
except subprocess.TimeoutExpired:r=dict(unavailable=True,error='read-only subprocess timeout')
b=json.dumps(r).encode();s=hashlib.sha256(b).hexdigest();z=zlib.compress(b,1)
for i in range(0,len(z),512):
 print('RPC_CHUNK '+json.dumps(dict(offset=i,bytes=len(b),sha256=s,compressed_bytes=len(z),data=base64.b64encode(z[i:i+512]).decode())),flush=True)
 if sys.stdin.readline().strip()!='NEXT':raise RuntimeError('Missing RPC acknowledgement')
print('RPC_DONE '+s,flush=True)
'''
    for attempt in range(attempts):
        try:
            with connection(host) as client:
                transport = client.get_transport()
                transport.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                transport.set_keepalive(30)
                channel = transport.open_session(window_size=64 << 10, max_packet_size=4096, timeout=30)
                channel.settimeout(timeout + 15)
                try:
                    payload = dict(interpreter=interpreter, source=source, arguments=arguments, timeout=timeout)
                    channel.exec_command(q(interpreter)+' -B -u -c '+q(sender)+' '+q(json.dumps(payload)))
                    stdout = channel.makefile('rb'); packed = bytearray(); offset = 0; identity = None; compressed_size = None
                    began = time.monotonic()
                    while True:
                        line = stdout.readline(2048).decode('utf8')
                        if not line.startswith('RPC_CHUNK '):
                            raise ValueError('Incomplete read-only RPC protocol')
                        row = json.loads(line[len('RPC_CHUNK '):])
                        if row['bytes'] > 16 << 20:
                            raise ValueError('Read-only RPC exceeds bounded output')
                        if identity is None: identity = (row['bytes'], row['sha256'])
                        if compressed_size is None: compressed_size = row['compressed_bytes']
                        if row['compressed_bytes'] != compressed_size: raise ValueError('RPC compressed identity changed')
                        piece = decode_piece(row, offset, identity)
                        packed.extend(piece); offset += len(piece)
                        channel.sendall(b'NEXT\n')
                        if time.monotonic() - began > timeout + 60:
                            raise TimeoutError('Read-only RPC total bound')
                        if offset == row['compressed_bytes']: break
                    if stdout.readline().decode('utf8').strip() != 'RPC_DONE '+identity[1]:
                        raise ValueError('Read-only RPC final identity differs')
                    if channel.recv_exit_status() != 0:
                        raise ValueError('Read-only RPC sender did not exit successfully')
                    raw = decode_message(packed, identity)
                finally: channel.close()
            result = json.loads(raw)
            if result.get('unavailable'): raise ReadOnlyUnavailable(result['error'])
            if result['rc']:
                raise RuntimeError('Remote read-only command failed: '+(result['out']+result['err'])[-4000:])
            marker = 'CAMPAIGN_JSON '
            if marker not in result['out']: raise ValueError('Complete read-only JSON marker missing')
            return json.loads(result['out'].rsplit(marker, 1)[1])
        except TRANSIENT as error:
            if attempt + 1 == attempts:
                raise ReadOnlyUnavailable(host+' transport temporarily unavailable: '+type(error).__name__) from error
            time.sleep(min(5, attempt + 1))


def accounting(host, jobs, interpreter, timeout=40):
    source = '''import sys,json,subprocess
c=json.loads(sys.argv[1]);ids=c['jobs']
try:
 p=subprocess.run(['sacct']+(['--array'] if c['expanded_array'] else [])+['-n','-P','-j',','.join(map(str,ids)),'-o','JobID%40,State,ExitCode'],capture_output=True,text=True,timeout=30)
 r=dict(available=p.returncode==0,rows=[x.split('|')[:3] for x in p.stdout.splitlines() if x],error=p.stderr[-2000:])
except subprocess.TimeoutExpired:r=dict(available=False,error='sacct timeout30s',rows=[])
print('CAMPAIGN_JSON '+json.dumps(r))
'''
    result = python_json(host, interpreter, source, dict(jobs=list(jobs),expanded_array=host=='gpu'), timeout=timeout)
    if not result['available'] or not result['rows']:
        raise ReadOnlyUnavailable(host+' accounting unavailable: '+result.get('error', 'empty'))
    return result['rows']


def preserve_partial(partial, history):
    partial, history = Path(partial).resolve(), Path(history).resolve()
    if not partial.exists(): return None
    # History is explicitly inside the same dataset. Never move a directory tree.
    if partial.is_dir() or history.parent != partial.parent:
        raise ValueError('Partial history must be a sibling file history directory')
    history.mkdir(exist_ok=True)
    h = hashlib.sha256(partial.read_bytes()).hexdigest()
    target = history/(partial.name+'.'+str(time.time_ns()))
    record = dict(original=str(partial), preserved=str(target), bytes=partial.stat().st_size, sha256=h)
    partial.rename(target)
    target.with_name(target.name+'.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
    return record


def cpu_member(remote, target, expected, history=None, maximum_seconds=600):
    """Retry only a file read; all incomplete bytes retained, no SHA bypass."""
    from ac225_transfer_io_20261104 import download_cpu
    from derenzo_campaign_common import digest
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    history = Path(history) if history else target.parent/'transfer_history_r6'
    if target.exists():
        if digest(target) != expected: raise ValueError('Existing accepted member differs')
        return dict(reused=True, sha256=expected, bytes=target.stat().st_size)
    partial = target.with_name(target.name+'.partial')
    if partial.exists():
        if digest(partial) == expected:
            partial.replace(target)
            return dict(recovered_complete_partial=True, sha256=expected, bytes=target.stat().st_size)
        preserve_partial(partial, history)
    for attempt in range(3):
        try:
            proof = download_cpu(remote, partial, expected, maximum_seconds=maximum_seconds)
            if digest(partial) != expected: raise ValueError('Complete downloaded SHA differs')
            partial.replace(target)
            return proof
        except TRANSIENT as error:
            preserve_partial(partial, history)
            if attempt == 2:
                raise ReadOnlyUnavailable('CPU member read deferred after bounded attempts') from error
            time.sleep(attempt + 1)
