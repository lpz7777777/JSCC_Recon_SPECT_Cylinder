"""Bounded byte-exact, optionally losslessly compressed SSH stdin transfer.

The local bridge holds at most a small socket buffer, not a matrix. Successful
destinations are immutable. A stopped SFTP prefix is verified at both endpoints
before resuming; the final uncompressed SHA must equal the original manifest.
"""
from pathlib import Path
import hashlib
import json
import shlex
import socket
import subprocess
import sys
import time
BASE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(BASE),str(BASE/'repair_20261116')]
from workflow import SERVER,SERVER_PY,SERVER_DEPS,connect,server,REMOTE_GPU,GPU_PY
from handoff import rpc,checksum
from common import read,write,now
from prepare import REPORT

PROGRESS=REPORT/'transfer_stream_progress_20261117.json'

SENDER=r'''import sys,json,os,zlib
c=json.loads(sys.argv[1]);f=open(c['source'],'rb',buffering=0)
assert os.fstat(f.fileno()).st_size==c['source_bytes']
compressor=zlib.compressobj(1) if c['compression']=='zlib1' else None
for start,length in c['ranges']:
 f.seek(start);left=length
 while left:
  b=f.read(min(1<<20,left))
  if not b:raise ValueError('Unexpected source EOF')
  left-=len(b);out=compressor.compress(b) if compressor else b
  if out:sys.stdout.buffer.write(out)
if compressor:sys.stdout.buffer.write(compressor.flush())
sys.stdout.buffer.flush()
assert os.fstat(f.fileno()).st_size==c['source_bytes']
'''

RECEIVER=r'''import sys,json,pathlib,hashlib,os,zlib,time
c=json.loads(sys.argv[1]);p=pathlib.Path(c['target']);p.parent.mkdir(parents=True,exist_ok=True)
expected=c['expected_bytes'];offset=c['offset'];started=time.monotonic();h=hashlib.sha256()
if offset:
 assert p.stat().st_size==offset
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 assert h.hexdigest()==c['prefix_sha256']
else:assert not p.exists()
count=offset;wire=0;decoder=zlib.decompressobj() if c['compression']=='zlib1' else None
with p.open('ab' if offset else 'xb',buffering=1<<20) as dst:
 def write(b):
  global count
  if count+len(b)>expected:raise ValueError('Uncompressed data exceeds manifest')
  dst.write(b);h.update(b);count+=len(b)
 while True:
  raw=sys.stdin.buffer.read(128<<10)
  if not raw:break
  wire+=len(raw)
  if decoder:
   pending=raw
   while pending:
    b=decoder.decompress(pending,4<<20);write(b);pending=decoder.unconsumed_tail
   if decoder.unused_data:raise ValueError('Unexpected trailing compressed stream')
  else:write(raw)
 if decoder:
  write(decoder.flush())
  if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:raise ValueError('Incomplete compressed stream')
 dst.flush();os.fsync(dst.fileno())
if count!=expected or h.hexdigest()!=c['expected_sha256']:raise ValueError('Final matrix byte count or SHA mismatch')
print(json.dumps(dict(passed=True,bytes=count,wire_bytes=wire,sha256=h.hexdigest(),offset=offset,
 compression=c['compression'],receiver_seconds=time.monotonic()-started,fsync=True)),flush=True)
'''


def bridge(client,source,source_bytes,ranges,target,expected_bytes,expected_sha256,compression,
           offset=0,prefix_sha256=None,progress=True,maximum_seconds=8*3600):
    started=time.monotonic();wire=0;last=started
    sendcfg=dict(source=source,source_bytes=source_bytes,ranges=ranges,compression=compression)
    recvcfg=dict(target=target,expected_bytes=expected_bytes,expected_sha256=expected_sha256,
                 compression=compression,offset=offset,prefix_sha256=prefix_sha256)
    source_cmd='env PYTHONPATH='+shlex.quote(SERVER_DEPS)+' '+shlex.quote(SERVER_PY)+' -B -u -c '+shlex.quote(SENDER)+' '+shlex.quote(json.dumps(sendcfg))
    transport=client.get_transport();transport.set_keepalive(20)
    transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    channel=transport.open_session(window_size=16<<20,max_packet_size=128<<10,timeout=30)
    channel.settimeout(120)
    channel.exec_command(shlex.quote(GPU_PY)+' -B -u -c '+shlex.quote(RECEIVER)+' '+shlex.quote(json.dumps(recvcfg)))
    sender=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
        '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-o','Compression=no',SERVER,source_cmd],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        while True:
            b=sender.stdout.read(128<<10)
            if not b:break
            channel.sendall(b);wire+=len(b)
            elapsed=time.monotonic()-started
            if elapsed>maximum_seconds:raise TimeoutError('Bounded SSH transfer deadline')
            if progress and time.monotonic()-last>=15:
                write(PROGRESS,dict(status='STREAMING',time=now(),source=source,target=target,
                    compression=compression,source_bytes=source_bytes,expected_sha256=expected_sha256,
                    verified_resume_offset_bytes=offset,wire_bytes_sent=wire,elapsed_seconds=elapsed,
                    wire_MiB_per_second=wire/max(elapsed,1e-6)/(1<<20)),replace=True)
                print('MATRIX_STREAM',compression,wire,round(elapsed,1),flush=True);last=time.monotonic()
        channel.shutdown_write()
        out=channel.makefile('rb').read().decode('utf8','replace')
        err=channel.makefile_stderr('rb').read().decode('utf8','replace')
        rc=channel.recv_exit_status();source_rc=sender.wait(timeout=45)
        source_err=sender.stderr.read().decode('utf8','replace')
        if source_rc or rc:raise RuntimeError('SSH stream failed: '+str((source_rc,rc))+' '+source_err[-1200:]+' '+err[-1200:])
        result=json.loads(out.strip())
        if not result['passed'] or result['bytes']!=expected_bytes or result['sha256']!=expected_sha256 or result['wire_bytes']!=wire:
            raise ValueError('Independent destination stream receipt differs')
        result.update(elapsed_seconds=time.monotonic()-started,source_exit=source_rc,destination_exit=rc,
            effective_MiB_per_second=(expected_bytes-offset)/(time.monotonic()-started)/(1<<20),
            source=source,target=target,local_full_matrix_cache=False,
            channel_window_bytes=16<<20,local_buffer_bytes=128<<10)
        if progress:write(PROGRESS,dict(result,status='TRANSFER_VERIFIED',time=now()),replace=True)
        return result
    finally:
        channel.close()
        if sender.poll() is None:sender.terminate();sender.wait(timeout=30)


def transfer_one(client,source,target,expected,size):
    # The frozen handoff calls this adapter; all production math stays unchanged.
    existing=checksum(client,target)
    if existing!='ABSENT':
        if existing!=expected:raise ValueError('Different immutable successful destination')
        return dict(source=source,target=target,bytes=size,sha256=expected,reused_verified=True)
    partial=target+'.partial'
    code='''from pathlib import Path
import json
p=Path(FILE)
print(json.dumps(dict(bytes=p.stat().st_size if p.exists() else 0,exists=p.exists())))'''.replace('FILE',repr(partial))
    prefix=json.loads(rpc(client,code,60));offset=prefix['bytes']
    if offset>size:raise ValueError('Retained prefix exceeds frozen matrix')
    prefix_sha=None
    if offset:
        target_hash=checksum(client,partial)
        source_hash=server('''import hashlib
f=open(FILE,'rb');h=hashlib.sha256();left=COUNT
while left:
 b=f.read(min(4<<20,left))
 if not b:raise ValueError('Source prefix incomplete')
 h.update(b);left-=len(b)
print(h.hexdigest())'''.replace('FILE',repr(source)).replace('COUNT',str(offset)),timeout=300).strip()
        if target_hash!=source_hash:raise ValueError('Stopped SFTP prefix differs; preserve without overwrite')
        prefix_sha=source_hash
    write(REPORT/('transfer_resume_'+Path(target).parent.name+'_20261117.json'),dict(time=now(),source=source,
        target=partial,preserved_prefix_bytes=offset,prefix_sha256=prefix_sha,expected_full_sha256=expected),replace=False)
    policy=read(REPORT/'transfer_benchmark_20261117.json')
    compression=policy['selected_compression']
    if offset==size:
        if prefix_sha!=expected:raise ValueError('Full existing partial SHA mismatch')
        result=dict(passed=True,bytes=size,sha256=expected,elapsed_seconds=0,offset=offset,wire_bytes=0)
    else:
        result=bridge(client,source,size,[[offset,size-offset]],partial,size,expected,compression,
                      offset=offset,prefix_sha256=prefix_sha)
    # Re-read destination independently, then atomic promotion, no success overwrite.
    if checksum(client,partial)!=expected:raise ValueError('Independent post-transfer SHA differs')
    rpc(client,'from pathlib import Path;import os;p=Path('+repr(partial)+');q=Path('+repr(target)+');assert not q.exists();os.rename(p,q)')
    result.update(reused_verified=False,scientific_payload_changed=False,lossless=True)
    write(REPORT/('transfer_complete_'+Path(target).parent.name+'_20261117.json'),result)
    return result
