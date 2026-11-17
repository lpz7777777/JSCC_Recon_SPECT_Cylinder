"""Length-framed lossless SSH transport; no bulk read depends on pipe EOF."""
from pathlib import Path
import json
import shlex
import socket
import struct
import subprocess
import tempfile
import threading
import time
import stream_transfer as original
from stream_transfer import SERVER,SERVER_PY,SERVER_DEPS,connect,server,REMOTE_GPU,GPU_PY,rpc,checksum

SENDER=r'''import sys,json,os,zlib,struct
c=json.loads(sys.argv[1]);f=open(c['source'],'rb',buffering=0)
assert os.fstat(f.fileno()).st_size==c['source_bytes']
for start,length in c['ranges']:
 f.seek(start);left=length
 while left:
  b=f.read(min(1<<20,left));assert b;left-=len(b)
  out=zlib.compress(b,1) if c['compression']=='zlib1' else b
  sys.stdout.buffer.write(struct.pack('!II',len(b),len(out)));sys.stdout.buffer.write(out)
sys.stdout.buffer.write(struct.pack('!II',0,0));sys.stdout.buffer.flush()
assert os.fstat(f.fileno()).st_size==c['source_bytes']
'''

RECEIVER=r'''import sys,json,pathlib,hashlib,os,zlib,struct,time
c=json.loads(sys.argv[1]);p=pathlib.Path(c['target']);p.parent.mkdir(parents=True,exist_ok=True)
expected=c['expected_bytes'];offset=c['offset'];h=hashlib.sha256();start=time.monotonic()
if offset:
 assert p.stat().st_size==offset
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 assert h.hexdigest()==c['prefix_sha256']
else:assert not p.exists()
def exact(n):
 data=bytearray()
 while len(data)<n:
  b=sys.stdin.buffer.read(n-len(data))
  if not b:raise ValueError('Truncated length-framed stream')
  data.extend(b)
 return bytes(data)
count=offset;wire=0;frames=0
with p.open('ab' if offset else 'xb',buffering=1<<20) as dst:
 while True:
  head=exact(8);wire+=8;raw_n,wire_n=struct.unpack('!II',head)
  if raw_n==wire_n==0:break
  if not 0<raw_n<=1<<20 or not 0<wire_n<=(1<<20)+65536:raise ValueError('Invalid frame bounds')
  payload=exact(wire_n);wire+=wire_n
  if c['compression']=='zlib1':
   decoder=zlib.decompressobj();b=decoder.decompress(payload,raw_n+1)
   if len(b)!=raw_n or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:raise ValueError('Invalid compressed frame')
  else:b=payload
  if len(b)!=raw_n or count+len(b)>expected:raise ValueError('Frame exceeds registered source range')
  dst.write(b);h.update(b);count+=len(b);frames+=1
 dst.flush();os.fsync(dst.fileno())
if count!=expected or h.hexdigest()!=c['expected_sha256']:raise ValueError('Uncompressed byte/SHA closure failed')
print(json.dumps(dict(passed=True,bytes=count,wire_bytes=wire,sha256=h.hexdigest(),offset=offset,
 compression=c['compression'],frames=frames,receiver_seconds=time.monotonic()-start,fsync=True)),flush=True)
'''


def bridge(client,source,source_bytes,ranges,target,expected_bytes,expected_sha256,compression,
           offset=0,prefix_sha256=None,progress=False,maximum_seconds=900):
    started=time.monotonic();wire=0;frame_count=0;timed_out=threading.Event()
    sendcfg=dict(source=source,source_bytes=source_bytes,ranges=ranges,compression=compression)
    recvcfg=dict(target=target,expected_bytes=expected_bytes,expected_sha256=expected_sha256,
                 compression=compression,offset=offset,prefix_sha256=prefix_sha256)
    cmd='env PYTHONPATH='+shlex.quote(SERVER_DEPS)+' '+shlex.quote(SERVER_PY)+' -B -u -c '+shlex.quote(SENDER)+' '+shlex.quote(json.dumps(sendcfg))
    transport=client.get_transport();transport.set_keepalive(20)
    transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    channel=transport.open_session(window_size=16<<20,max_packet_size=128<<10,timeout=30);channel.settimeout(120)
    channel.exec_command(shlex.quote(GPU_PY)+' -B -u -c '+shlex.quote(RECEIVER)+' '+shlex.quote(json.dumps(recvcfg)))
    error_log=tempfile.TemporaryFile()
    sender=subprocess.Popen(['ssh','-n','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=30',
        '-o','ServerAliveCountMax=3','-o','Compression=no',SERVER,cmd],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=error_log,bufsize=0)
    def deadline():
        timed_out.set()
        if sender.poll() is None:sender.terminate()
        channel.close()
    watchdog=threading.Timer(maximum_seconds,deadline);watchdog.daemon=True;watchdog.start()
    def exact(n):
        data=bytearray()
        while len(data)<n:
            b=sender.stdout.read(n-len(data))
            if not b:raise EOFError('Source ended before framed range completed')
            data.extend(b)
        return bytes(data)
    try:
        raw_count=0
        while True:
            head=exact(8);raw_n,wire_n=struct.unpack('!II',head)
            if raw_n==wire_n==0:
                channel.sendall(head);wire+=8;break
            if not 0<raw_n<=1<<20 or not 0<wire_n<=(1<<20)+65536:raise ValueError('Source frame bounds invalid')
            channel.sendall(head);wire+=8;left=wire_n
            while left:
                b=exact(min(left,128<<10));channel.sendall(b);left-=len(b);wire+=len(b)
            raw_count+=raw_n;frame_count+=1
            if raw_count>expected_bytes-offset:raise ValueError('Source framed payload grew')
        if raw_count!=expected_bytes-offset:raise ValueError('Framed range byte count differs')
        channel.shutdown_write()
        out=channel.makefile('rb').read().decode('utf8','replace')
        err=channel.makefile_stderr('rb').read().decode('utf8','replace')
        rc=channel.recv_exit_status()
        # The source process is bounded even if Windows OpenSSH fails to close.
        source_rc=sender.wait(timeout=45)
        error_log.seek(0);source_err=error_log.read().decode('utf8','replace')
        if rc or source_rc:raise RuntimeError('Framed sender/receiver failure '+str((source_rc,rc))+' '+source_err[-1000:]+' '+err[-1000:])
        result=json.loads(out)
        assert result['passed'] and result['bytes']==expected_bytes and result['sha256']==expected_sha256 and result['wire_bytes']==wire
        result.update(source=source,target=target,source_exit=source_rc,destination_exit=rc,
            elapsed_seconds=time.monotonic()-started,effective_MiB_per_second=(expected_bytes-offset)/(time.monotonic()-started)/(1<<20),
            protocol='bounded length frames, explicit end frame, stdin DEVNULL',local_full_matrix_cache=False,
            channel_window_bytes=16<<20,local_buffer_bytes=128<<10,frame_count=frame_count)
        return result
    finally:
        watchdog.cancel();channel.close()
        if sender.poll() is None:
            sender.terminate()
            try:sender.wait(timeout=10)
            except subprocess.TimeoutExpired:sender.kill();sender.wait(timeout=10)
        error_log.close()
