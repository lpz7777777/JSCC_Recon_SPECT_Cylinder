"""Byte transport regression: explicit-end ACK while stdin remains OPEN."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import threading
import zlib
import framed_stream_r2 as st
from common import write
from prepare import REPORT

def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='cart_frame_r2_') as tmp:
        base=Path(tmp);source=base/'source.bin';raw=bytes(range(256))*20000+b'tail37'
        source.write_bytes(raw)
        for compression in ('raw','zlib1'):
            cfg=dict(source=str(source),source_bytes=len(raw),ranges=[[17,len(raw)-54],[0,19]],compression=compression)
            sender=subprocess.run([sys.executable,'-B','-c',st.SENDER,json.dumps(cfg)],capture_output=True,check=True)
            expected=raw[17:-37]+raw[:19];pos=0;decoded=bytearray()
            while True:
                n,w=struct.unpack('!II',sender.stdout[pos:pos+8]);pos+=8
                if n==w==0:break
                b=sender.stdout[pos:pos+w];pos+=w
                data=zlib.decompress(b) if compression=='zlib1' else b
                assert len(data)==n;decoded.extend(data)
            assert pos==len(sender.stdout) and bytes(decoded)==expected
            for offset in (0,12345):
                target=base/(compression+str(offset)+'.bin')
                if offset:target.write_bytes(expected[:offset])
                payload=bytearray()
                for start in range(offset,len(expected),1<<20):
                    b=expected[start:start+(1<<20)];body=zlib.compress(b,1) if compression=='zlib1' else b
                    payload.extend(struct.pack('!II',len(b),len(body)));payload.extend(body)
                payload.extend(struct.pack('!II',0,0))
                cfg=dict(target=str(target),expected_bytes=len(expected),expected_sha256=hashlib.sha256(expected).hexdigest(),
                    compression=compression,offset=offset,prefix_sha256=hashlib.sha256(expected[:offset]).hexdigest() if offset else None)
                p=subprocess.Popen([sys.executable,'-B','-c',st.RECEIVER,json.dumps(cfg)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                timer=threading.Timer(10,lambda:p.terminate() if p.poll() is None else None);timer.start()
                try:
                    p.stdin.write(payload);p.stdin.flush()
                    # Deliberately do not close stdin: old protocol blocked here.
                    result=json.loads(p.stdout.readline())
                    assert p.wait(timeout=10)==0 and result['passed'] and target.read_bytes()==expected
                    checks.append(dict(compression=compression,offset=offset,explicit_end_ACK_without_EOF=True,passed=True))
                finally:
                    timer.cancel();p.stdin.close()
                    if p.poll() is None:p.terminate();p.wait(timeout=10)
        target=base/'invalid.bin'
        cfg=dict(target=str(target),expected_bytes=3,expected_sha256='0'*64,compression='raw',offset=0,prefix_sha256=None)
        invalid=struct.pack('!II',(1<<20)+1,3)+b'abc'
        result=subprocess.run([sys.executable,'-B','-c',st.RECEIVER,json.dumps(cfg)],input=invalid,capture_output=True)
        assert result.returncode and target.stat().st_size==0
        checks.append(dict(oversized_frame_rejected_before_payload_write=True))
    result=dict(passed=True,checks=checks,synthetic_bytes_only=True,explicit_end_not_EOF=True,source_models_unchanged=True)
    write(REPORT/'transfer_framed_checks_r2_20261117.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
