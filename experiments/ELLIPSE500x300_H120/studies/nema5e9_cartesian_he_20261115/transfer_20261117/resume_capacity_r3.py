"""Keep original storage reservation, credit only complete SHA-verified matrices."""
import framed_stream_r3 as st
from workflow import gpu_preflight as original_preflight,connect,REMOTE_GPU
from prepare import REPORT
from common import read,write,now

def remaining_reservation(verified_bytes):
    if not isinstance(verified_bytes,int) or not 0<=verified_bytes<=3*21_285_888_000:
        raise ValueError('Invalid verified matrix allocation credit')
    return max(8<<30,(80<<30)-verified_bytes)

def gpu_preflight():
    result=original_preflight()
    if 'available_bytes' not in result:return result
    complete=read(REPORT/'matrix_repair_latest_status_20261116.json')['completed.json']
    credit=0;verified={}
    with connect() as c:
        for channel,r in complete['channels'].items():
            target=REMOTE_GPU+'/matrices/'+channel+'/B_detector_zyx.float32'
            digest=st.checksum(c,target)
            if digest=='ABSENT':continue
            if digest!=r['sha256']:raise ValueError('Cannot credit altered successful target matrix')
            credit+=r['bytes'];verified[channel]=dict(bytes=r['bytes'],sha256=digest)
    result=dict(result,time=now(),original_full_reservation_bytes=80<<30,
        SHA_verified_completed_matrix_credit_bytes=credit,credited_matrices=verified,
        required_bytes=remaining_reservation(credit),
        passed=result['available_bytes']>=remaining_reservation(credit) and 'State=UP' in result['partition'],
        original_20_percent_storage_reservation_retained=True)
    write(REPORT/'gpu_preflight_transfer_r3_latest.json',result,replace=True)
    return result
