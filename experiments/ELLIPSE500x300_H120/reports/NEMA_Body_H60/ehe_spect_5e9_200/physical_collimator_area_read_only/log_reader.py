import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(here))
from ehe_common import DATA, REPORT, GPU_BASE, RESPONSES, digest, read, write
from ehe_5e9_workflow import connection

target = DATA/'collimator_area_read_only'
target.mkdir(exist_ok=True)
assert not (target/'original_log_binding.json').exists(), 'Do not repeat accepted log extraction'
records = []
with connection('gpu') as c, c.open_sftp() as s:
    for response in RESPONSES:
        for slab in range(4):
            receipt = REPORT/'response_preservation_1672966'/response/f'slab_{slab}'/'receipt.json'
            expected = read(receipt)['files']['scatter.log']
            path = GPU_BASE+f'/responses/{response}/slab_{slab}/scatter.log'
            size = s.stat(path).st_size
            assert size <= 2*1024**2
            with s.open(path, 'rb') as f:
                raw = f.read()
            assert len(raw) == size and hashlib.sha256(raw).hexdigest() == expected
            lines = [line for line in raw.decode().splitlines() if line.startswith('Collimator layer ')]
            assert len(lines) == 1
            local = target/f'{response}_slab_{slab}_scatter.log'
            local.write_bytes(raw)
            assert digest(local) == expected
            records.append(dict(response=response,slab=slab,remote_path=path,bytes=size,
                original_scatter_log_sha256=expected,pre_stop_receipt_sha256=digest(receipt),
                collimator_area_lines=lines))
write(target/'original_log_binding.json',dict(passed=True,original_job=1672966,
    scope='New extraction of original area-sampling log lines only; no stage re-verification or computation',
    records=records,reader_code_sha256=digest(__file__)))
print(json.dumps([dict(response=r['response'],slab=r['slab'],lines=r['collimator_area_lines']) for r in records]))
