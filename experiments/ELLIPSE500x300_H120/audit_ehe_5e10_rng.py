"""Read-only actual seed/state identity for the 1000 concurrently launched workers."""
import json, time
from pathlib import Path
from ehe_common import read, write, digest
from ehe_5e10_workflow import DATA, REPORT, CPU_BASE, connection, command, q


def audit():
    registry=read(DATA/'simulation/jobs.json')
    code=r'''from pathlib import Path
import hashlib,json,re
root=Path(ROOT)
rows=[]
for i in range(1000):
    raw=(root/f"worker_{i:04d}/transport.log").read_bytes()
    text=raw.decode()
    seed=int(re.search(r"EHE random seed: (\d+) \(from EHE_RANDOM_SEED\)",text)[1])
    state=re.search(r"Initial seed \(index\) = (\d+)\s+Current couple of seeds = (\d+), (\d+)",text)
    rows.append(dict(index=i,explicit_seed=seed,engine_index=int(state[1]),
                     initial_seed_pair=[int(state[2]),int(state[3])],
                     startup_log_snapshot_sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(rows))
'''.replace('ROOT',repr(CPU_BASE+'/transport'))
    with connection('maty') as c:
        rows=json.loads(command(c,'python3 -c '+q(code),120))
    if len(rows)!=1000 or any(v['index']!=i or v['explicit_seed']!=registry['jobs'][i]['seed'] for i,v in enumerate(rows)):
        raise ValueError('Actual printed seeds differ from new registry')
    pairs={tuple(v['initial_seed_pair']) for v in rows}
    if len(pairs)!=1000:raise ValueError('Repeated actual initial random-engine state')
    proof=dict(passed=True,checked_epoch=time.time(),workers=1000,actual_unique_seed_pairs=len(pairs),
        seed_first=33100101,seed_last=33101100,rows=rows,
        simulation_registry_sha256=digest(DATA/'simulation/jobs.json'),audit_source_sha256=digest(Path(__file__)),
        method='Read actual startup log seed/state; no random draws, simulation or source modification',
        limitation='Unique actual initial states exclude identical starting streams; not a mathematical proof of statistical independence. Log hashes are running snapshots, not final receipt hashes.')
    write(REPORT/'source_rng_startup_audit.json',proof)
    print('ACTUAL1000_SEEDS_AND_DISTINCT_RANECU_INITIAL_PAIRS_PASS',flush=True)


if __name__=='__main__':audit()
