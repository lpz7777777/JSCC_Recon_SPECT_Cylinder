"""Strict scheduler/receipt identity for independently allocated CPU workers."""
import re

def array_accounting(text, parent, workers=1000):
    roots, steps, failures = {}, {}, []
    pattern = re.compile(re.escape(str(parent)) + r'_(\d+)(?:\.(.+))?$')
    for line in text.splitlines():
        row = line.strip().split('|')
        if not line.strip(): continue
        if len(row) < 10: raise ValueError('Truncated array accounting row')
        match = pattern.fullmatch(row[0])
        if not match: raise ValueError('Unexpected accounting member: '+row[0])
        index = int(match[1])
        if not 0 <= index < workers: raise ValueError('Unexpected worker index')
        state = row[2].split()[0]
        if state in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'):
            failures.append(row[0]+':'+row[2]+':'+row[3])
        if match[2] is None:
            if index in roots: raise ValueError('Duplicate array worker accounting')
            roots[index] = dict(job_id=row[0],allocation_job=row[1],state=state,exit_code=row[3],
                nodes=int(row[4]),cpus=int(row[5]),max_rss=row[6],elapsed=row[7],alloc_tres=row[8],node_list=row[9])
        else:
            group=steps.setdefault(index,{})
            if match[2] in group: raise ValueError('Duplicate accounting step')
            group[match[2]]=(state,row[3])
    complete = len(roots)==workers and not failures and all(
        r['state']=='COMPLETED' and r['exit_code']=='0:0' and r['nodes']==1 and r['cpus']==1
        and r['allocation_job'].isdigit()
        and {'batch','extern','0'} <= set(steps.get(i,{}))
        and all(s==('COMPLETED','0:0') for s in steps[i].values()) for i,r in roots.items())
    counts={}
    for r in roots.values(): counts[r['state']]=counts.get(r['state'],0)+1
    return dict(passed=complete,parent_job=int(parent),workers=workers,roots={str(i):r for i,r in sorted(roots.items())},
                state_counts=counts,failures=failures,accounting=text)

def array_receipt_identity(receipt, allocation, proof, index):
    if not proof['passed'] or proof['workers']!=1000: raise ValueError('Full array completion proof required')
    root=proof['roots'][str(index)]
    if (str(receipt.get('array_job'))!=str(proof['parent_job']) or receipt.get('array_task')!=index
        or receipt['allocation_job']!=root['allocation_job'] or allocation['job']!=root['allocation_job']):
        raise ValueError('Actual independent allocation identity differs')
    text=allocation['scontrol']
    fields={'JobId':root['allocation_job'],'ArrayJobId':str(proof['parent_job']),'ArrayTaskId':str(index),
            'NumNodes':'1','NumCPUs':'1','NumTasks':'1','CPUs/Task':'1'}
    for name,value in fields.items():
        if not re.search(r'(?<!\S)'+re.escape(name)+'='+re.escape(value)+r'(?=\s|$)',text):
            raise ValueError('Actual one-node one-core allocation differs: '+name)
    return True
