"""Compare every saved CPU/device integral in the bounded 128-event preflight."""
import argparse
import json
from pathlib import Path
import numpy as np
from generate_compton_a_guard import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cpu','device','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();left=json.loads((a.cpu/'benchmark_gate.json').read_text());right=json.loads((a.device/'benchmark_gate.json').read_text())
    for key in ('geometry_sha256','config_sha256','measure_manifest_sha256','R1_scan_gate_sha256','kernel_sha256','field_provenance'):
        if left[key]!=right[key]:raise ValueError('Response contract differs: '+key)
    lc={(c['view'],tuple(c['order'])):c for c in left['cases']}
    rc={(c['view'],tuple(c['order'])):c for c in right['cases']}
    if lc.keys()!=rc.keys() or len(lc)!=6:raise ValueError('Preflight view/order set differs')
    results=[]
    for (view,order),c in lc.items():
        d=rc[(view,order)];oi=((8,4,4),(16,8,8),(32,12,12)).index(order)
        name=f'v{view:02d}_order{oi}.npz';lp=a.cpu/name;rp=a.device/name
        if digest(lp)!=c['integral_file_sha256'] or digest(rp)!=d['integral_file_sha256']:
            raise ValueError('Saved integral differs from receipt')
        u=np.load(lp);v=np.load(rp)
        if not np.array_equal(u['raw_rows'],v['raw_rows']) or not np.array_equal(u['partial_indices'],v['partial_indices']):
            raise ValueError('Event or cell identities differ')
        if len(u['raw_rows'])!=64 or len(u['partial_indices'])!=6880:raise ValueError('Full bounded batch differs')
        z=np.asarray(c['common_R1_reference']);fields={}
        for field in ('object_integrals','reference_integrals'):
            old=u[field];new=v[field]
            if old.shape!=(64,6880) or new.shape!=old.shape:raise ValueError('Complete integral shape differs')
            if not np.isfinite(new).all() or np.any(new<0):raise ValueError('Invalid device response')
            delta=abs(new-old);passed=delta<=1e-10*abs(old)+1e-14*z[:,None]
            fields[field]=dict(entries=passed.size,passed=int(passed.sum()),
                maximum_relative_error=float((delta/np.maximum(abs(old),1e-10*z[:,None])).max()),
                maximum_absolute_error_over_common_Z=float((delta/z[:,None]).max()))
        dz=abs(u['mixed_Z']-v['mixed_Z'])
        norm_ok=bool(np.all(dz<=1e-10*abs(u['mixed_Z'])+1e-14*z))
        results.append(dict(view=view,order=list(order),events=64,fields=fields,mixed_normalization_passed=norm_ok,
            cpu_sha256=digest(lp),device_sha256=digest(rp)))
    passed=all(r['mixed_normalization_passed'] and all(v['entries']==v['passed'] for v in r['fields'].values()) for r in results)
    result=dict(status='PASSED_NUMERICAL_EQUIVALENCE_ONLY' if passed else 'HOLD_BACKEND_DIFFERENCE',cases=results,
        total_integral_entries=sum(v['entries'] for r in results for v in r['fields'].values()),
        passed_integral_entries=sum(v['passed'] for r in results for v in r['fields'].values()),
        cpu_gate_sha256=digest(a.cpu/'benchmark_gate.json'),device_gate_sha256=digest(a.device/'benchmark_gate.json'),
        code_sha256=digest(Path(__file__)),reconstruction_permitted=False,S2_generated=False,
        interpretation='Numerical CPU/device equivalence on 128 fixed events, all 6880 cells and three orders; '
                       'does not validate global A interpolation or constitute reconstruction.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
