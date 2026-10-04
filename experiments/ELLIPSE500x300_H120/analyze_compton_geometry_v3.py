"""Stable R1 full-candidate scan and independent matched-S gates.

Consumes immutable first-scatter transport; writes only to an isolated output.
"""
import argparse
import json
from pathlib import Path
from analyze_first_scatter import analyze,digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','factors','geometry','output','study-config'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();config=json.loads(a.study_config.read_text())
    if config['study']!='compton_response_geometry_v3' or config['geometry_mode']!='stable_float64':
        raise ValueError('Unexpected frozen response study')
    if digest(a.geometry)!=config['geometry_sha256']:
        raise ValueError('Geometry differs from frozen plan')
    if a.output.resolve().is_relative_to(a.inputs.resolve()):
        raise ValueError('Never write into immutable transport inputs')
    a.geometry_mode='stable_float64';a.groups=('ideal',);a.study_name=config['study']
    gate=analyze(a)
    gate['study_config_sha256']=digest(a.study_config)
    (a.output/'validation_gate.json').write_text(json.dumps(gate,indent=2)+'\n')

if __name__=='__main__':main()
