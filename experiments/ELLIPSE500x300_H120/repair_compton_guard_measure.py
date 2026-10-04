"""Finish an interrupted CPU diagnostic without overwriting its immutable A field."""
import argparse
import json
from pathlib import Path
import numpy as np
from generate_compton_a_guard import digest
from prepare_compton_overlap_measure import prepare
from validate_compton_guard_field import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('field','guard','geometry','config','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.field/'field_manifest.json').read_text())
    if digest(a.field/'A_field.float64')!=manifest['A_field_sha256']:raise ValueError('A field changed')
    if digest(a.geometry)!=manifest['baseline_geometry_sha256']:raise ValueError('Frozen geometry changed')
    a.output.mkdir(parents=True,exist_ok=False)
    measure=prepare(a.geometry,a.config,a.output/'precise_measure')
    gate=validate(a.field,a.guard,a.output/'midpoint_gate.json')
    (a.output/'stage_complete.json').write_text(json.dumps(dict(status='DIAGNOSTICS_COMPLETE',
        midpoint_gate=gate['status'],measure_gate=measure['status'],A_field_reused_read_only=True,
        field_manifest_sha256=digest(a.field/'field_manifest.json'),S2_generated=False,
        reconstruction_submitted=False),indent=2)+'\n')


if __name__=='__main__':main()
