"""Bounded physical compact-reader/interface diagnostic; no S2 or imaging."""
import argparse
import json
from pathlib import Path
import numpy as np
from compton_tiled_a_field import CompactCartesianTiles
from generate_compton_a_guard import digest,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('pilot','field','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();fm=json.loads((a.field/'field_manifest.json').read_text());scales=fm['calibration_scales']
    f=CompactCartesianTiles.from_pilot(a.pilot,scales)
    xyz=np.array([[x,246,z] for x in (-4.5,0,4.5) for z in (1.5,6,10.5)])
    left=xyz.copy();left[:,1]-=1e-9;right=xyz.copy();right[:,1]+=1e-9
    cache=[f.cache(q) for q in (left,xyz,right)]
    # Fixed distributed detector-row samples, independent of NEMA image peaks.
    crystals=np.unique(np.linspace(0,10495,32,dtype=int));records=[]
    for crystal in crystals:
        values=[f.evaluate(int(crystal),c) for c in cache];middle=values[1]
        denominator=max(float(middle.max()),1e-30)
        error=max(float(np.max(abs(v-middle))) for v in (values[0],values[2]))/denominator
        records.append(dict(crystal_selected_index=int(crystal),maximum_interface_jump_over_sample_peak=error))
    passed=max(r['maximum_interface_jump_over_sample_peak'] for r in records)<1e-7
    result=dict(status='COMPACT_READER_CONTINUITY_PASSED_ACCURACY_HOLD' if passed else 'HOLD_COMPACT_READER_CONTINUITY',
        pilot_gate_sha256=digest(a.pilot/'tile_pilot_gate.json'),field_manifest_sha256=digest(a.field/'field_manifest.json'),
        reader_sha256=digest(Path(__file__).with_name('compton_tiled_a_field.py')),source_sha256=digest(Path(__file__)),
        rows=len(crystals),interface_locations=len(xyz),epsilon_mm=1e-9,records=records,
        maximum_jump_over_sample_peak=max(r['maximum_interface_jump_over_sample_peak'] for r in records),
        shared_vertex_owner='lexicographically first available tile index, independent of file/load order',
        calibration_applied_once=True,reconstruction_permitted=False,S2_generated=False,
        interpretation='Interface continuity and bitwise compact extraction only; no claim of full A accuracy or spike reduction.')
    write(a.output,result);print(json.dumps(result,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
