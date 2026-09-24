"""Assess pilot layer statistics and response scales without installing Factors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from calibrate import NAMES,layer_scales
from detector_csv import load_detector_coordinates


def assess(raw_root,data_root):
    report={}
    for response,name in NAMES.items():
        source=raw_root/name
        energy=218 if response=='A218' else 440
        window=440 if response=='A440' else 218
        metadata=json.loads((data_root/'collections'/f'calibration_{energy}_pilot.json').read_text())
        primary=metadata['primary_counts']
        if primary[2] or primary[0 if energy==440 else 1]:
            raise ValueError('Pilot calibration must be monoenergetic')
        observed=np.loadtxt(data_root/f'CntStat/{window}keV_RotateNum20_Geant4JSCC/CntStat_calibration_{energy}_pilot.csv',delimiter=',').reshape(-1)
        volumes=np.fromfile(source/'polar_cell_volume_mm3.float64',dtype='<f8')
        detector=load_detector_coordinates(source/'Detector.csv',10496)
        matrix=np.memmap(source/'SysMat_polar',dtype='<f4',mode='r',shape=(len(volumes),10496))
        _,metrics=layer_scales(matrix,volumes,detector,observed,sum(primary))
        report[response]={'primary_count':sum(primary),'layers':metrics}
        del matrix
    report['production_calibration_installed']=False
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=assess(args.raw_root,args.data_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
